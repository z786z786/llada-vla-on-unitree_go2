from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada_vla_common import (
    ACTION_FIELDS,
    CONTROLLED_INSTRUCTIONS,
    KNOWN_TARGET_TYPES,
    KNOWN_TASK_FAMILIES,
    SCHEMA_VERSION,
    VISUAL_TASK_FAMILIES,
    control_action_from_frame,
    discover_session_roots,
    episode_task_metadata,
    infer_task_family,
    instruction_matches_known_template,
    load_episode_derived_labels,
    load_quality_review_index,
    load_json,
    resolved_episode_quality,
    raw_action_from_frame,
    state_from_frame,
)

STATE_ACTION_MAX_DELTA_SECONDS = 1.0
STATE_IMAGE_MAX_DELTA_SECONDS = 1.0
CONTROL_ACTION_LIMITS = {
    "vx": (-1.0, 1.0),
    "vy": (-1.0, 1.0),
    "wz": (-1.0, 1.0),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="验证 Go2 collector session 是否具备训练可用性")
    parser.add_argument("--dataset-root", type=Path, required=True, help="数据集根目录，或包含多个 session 根目录的父目录")
    parser.add_argument("--report-path", type=Path, help="可选：将验证报告写入 JSON 文件的路径")
    parser.add_argument("--quality-review-path", type=Path, help="可选：筛检结果 JSON，quality_label=3 的 episode 会标记为后续处理排除")
    return parser


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _add_issue(bucket: Dict[str, List[str]], severity: str, code: str) -> None:
    bucket[severity].append(code)


def _alignment_deltas(meta: Dict[str, Any], frame_timestamp: float) -> Tuple[float, float]:
    state_timestamp = _safe_float(meta.get("state_timestamp"), frame_timestamp)
    action_timestamp = _safe_float(
        meta.get("control_action_timestamp", meta.get("action_timestamp")),
        frame_timestamp,
    )
    image_timestamp = _safe_float(meta.get("image_timestamp"), frame_timestamp)
    return abs(state_timestamp - action_timestamp), abs(state_timestamp - image_timestamp)


def _validate_episode(
    session_root: Path,
    episode_path: Path,
    session_id: str,
    episode_meta: Dict[str, Any],
    quality_review_index: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    payload = load_json(episode_path)
    episode_id = str(payload.get("episode_id") or episode_path.stem)
    instruction = str(payload.get("instruction") or "")
    scene_id = str(payload.get("scene_id") or "")
    operator_id = str(payload.get("operator_id") or "")
    task_metadata = episode_task_metadata(payload, episode_meta)
    task_family = str(task_metadata.get("task_family") or infer_task_family(instruction))
    target_type = str(task_metadata.get("target_type") or "")
    target_label = str(task_metadata.get("target_label") or "")
    target_description = str(task_metadata.get("target_description") or "")
    derived_labels = load_episode_derived_labels(session_root, episode_id)
    quality_review = resolved_episode_quality(payload, episode_meta, quality_review_index, session_id, episode_id)
    frames = list(payload.get("frames") or [])

    issues = {"fatal": [], "warning": [], "info": []}
    if not instruction:
        _add_issue(issues, "fatal", "missing_instruction")
    elif not instruction_matches_known_template(instruction, task_family):
        _add_issue(issues, "warning", "instruction_template_unknown")
    if not scene_id:
        _add_issue(issues, "fatal", "missing_scene_id")
    if not operator_id:
        _add_issue(issues, "fatal", "missing_operator_id")
    if len(frames) < 2:
        _add_issue(issues, "fatal", "too_few_frames")
    if task_family and task_family not in KNOWN_TASK_FAMILIES:
        _add_issue(issues, "warning", "unknown_task_family")
    if target_type and target_type not in KNOWN_TARGET_TYPES:
        _add_issue(issues, "warning", "unknown_target_type")
    if task_family == "visual_following" and target_type and target_type != "person":
        _add_issue(issues, "warning", "visual_following_target_not_person")
    if task_family in VISUAL_TASK_FAMILIES and not derived_labels:
        _add_issue(issues, "info", "derived_labels_missing")
    if task_family == "legacy_motion" and instruction not in CONTROLLED_INSTRUCTIONS:
        _add_issue(issues, "warning", "legacy_motion_instruction_unrecognized")
    previous_timestamp: Optional[float] = None
    max_state_action_delta = 0.0
    max_state_image_delta = 0.0
    for frame in frames:
        timestamp = _safe_float(frame.get("timestamp"))
        if previous_timestamp is not None and timestamp < previous_timestamp:
            _add_issue(issues, "fatal", "timestamps_not_sorted")
            break
        previous_timestamp = timestamp

        image_rel = str(frame.get("image") or "")
        if not image_rel:
            _add_issue(issues, "fatal", "missing_image")
            break
        if not (session_root / image_rel).exists():
            _add_issue(issues, "fatal", "image_path_missing_on_disk")
            break

        state = state_from_frame(frame)
        if any(field not in state for field in ("vx", "vy", "wz", "yaw")):
            _add_issue(issues, "fatal", "missing_state_fields")
            break

        raw_action = raw_action_from_frame(frame)
        control_action = control_action_from_frame(frame)
        if all(abs(control_action[field]) <= 1e-6 for field in ACTION_FIELDS):
            _add_issue(issues, "info", "contains_zero_control_action_frame")

        for field, (lower, upper) in CONTROL_ACTION_LIMITS.items():
            value = control_action[field]
            if value < lower or value > upper:
                _add_issue(issues, "fatal", f"control_action_out_of_range_{field}")

        if raw_action.get("keys", 0) < 0:
            _add_issue(issues, "warning", "raw_action_keys_negative")

        meta = frame.get("meta") or {}
        state_action_delta, state_image_delta = _alignment_deltas(meta, timestamp)
        max_state_action_delta = max(max_state_action_delta, state_action_delta)
        max_state_image_delta = max(max_state_image_delta, state_image_delta)

    if max_state_action_delta > STATE_ACTION_MAX_DELTA_SECONDS:
        _add_issue(issues, "fatal", "state_action_delta_too_large")
    elif max_state_action_delta > 0.25:
        _add_issue(issues, "warning", "state_action_delta_high")

    if max_state_image_delta > STATE_IMAGE_MAX_DELTA_SECONDS:
        _add_issue(issues, "fatal", "state_image_delta_too_large")
    elif max_state_image_delta > 0.25:
        _add_issue(issues, "warning", "state_image_delta_high")
    if quality_review.get("exclude_from_processing"):
        _add_issue(issues, "info", "quality_filtered")

    return {
        "episode_id": episode_id,
        "schema_version": str(payload.get("schema_version") or ""),
        "instruction": instruction,
        "task_family": task_family,
        "target_type": target_type,
        "target_label": target_label,
        "target_description": target_description,
        "derived_labels": derived_labels,
        "scene_id": scene_id,
        "operator_id": operator_id,
        "original_quality_label": int(quality_review.get("original_quality_label") or 0),
        "quality_label": int(quality_review.get("quality_label") or 0),
        "quality_overridden": bool(quality_review.get("quality_overridden")),
        "quality_notes": str(quality_review.get("quality_notes") or ""),
        "exclude_from_processing": bool(quality_review.get("exclude_from_processing")),
        "num_frames": len(frames),
        "max_state_action_delta_s": max_state_action_delta,
        "max_state_image_delta_s": max_state_image_delta,
        "issues": {name: sorted(set(values)) for name, values in issues.items()},
        "is_trainable": not issues["fatal"],
        "is_selected_for_processing": (not issues["fatal"]) and (not quality_review.get("exclude_from_processing")),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    session_roots = discover_session_roots(dataset_root)
    if not session_roots:
        raise FileNotFoundError(f"no session roots found under {dataset_root}")
    quality_review_index = load_quality_review_index(dataset_root, args.quality_review_path)

    episode_reports: List[Dict[str, Any]] = []
    instruction_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    task_family_counts: Counter[str] = Counter()
    target_type_counts: Counter[str] = Counter()
    target_label_counts: Counter[str] = Counter()
    target_description_counts: Counter[str] = Counter()
    derived_target_side_counts: Counter[str] = Counter()
    derived_target_distance_counts: Counter[str] = Counter()
    quality_label_counts: Counter[str] = Counter()
    instruction_scenes: Dict[str, set] = {}
    instruction_targets: Dict[str, set] = {}
    instruction_sessions: Dict[str, set] = {}

    for session_root in session_roots:
        index_payload = load_json(session_root / "index.json")
        session_id = str(index_payload.get("session_id") or session_root.name)
        for episode_meta in index_payload.get("episodes") or []:
            episode_id = str(episode_meta.get("episode_id") or "")
            if not episode_id:
                continue
            episode_path = session_root / "episodes" / f"{episode_id}.json"
            report = _validate_episode(session_root, episode_path, session_id, episode_meta, quality_review_index)
            report["session_id"] = session_id
            episode_reports.append(report)
            instruction_counts[report["instruction"]] += 1
            scene_counts[report["scene_id"]] += 1
            if report["task_family"]:
                task_family_counts[report["task_family"]] += 1
            if report["target_type"]:
                target_type_counts[report["target_type"]] += 1
            if report["target_label"]:
                target_label_counts[report["target_label"]] += 1
            if report["target_description"]:
                target_description_counts[report["target_description"]] += 1
            derived_labels = dict(report.get("derived_labels") or {})
            side_band = str(derived_labels.get("target_side_band") or "")
            distance_band = str(derived_labels.get("target_distance_band") or "")
            if side_band:
                derived_target_side_counts[side_band] += 1
            if distance_band:
                derived_target_distance_counts[distance_band] += 1
            quality_label_counts[str(report["quality_label"])] += 1
            instruction_scenes.setdefault(report["instruction"], set()).add(report["scene_id"])
            if report["target_label"]:
                instruction_targets.setdefault(report["instruction"], set()).add(report["target_label"])
            elif report["target_description"]:
                instruction_targets.setdefault(report["instruction"], set()).add(report["target_description"])
            elif report["target_type"]:
                instruction_targets.setdefault(report["instruction"], set()).add(report["target_type"])
            elif side_band or distance_band:
                instruction_targets.setdefault(report["instruction"], set()).add(
                    f"{side_band or 'unknown'}:{distance_band or 'unknown'}"
                )
            instruction_sessions.setdefault(report["instruction"], set()).add(report["session_id"])

    global_issues: List[str] = []
    if instruction_counts:
        total = sum(instruction_counts.values())
        most_common = instruction_counts.most_common(1)[0][1]
        if most_common / max(1, total) >= 0.8:
            global_issues.append("instruction_distribution_imbalanced")
    if len([scene for scene in scene_counts if scene]) <= 1:
        global_issues.append("scene_coverage_narrow")
    if not task_family_counts:
        global_issues.append("task_family_metadata_missing")
    if sum(count for family, count in task_family_counts.items() if family in VISUAL_TASK_FAMILIES) == 0:
        global_issues.append("no_visually_necessary_task_family")

    multi_scene_instructions = {
        instruction: sorted(scene for scene in scenes if scene)
        for instruction, scenes in instruction_scenes.items()
        if len({scene for scene in scenes if scene}) >= 2
    }
    multi_target_instructions = {
        instruction: sorted(targets)
        for instruction, targets in instruction_targets.items()
        if len(targets) >= 2
    }
    multi_session_instructions = {
        instruction: sorted(sessions)
        for instruction, sessions in instruction_sessions.items()
        if len(sessions) >= 2
    }
    if not multi_scene_instructions:
        global_issues.append("instruction_scene_overlap_low")
    if not multi_target_instructions:
        global_issues.append("instruction_target_variation_low")

    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "session_count": len(session_roots),
        "episode_count": len(episode_reports),
        "trainable_episode_count": sum(1 for item in episode_reports if item["is_trainable"]),
        "selected_episode_count": sum(1 for item in episode_reports if item["is_selected_for_processing"]),
        "excluded_episode_count": sum(1 for item in episode_reports if item["exclude_from_processing"]),
        "fatal_episode_count": sum(1 for item in episode_reports if item["issues"]["fatal"]),
        "warning_episode_count": sum(1 for item in episode_reports if item["issues"]["warning"]),
        "instruction_counts": dict(sorted(instruction_counts.items())),
        "scene_counts": dict(sorted(scene_counts.items())),
        "task_family_counts": dict(sorted(task_family_counts.items())),
        "target_type_counts": dict(sorted(target_type_counts.items())),
        "target_label_counts": dict(sorted(target_label_counts.items(), key=lambda item: (-item[1], item[0]))),
        "target_description_counts": dict(sorted(target_description_counts.items(), key=lambda item: (-item[1], item[0]))),
        "derived_target_side_counts": dict(sorted(derived_target_side_counts.items(), key=lambda item: (-item[1], item[0]))),
        "derived_target_distance_counts": dict(sorted(derived_target_distance_counts.items(), key=lambda item: (-item[1], item[0]))),
        "quality_label_counts": dict(sorted(quality_label_counts.items(), key=lambda item: item[0])),
        "visual_variation_summary": {
            "scene_count": len([scene for scene in scene_counts if scene]),
            "task_family_count": len(task_family_counts),
            "target_type_count": len(target_type_counts),
            "target_label_count": len(target_label_counts),
            "target_description_count": len(target_description_counts),
            "derived_target_side_count": len(derived_target_side_counts),
            "derived_target_distance_count": len(derived_target_distance_counts),
            "visually_necessary_episode_count": sum(
                1 for item in episode_reports if str(item.get("task_family") or "") in VISUAL_TASK_FAMILIES
            ),
            "instructions_with_multiple_scenes": multi_scene_instructions,
            "instructions_with_multiple_targets": multi_target_instructions,
            "instructions_with_multiple_sessions": multi_session_instructions,
        },
        "global_issues": global_issues,
        "episodes": episode_reports,
    }

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        with args.report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")

    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

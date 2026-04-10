#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada_vla_common import (
    DERIVED_LABELS_DIRNAME,
    SCHEMA_VERSION,
    VISUAL_TASK_FAMILIES,
    control_action_from_frame,
    discover_session_roots,
    dump_json,
    episode_task_metadata,
    load_json,
)

KINEMATIC_LABEL_SOURCE = "collector_kinematic_heuristic_v1"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为 Go2 collector session 生成目标左右/远近派生标签")
    parser.add_argument("--dataset-root", type=Path, required=True, help="单个 session 根目录，或包含多个 session 的父目录")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 derived_labels sidecar")
    parser.add_argument("--max-episodes", type=int, help="仅处理前 N 个 episode，便于冒烟测试")
    parser.add_argument("--initial-window-seconds", type=float, default=1.0, help="估计左右方位时使用的起始时间窗口（秒）")
    parser.add_argument("--activity-threshold", type=float, default=0.05, help="动作活跃阈值，低于该值的帧不参与方位判断")
    parser.add_argument("--side-threshold", type=float, default=0.12, help="左右方位判定阈值")
    parser.add_argument("--distance-near-threshold", type=float, default=0.75, help="累计平面位移小于该值时标为 near")
    parser.add_argument("--distance-mid-threshold", type=float, default=2.0, help="累计平面位移小于该值时标为 mid，否则为 far")
    return parser


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _load_episode_frames(session_root: Path, episode_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    payload = load_json(session_root / "episodes" / f"{episode_id}.json")
    frames = list(payload.get("frames") or [])
    return payload, frames


def _active_frames(
    frames: Sequence[Dict[str, Any]],
    initial_window_seconds: float,
    activity_threshold: float,
) -> List[Tuple[float, Dict[str, float]]]:
    if not frames:
        return []

    first_timestamp = _safe_float(frames[0].get("timestamp"))
    window_deadline = first_timestamp + max(initial_window_seconds, 0.0)
    active: List[Tuple[float, Dict[str, float]]] = []
    for frame in frames:
        timestamp = _safe_float(frame.get("timestamp"))
        if timestamp > window_deadline:
            break
        action = control_action_from_frame(frame)
        magnitude = math.hypot(action["vx"], action["vy"]) + 0.6 * abs(action["wz"])
        if magnitude >= activity_threshold:
            active.append((timestamp, action))
    return active


def _estimate_target_side(
    frames: Sequence[Dict[str, Any]],
    initial_window_seconds: float,
    activity_threshold: float,
    side_threshold: float,
) -> Dict[str, Any]:
    active = _active_frames(frames, initial_window_seconds, activity_threshold)
    if not active:
        return {
            "band": "unknown",
            "confidence": 0.0,
            "score": 0.0,
            "frames_used": 0,
            "reason": "no_active_frames_in_initial_window",
        }

    side_scores = [action["wz"] + 0.35 * action["vy"] for _, action in active]
    side_score = sum(side_scores) / len(side_scores)
    threshold = max(side_threshold, 1e-6)
    if side_score >= threshold:
        band = "left"
    elif side_score <= -threshold:
        band = "right"
    else:
        band = "center"

    confidence = _clamp(abs(side_score) / (threshold * 3.0), 0.0, 1.0)
    if band == "center":
        confidence = _clamp(1.0 - (abs(side_score) / threshold), 0.0, 1.0)
    return {
        "band": band,
        "confidence": confidence,
        "score": side_score,
        "frames_used": len(active),
        "reason": "initial_turn_strafe_bias",
    }


def _integrated_planar_distance(frames: Sequence[Dict[str, Any]]) -> float:
    if len(frames) < 2:
        return 0.0

    distance_m = 0.0
    for current, nxt in zip(frames, frames[1:]):
        current_action = control_action_from_frame(current)
        current_timestamp = _safe_float(current.get("timestamp"))
        next_timestamp = _safe_float(nxt.get("timestamp"))
        dt = _clamp(next_timestamp - current_timestamp, 0.0, 1.0)
        planar_speed = math.hypot(current_action["vx"], current_action["vy"])
        distance_m += planar_speed * dt
    return distance_m


def _estimate_target_distance(
    frames: Sequence[Dict[str, Any]],
    near_threshold: float,
    mid_threshold: float,
) -> Dict[str, Any]:
    if not frames:
        return {
            "band": "unknown",
            "confidence": 0.0,
            "distance_m": 0.0,
            "reason": "no_frames",
        }

    near_threshold = max(near_threshold, 0.0)
    mid_threshold = max(mid_threshold, near_threshold)
    distance_m = _integrated_planar_distance(frames)
    if distance_m <= near_threshold:
        band = "near"
        gap = near_threshold - distance_m
        scale = max(near_threshold, 0.25)
    elif distance_m <= mid_threshold:
        band = "mid"
        gap = min(distance_m - near_threshold, mid_threshold - distance_m)
        scale = max(mid_threshold - near_threshold, 0.25)
    else:
        band = "far"
        gap = distance_m - mid_threshold
        scale = max(mid_threshold, 0.5)

    confidence = _clamp(gap / scale, 0.1, 1.0)
    return {
        "band": band,
        "confidence": confidence,
        "distance_m": distance_m,
        "reason": "integrated_planar_distance",
    }


def _derive_labels_for_episode(
    session_id: str,
    episode_id: str,
    payload: Dict[str, Any],
    frames: Sequence[Dict[str, Any]],
    *,
    initial_window_seconds: float,
    activity_threshold: float,
    side_threshold: float,
    distance_near_threshold: float,
    distance_mid_threshold: float,
) -> Dict[str, Any]:
    instruction = str(payload.get("instruction") or "")
    task_metadata = episode_task_metadata(payload, {})
    task_family = str(task_metadata.get("task_family") or "")

    notes: List[str] = []
    side = {
        "band": "unknown",
        "confidence": 0.0,
        "score": 0.0,
        "frames_used": 0,
        "reason": "non_visual_task_family",
    }
    distance = {
        "band": "unknown",
        "confidence": 0.0,
        "distance_m": 0.0,
        "reason": "non_visual_task_family",
    }

    if task_family in VISUAL_TASK_FAMILIES:
        side = _estimate_target_side(frames, initial_window_seconds, activity_threshold, side_threshold)
        distance = _estimate_target_distance(frames, distance_near_threshold, distance_mid_threshold)
    else:
        notes.append("legacy_motion_or_unknown_task_family_kept_as_unknown")

    if side["band"] == "unknown":
        notes.append(side["reason"])
    if distance["band"] == "unknown":
        notes.append(distance["reason"])

    first_timestamp = _safe_float(frames[0].get("timestamp")) if frames else 0.0
    last_timestamp = _safe_float(frames[-1].get("timestamp")) if frames else 0.0
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "episode_id": episode_id,
        "instruction": instruction,
        "task_family": task_family,
        "target_type": str(task_metadata.get("target_type") or ""),
        "target_description": str(task_metadata.get("target_description") or ""),
        "target_side_band": side["band"],
        "target_distance_band": distance["band"],
        "label_source": KINEMATIC_LABEL_SOURCE,
        "label_confidence": {
            "target_side_band": round(float(side["confidence"]), 4),
            "target_distance_band": round(float(distance["confidence"]), 4),
        },
        "metrics": {
            "frame_count": len(frames),
            "duration_seconds": max(0.0, last_timestamp - first_timestamp),
            "integrated_planar_distance_m": round(float(distance["distance_m"]), 4),
            "initial_side_score": round(float(side["score"]), 4),
            "side_frames_used": int(side.get("frames_used", 0)),
        },
        "thresholds": {
            "initial_window_seconds": float(initial_window_seconds),
            "activity_threshold": float(activity_threshold),
            "side_threshold": float(side_threshold),
            "distance_near_threshold": float(distance_near_threshold),
            "distance_mid_threshold": float(distance_mid_threshold),
        },
        "notes": sorted(set(note for note in notes if note)),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    session_roots = discover_session_roots(dataset_root)
    if not session_roots:
        raise FileNotFoundError(f"no session roots found under {dataset_root}")

    processed = 0
    skipped_existing = 0
    session_count = 0

    for session_root in session_roots:
        index_payload = load_json(session_root / "index.json")
        session_id = str(index_payload.get("session_id") or session_root.name)
        derived_dir = session_root / DERIVED_LABELS_DIRNAME
        derived_dir.mkdir(parents=True, exist_ok=True)
        session_count += 1

        for episode_meta in index_payload.get("episodes") or []:
            if args.max_episodes is not None and processed >= args.max_episodes:
                break
            episode_id = str(episode_meta.get("episode_id") or "")
            if not episode_id:
                continue
            output_path = derived_dir / f"{episode_id}.json"
            if output_path.exists() and not args.overwrite:
                skipped_existing += 1
                continue

            payload, frames = _load_episode_frames(session_root, episode_id)
            derived_labels = _derive_labels_for_episode(
                session_id,
                episode_id,
                payload,
                frames,
                initial_window_seconds=args.initial_window_seconds,
                activity_threshold=args.activity_threshold,
                side_threshold=args.side_threshold,
                distance_near_threshold=args.distance_near_threshold,
                distance_mid_threshold=args.distance_mid_threshold,
            )
            dump_json(output_path, derived_labels)
            processed += 1

        if args.max_episodes is not None and processed >= args.max_episodes:
            break

    print(
        f"已处理 {processed} 个 episode，session 数 {session_count}，"
        f"已跳过现有 sidecar {skipped_existing} 个；输出目录名：{DERIVED_LABELS_DIRNAME}"
    )


if __name__ == "__main__":
    main()

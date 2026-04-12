#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada_vla_common import (
    SCHEMA_VERSION,
    VISUAL_TASK_FAMILIES,
    control_action_from_frame,
    discover_session_roots,
    dump_json,
    episode_task_metadata,
    load_json,
    load_quality_review_index,
    resolved_episode_quality,
    state_from_frame,
)

LABEL_SOURCE = "collector_distribution_heuristic_v1"
DEFAULT_OUTPUT_DIRNAME = "distribution_labels"
SIDE_ZH_LABELS = {
    "left": "左",
    "center": "中",
    "right": "右",
    "unknown": "未知",
}
DISTANCE_ZH_LABELS = {
    "near": "近",
    "mid": "中",
    "far": "远",
    "unknown": "未知",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 episode 属性分布打左右/远近标签")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="单个 session 根目录，或包含多个 session 的父目录",
    )
    parser.add_argument(
        "--output-dirname",
        default=DEFAULT_OUTPUT_DIRNAME,
        help=f"标签 sidecar 输出目录名，默认 {DEFAULT_OUTPUT_DIRNAME}",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的标签 sidecar")
    parser.add_argument("--max-episodes", type=int, help="仅处理前 N 个 episode，便于冒烟测试")
    parser.add_argument(
        "--side-metric",
        choices=["lateral_displacement", "heading_change"],
        default="lateral_displacement",
        help="左右标签使用的排序属性",
    )
    parser.add_argument(
        "--distance-metric",
        choices=["integrated_planar_distance", "forward_displacement"],
        default="integrated_planar_distance",
        help="远近标签使用的排序属性",
    )
    parser.add_argument(
        "--invert-side-sign",
        action="store_true",
        help="将左右标签翻转；当本地坐标定义与实际语义相反时使用",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        help="可选的汇总 JSON 输出路径，用于记录阈值与标签分布",
    )
    parser.add_argument(
        "--quality-review-path",
        type=Path,
        help="可选：筛检结果 JSON，quality_label=3 的 episode 会被跳过",
    )
    return parser


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    q = _clamp(q, 0.0, 1.0)
    position = q * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + (upper_value - lower_value) * weight


def _percentile_rank(value: float, values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    lower_equal = sum(1 for item in ordered if item <= value)
    return lower_equal / len(ordered)


def _load_episode_frames(session_root: Path, episode_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    payload = load_json(session_root / "episodes" / f"{episode_id}.json")
    return payload, list(payload.get("frames") or [])


def _trajectory_metrics(frames: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if len(frames) < 2:
        return {
            "frame_count": float(len(frames)),
            "duration_seconds": 0.0,
            "integrated_planar_distance_m": 0.0,
            "forward_displacement_m": 0.0,
            "lateral_displacement_m": 0.0,
            "heading_change_rad": 0.0,
        }

    first_timestamp = _safe_float(frames[0].get("timestamp"))
    last_timestamp = _safe_float(frames[-1].get("timestamp"))
    initial_state = state_from_frame(frames[0])
    initial_yaw = initial_state["yaw"]

    integrated_distance_m = 0.0
    forward_displacement_m = 0.0
    lateral_displacement_m = 0.0

    for current_frame, next_frame in zip(frames, frames[1:]):
        action = control_action_from_frame(current_frame)
        current_state = state_from_frame(current_frame)
        dt = _clamp(_safe_float(next_frame.get("timestamp")) - _safe_float(current_frame.get("timestamp")), 0.0, 1.0)
        vx = action["vx"]
        vy = action["vy"]
        integrated_distance_m += math.hypot(vx, vy) * dt

        relative_yaw = _wrap_angle(current_state["yaw"] - initial_yaw)
        forward_displacement_m += (vx * math.cos(relative_yaw) - vy * math.sin(relative_yaw)) * dt
        lateral_displacement_m += (vx * math.sin(relative_yaw) + vy * math.cos(relative_yaw)) * dt

    final_state = state_from_frame(frames[-1])
    return {
        "frame_count": float(len(frames)),
        "duration_seconds": max(0.0, last_timestamp - first_timestamp),
        "integrated_planar_distance_m": integrated_distance_m,
        "forward_displacement_m": forward_displacement_m,
        "lateral_displacement_m": lateral_displacement_m,
        "heading_change_rad": _wrap_angle(final_state["yaw"] - initial_yaw),
    }


def _side_band_from_score(score: float, low_threshold: float, high_threshold: float) -> str:
    if score <= low_threshold:
        return "right"
    if score >= high_threshold:
        return "left"
    return "center"


def _distance_band_from_score(score: float, low_threshold: float, high_threshold: float) -> str:
    if score <= low_threshold:
        return "near"
    if score >= high_threshold:
        return "far"
    return "mid"


def _confidence_from_band(score: float, band: str, low_threshold: float, high_threshold: float) -> float:
    spread = max(high_threshold - low_threshold, 1e-6)
    if band in {"left", "far"}:
        return _clamp((score - high_threshold) / spread + 0.35, 0.05, 1.0)
    if band in {"right", "near"}:
        return _clamp((low_threshold - score) / spread + 0.35, 0.05, 1.0)

    midpoint = (low_threshold + high_threshold) * 0.5
    half_span = max((high_threshold - low_threshold) * 0.5, 1e-6)
    return _clamp(1.0 - abs(score - midpoint) / half_span, 0.05, 1.0)


def _metric_value(metrics: Dict[str, float], metric_name: str) -> float:
    metric_map = {
        "lateral_displacement": metrics["lateral_displacement_m"],
        "heading_change": metrics["heading_change_rad"],
        "integrated_planar_distance": metrics["integrated_planar_distance_m"],
        "forward_displacement": metrics["forward_displacement_m"],
    }
    return float(metric_map[metric_name])


def _dataset_summary(
    side_scores: Sequence[float],
    distance_scores: Sequence[float],
    side_metric: str,
    distance_metric: str,
) -> Dict[str, Any]:
    side_low = _quantile(side_scores, 1.0 / 3.0)
    side_high = _quantile(side_scores, 2.0 / 3.0)
    distance_low = _quantile(distance_scores, 1.0 / 3.0)
    distance_high = _quantile(distance_scores, 2.0 / 3.0)
    return {
        "side_metric": side_metric,
        "distance_metric": distance_metric,
        "episode_count": len(side_scores),
        "side_thresholds": {
            "right_upper_bound": side_low,
            "left_lower_bound": side_high,
        },
        "distance_thresholds": {
            "near_upper_bound": distance_low,
            "far_lower_bound": distance_high,
        },
    }


def generate_distribution_labels(
    dataset_root: Path,
    *,
    output_dirname: str = DEFAULT_OUTPUT_DIRNAME,
    overwrite: bool = False,
    max_episodes: Optional[int] = None,
    side_metric: str = "lateral_displacement",
    distance_metric: str = "integrated_planar_distance",
    invert_side_sign: bool = False,
    summary_path: Optional[Path] = None,
    quality_review_index: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
    quality_review_path: Optional[Path] = None,
) -> Dict[str, Any]:
    dataset_root = dataset_root.resolve()
    session_roots = discover_session_roots(dataset_root)
    if not session_roots:
        raise FileNotFoundError(f"no session roots found under {dataset_root}")
    if quality_review_index is None:
        quality_review_index = load_quality_review_index(dataset_root, quality_review_path)

    episode_records: List[Dict[str, Any]] = []
    skipped_quality_filtered = 0
    skipped_non_visual = 0
    skipped_existing = 0
    skipped_missing_frames = 0

    for session_root in session_roots:
        index_payload = load_json(session_root / "index.json")
        session_id = str(index_payload.get("session_id") or session_root.name)
        output_dir = session_root / output_dirname
        output_dir.mkdir(parents=True, exist_ok=True)

        for episode_meta in index_payload.get("episodes") or []:
            if max_episodes is not None and len(episode_records) >= max_episodes:
                break

            episode_id = str(episode_meta.get("episode_id") or "")
            if not episode_id:
                continue

            output_path = output_dir / f"{episode_id}.json"
            if output_path.exists() and not overwrite:
                skipped_existing += 1
                continue

            payload, frames = _load_episode_frames(session_root, episode_id)
            quality_review = resolved_episode_quality(payload, episode_meta, quality_review_index, session_id, episode_id)
            if quality_review.get("exclude_from_processing"):
                skipped_quality_filtered += 1
                continue
            task_metadata = episode_task_metadata(payload, episode_meta)
            task_family = str(task_metadata.get("task_family") or "")
            if task_family not in VISUAL_TASK_FAMILIES:
                skipped_non_visual += 1
                continue
            if len(frames) < 2:
                skipped_missing_frames += 1
                continue

            metrics = _trajectory_metrics(frames)
            side_score = _metric_value(metrics, side_metric)
            if invert_side_sign:
                side_score *= -1.0
            distance_score = _metric_value(metrics, distance_metric)

            episode_records.append(
                {
                    "session_root": session_root,
                    "session_id": session_id,
                    "episode_id": episode_id,
                    "payload": payload,
                    "task_metadata": task_metadata,
                    "metrics": metrics,
                    "side_score": side_score,
                    "distance_score": distance_score,
                    "output_path": output_path,
                }
            )

        if max_episodes is not None and len(episode_records) >= max_episodes:
            break

    if not episode_records:
        raise RuntimeError("no eligible visual episodes found for distribution labeling")

    side_scores = [float(item["side_score"]) for item in episode_records]
    distance_scores = [float(item["distance_score"]) for item in episode_records]
    summary = _dataset_summary(side_scores, distance_scores, side_metric, distance_metric)
    side_low = float(summary["side_thresholds"]["right_upper_bound"])
    side_high = float(summary["side_thresholds"]["left_lower_bound"])
    distance_low = float(summary["distance_thresholds"]["near_upper_bound"])
    distance_high = float(summary["distance_thresholds"]["far_lower_bound"])

    side_counter: Counter[str] = Counter()
    distance_counter: Counter[str] = Counter()

    for item in episode_records:
        payload = item["payload"]
        task_metadata = item["task_metadata"]
        metrics = item["metrics"]
        side_score = float(item["side_score"])
        distance_score = float(item["distance_score"])
        target_side_band = _side_band_from_score(side_score, side_low, side_high)
        target_distance_band = _distance_band_from_score(distance_score, distance_low, distance_high)
        side_counter[target_side_band] += 1
        distance_counter[target_distance_band] += 1

        derived_labels = {
            "schema_version": SCHEMA_VERSION,
            "session_id": item["session_id"],
            "episode_id": item["episode_id"],
            "instruction": str(payload.get("instruction") or ""),
            "task_family": str(task_metadata.get("task_family") or ""),
            "target_type": str(task_metadata.get("target_type") or ""),
            "target_label": str(task_metadata.get("target_label") or ""),
            "target_description": str(task_metadata.get("target_description") or ""),
            "target_side_band": target_side_band,
            "target_distance_band": target_distance_band,
            "target_side_label_zh": SIDE_ZH_LABELS[target_side_band],
            "target_distance_label_zh": DISTANCE_ZH_LABELS[target_distance_band],
            "target_band_description_zh": (
                f"{SIDE_ZH_LABELS[target_side_band]}-{DISTANCE_ZH_LABELS[target_distance_band]}"
            ),
            "label_source": LABEL_SOURCE,
            "label_method": "dataset_quantile_distribution",
            "label_confidence": {
                "target_side_band": round(_confidence_from_band(side_score, target_side_band, side_low, side_high), 4),
                "target_distance_band": round(
                    _confidence_from_band(distance_score, target_distance_band, distance_low, distance_high),
                    4,
                ),
            },
            "metrics": {
                "frame_count": int(metrics["frame_count"]),
                "duration_seconds": round(metrics["duration_seconds"], 4),
                "integrated_planar_distance_m": round(metrics["integrated_planar_distance_m"], 4),
                "forward_displacement_m": round(metrics["forward_displacement_m"], 4),
                "lateral_displacement_m": round(metrics["lateral_displacement_m"], 4),
                "heading_change_rad": round(metrics["heading_change_rad"], 4),
                "side_score": round(side_score, 4),
                "distance_score": round(distance_score, 4),
                "side_percentile": round(_percentile_rank(side_score, side_scores), 4),
                "distance_percentile": round(_percentile_rank(distance_score, distance_scores), 4),
            },
            "thresholds": {
                "side_metric": side_metric,
                "distance_metric": distance_metric,
                "side_right_upper_bound": round(side_low, 4),
                "side_left_lower_bound": round(side_high, 4),
                "distance_near_upper_bound": round(distance_low, 4),
                "distance_far_lower_bound": round(distance_high, 4),
                "invert_side_sign": bool(invert_side_sign),
                "dataset_episode_count": len(episode_records),
            },
            "notes": [],
        }
        dump_json(item["output_path"], derived_labels)

    summary["side_band_counts"] = dict(sorted(side_counter.items()))
    summary["distance_band_counts"] = dict(sorted(distance_counter.items()))
    summary["skipped_quality_filtered"] = skipped_quality_filtered
    summary["skipped_existing"] = skipped_existing
    summary["skipped_non_visual"] = skipped_non_visual
    summary["skipped_missing_frames"] = skipped_missing_frames

    if summary_path is not None:
        dump_json(summary_path.resolve(), summary)

    return summary


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = generate_distribution_labels(
        args.dataset_root,
        output_dirname=args.output_dirname,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
        side_metric=args.side_metric,
        distance_metric=args.distance_metric,
        invert_side_sign=args.invert_side_sign,
        summary_path=args.summary_path,
        quality_review_path=args.quality_review_path,
    )
    print(
        f"已写入 {summary['episode_count']} 个 episode 的分布标签，"
        f"side={summary['side_band_counts']}，"
        f"distance={summary['distance_band_counts']}，"
        f"输出目录名：{args.output_dirname}"
    )


if __name__ == "__main__":
    main()

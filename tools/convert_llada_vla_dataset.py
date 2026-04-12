from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada_vla_common import (
    SCHEMA_VERSION,
    Sample,
    assign_splits,
    discover_session_roots,
    dump_jsonl,
    ensure_session_materialized,
    load_quality_review_index,
    load_session_samples,
    summarize_trajectory_actions,
    summarize_trajectory_metric_series,
)
from derive_distribution_labels import generate_distribution_labels


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 Go2 collector session 转换为 train/val/test 清单")
    parser.add_argument("--raw-root", type=Path, default=Path(__file__).resolve().parent.parent / "data", help="包含原始 session 目录的根路径")
    parser.add_argument("--output-root", type=Path, required=True, help="转换后清单的输出目录")
    parser.add_argument("--action-horizon", type=int, default=4, help="每个 action chunk 中包含的未来动作步数")
    parser.add_argument("--min-episode-length", type=int, default=1, help="转换前丢弃长度小于该值的 episode")
    parser.add_argument("--split-mode", choices=["auto", "by_session", "by_trajectory"], default="auto", help="train/val/test 划分方式")
    parser.add_argument("--split-seed", type=int, help="可选：划分前随机打散 session/trajectory 分组的随机种子")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument(
        "--derive-labels",
        choices=["none", "distribution"],
        default="distribution",
        help="转换前是否自动生成 derived_labels",
    )
    parser.add_argument(
        "--derive-side-metric",
        choices=["lateral_displacement", "heading_change"],
        default="lateral_displacement",
        help="自动生成分布标签时使用的左右属性",
    )
    parser.add_argument(
        "--derive-distance-metric",
        choices=["integrated_planar_distance", "forward_displacement"],
        default="integrated_planar_distance",
        help="自动生成分布标签时使用的远近属性",
    )
    parser.add_argument(
        "--derive-invert-side-sign",
        action="store_true",
        help="自动生成分布标签时翻转左右方向",
    )
    parser.add_argument(
        "--derive-summary-path",
        type=Path,
        help="自动生成分布标签时输出汇总 JSON 的路径",
    )
    parser.add_argument(
        "--quality-review-path",
        type=Path,
        help="可选：筛检结果 JSON，quality_label=3 的 episode 会被排除",
    )
    parser.add_argument("--overwrite", action="store_true", help="写入前删除已有输出目录")
    return parser


def _sort_key(sample: Sample) -> Tuple[str, int, int, int]:
    return (sample.session_id, sample.trajectory_index, sample.step_id, sample.trajectory_step_index)


def _record_for_sample(sample: Sample, output_root: Path, split_name: str) -> Dict[str, Any]:
    record = sample.to_manifest_record(output_root)
    record["split"] = split_name
    return record


def _stats_payload(output_root: Path, raw_root: Path, session_roots: Sequence[Path], all_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    split_counter = Counter(str(record.get("split") or "") for record in all_records)
    instruction_counter = Counter(str(record.get("instruction") or "") for record in all_records)
    session_counter = Counter(str(record.get("session_id") or "") for record in all_records)
    trajectory_counter = Counter((str(record.get("session_id") or ""), str(record.get("trajectory_id") or "")) for record in all_records)
    capture_mode_counter = Counter(str(record.get("capture_mode") or "") for record in all_records if str(record.get("capture_mode") or ""))
    task_family_counter = Counter(str(record.get("task_family") or "") for record in all_records if str(record.get("task_family") or ""))
    target_type_counter = Counter(str(record.get("target_type") or "") for record in all_records if str(record.get("target_type") or ""))
    target_label_counter = Counter(str(record.get("target_label") or "") for record in all_records if str(record.get("target_label") or ""))
    quality_label_counter = Counter(str(record.get("quality_label")) for record in all_records if record.get("quality_label") is not None)
    derived_target_side_counter = Counter(
        str((record.get("derived_labels") or {}).get("target_side_band") or "")
        for record in all_records
        if str((record.get("derived_labels") or {}).get("target_side_band") or "")
    )
    derived_target_distance_counter = Counter(
        str((record.get("derived_labels") or {}).get("target_distance_band") or "")
        for record in all_records
        if str((record.get("derived_labels") or {}).get("target_distance_band") or "")
    )

    trajectory_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in all_records:
        key = (str(record.get("session_id") or ""), str(record.get("trajectory_id") or ""))
        trajectory_groups.setdefault(key, []).append(record)

    trajectory_metrics_rows: List[Dict[str, Any]] = []
    trajectory_metrics_rows_only: List[Dict[str, Any]] = []
    for records in trajectory_groups.values():
        ordered = sorted(records, key=lambda item: int(item.get("trajectory_step_index") or 0))
        metrics = summarize_trajectory_actions(
            [dict(item.get("control_action") or {}) for item in ordered],
            [float(item.get("timestamp") or 0.0) for item in ordered],
        )
        metrics["capture_mode"] = str(ordered[0].get("capture_mode") or "")
        trajectory_metrics_rows.append(metrics)
        if metrics["capture_mode"] == "trajectory":
            trajectory_metrics_rows_only.append(metrics)

    return {
        "schema_version": SCHEMA_VERSION,
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "session_count": len(session_roots),
        "trajectory_count": len(trajectory_counter),
        "sample_count": len(all_records),
        "split_counts": dict(sorted(split_counter.items())),
        "session_counts": dict(sorted(session_counter.items())),
        "instruction_counts": dict(sorted(instruction_counter.items(), key=lambda item: (-item[1], item[0]))),
        "capture_mode_counts": dict(sorted(capture_mode_counter.items(), key=lambda item: (-item[1], item[0]))),
        "task_family_counts": dict(sorted(task_family_counter.items(), key=lambda item: (-item[1], item[0]))),
        "target_type_counts": dict(sorted(target_type_counter.items(), key=lambda item: (-item[1], item[0]))),
        "target_label_counts": dict(sorted(target_label_counter.items(), key=lambda item: (-item[1], item[0]))),
        "quality_label_counts": dict(sorted(quality_label_counter.items(), key=lambda item: item[0])),
        "derived_target_side_counts": dict(sorted(derived_target_side_counter.items(), key=lambda item: (-item[1], item[0]))),
        "derived_target_distance_counts": dict(sorted(derived_target_distance_counter.items(), key=lambda item: (-item[1], item[0]))),
        "trajectory_metrics": {
            "frame_count": summarize_trajectory_metric_series([float(item["frame_count"]) for item in trajectory_metrics_rows]),
            "duration_seconds": summarize_trajectory_metric_series([float(item["duration_seconds"]) for item in trajectory_metrics_rows]),
            "action_change_count": summarize_trajectory_metric_series([float(item["action_change_count"]) for item in trajectory_metrics_rows]),
            "stop_ratio": summarize_trajectory_metric_series([float(item["stop_ratio"]) for item in trajectory_metrics_rows]),
            "turn_ratio": summarize_trajectory_metric_series([float(item["turn_ratio"]) for item in trajectory_metrics_rows]),
        },
        "trajectory_mode_metrics": {
            "trajectory_count": len(trajectory_metrics_rows_only),
            "frame_count": summarize_trajectory_metric_series([float(item["frame_count"]) for item in trajectory_metrics_rows_only]),
            "duration_seconds": summarize_trajectory_metric_series([float(item["duration_seconds"]) for item in trajectory_metrics_rows_only]),
            "action_change_count": summarize_trajectory_metric_series([float(item["action_change_count"]) for item in trajectory_metrics_rows_only]),
            "stop_ratio": summarize_trajectory_metric_series([float(item["stop_ratio"]) for item in trajectory_metrics_rows_only]),
            "turn_ratio": summarize_trajectory_metric_series([float(item["turn_ratio"]) for item in trajectory_metrics_rows_only]),
        },
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"输出目录已存在且非空：{output_root}")
        shutil.rmtree(output_root)

    discovered_session_roots = discover_session_roots(raw_root)
    if not discovered_session_roots:
        raise FileNotFoundError(f"在以下路径下没有找到 session 根目录：{raw_root}")
    quality_review_index = load_quality_review_index(raw_root, args.quality_review_path)

    derived_label_summary: Dict[str, Any] = {}
    if args.derive_labels == "distribution":
        derived_label_summary = generate_distribution_labels(
            raw_root,
            output_dirname="derived_labels",
            overwrite=True,
            side_metric=args.derive_side_metric,
            distance_metric=args.derive_distance_metric,
            invert_side_sign=args.derive_invert_side_sign,
            summary_path=args.derive_summary_path,
            quality_review_index=quality_review_index,
        )

    samples: List[Sample] = []
    retained_session_roots: List[Path] = []
    for session_root in discovered_session_roots:
        session_samples = load_session_samples(
            session_root,
            action_horizon=args.action_horizon,
            min_trajectory_length=args.min_episode_length,
            quality_review_index=quality_review_index,
        )
        if not session_samples:
            continue
        samples.extend(session_samples)
        retained_session_roots.append(session_root)

    if not samples:
        raise RuntimeError(f"在以下路径下没有找到样本：{raw_root}")

    split_samples = assign_splits(
        samples,
        split_mode=args.split_mode,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    for session_root in retained_session_roots:
        ensure_session_materialized(output_root, session_root, session_root.name)

    all_records: List[Dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        split_records = [_record_for_sample(sample, output_root, split_name) for sample in sorted(split_samples[split_name], key=_sort_key)]
        dump_jsonl(output_root / f"{split_name}.jsonl", split_records)
        all_records.extend(split_records)

    all_records.sort(key=lambda record: (str(record.get("session_id") or ""), int(record.get("trajectory_index") or 0), int(record.get("step_id") or 0)))
    dump_jsonl(output_root / "dataset.jsonl", all_records)

    stats = _stats_payload(output_root, raw_root, retained_session_roots, all_records)
    stats["min_episode_length"] = args.min_episode_length
    stats["split_mode"] = args.split_mode
    if args.split_seed is not None:
        stats["split_seed"] = args.split_seed
    if derived_label_summary:
        stats["derived_labels_generation"] = derived_label_summary
    with (output_root / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"已转换 {stats['sample_count']} 个样本，来自 {stats['session_count']} retained sessions "
        f"into {output_root}"
    )
    print(f"数据划分统计：{stats['split_counts']}")
    print(f"指令统计：{stats['instruction_counts']}")
    if stats["capture_mode_counts"]:
        print(f"采集模式统计：{stats['capture_mode_counts']}")
    if stats["task_family_counts"]:
        print(f"任务族统计：{stats['task_family_counts']}")
    if stats["target_label_counts"]:
        print(f"语义目标统计：{stats['target_label_counts']}")
    if stats["quality_label_counts"]:
        print(f"筛检标签统计：{stats['quality_label_counts']}")
    if stats["derived_target_side_counts"]:
        print(f"派生左右标签统计：{stats['derived_target_side_counts']}")
    if stats["derived_target_distance_counts"]:
        print(f"派生远近标签统计：{stats['derived_target_distance_counts']}")
    if derived_label_summary:
        print(
            "自动生成标签： "
            f"side={derived_label_summary['side_band_counts']} "
            f"distance={derived_label_summary['distance_band_counts']}"
        )
    print(
        "轨迹统计： "
        f"平均帧数={stats['trajectory_metrics']['frame_count']['mean']:.2f} "
        f"平均时长(秒)={stats['trajectory_metrics']['duration_seconds']['mean']:.2f} "
        f"平均动作切换次数={stats['trajectory_metrics']['action_change_count']['mean']:.2f} "
        f"平均停止占比={stats['trajectory_metrics']['stop_ratio']['mean']:.1%} "
        f"平均转向占比={stats['trajectory_metrics']['turn_ratio']['mean']:.1%}"
    )


if __name__ == "__main__":
    main()

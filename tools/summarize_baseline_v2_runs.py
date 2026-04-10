#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="汇总多个 baseline_v2 实验目录的关键指标")
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        help="单个实验输出目录，可重复传入",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        help="自动扫描其下一层包含 metrics.json 的实验输出目录",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="可选：把 markdown 汇总表写入文件",
    )
    parser.add_argument(
        "--sort-by",
        choices=["test_rmse", "val_rmse", "name"],
        default="test_rmse",
        help="结果排序字段",
    )
    return parser


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _discover_run_dirs(args: argparse.Namespace) -> List[Path]:
    run_dirs: List[Path] = []
    if args.outputs_root is not None:
        outputs_root = args.outputs_root.resolve()
        if outputs_root.exists():
            for child in sorted(outputs_root.iterdir()):
                if child.is_dir() and (child / "metrics.json").exists():
                    run_dirs.append(child)
    for run_dir in args.run_dir or []:
        resolved = run_dir.resolve()
        if resolved not in run_dirs:
            run_dirs.append(resolved)
    return run_dirs


def _best_epoch(training_summary: Dict[str, Any]) -> Optional[int]:
    history = list(training_summary.get("history") or [])
    if not history:
        return None
    best_row = min(
        history,
        key=lambda row: (
            _safe_float(row.get("val_overall_rmse")) if _safe_float(row.get("val_overall_rmse")) is not None else float("inf"),
            int(row.get("epoch") or 0),
        ),
    )
    try:
        return int(best_row.get("epoch"))
    except (TypeError, ValueError):
        return None


def _modalities_label(metrics_payload: Dict[str, Any]) -> str:
    modalities = dict(metrics_payload.get("modalities") or {})
    enabled: List[str] = []
    if modalities.get("text"):
        enabled.append("instruction")
    if modalities.get("image"):
        enabled.append("image")
    if modalities.get("state"):
        enabled.append("state")
    return "+".join(enabled) if enabled else "-"


def _row_from_run_dir(run_dir: Path) -> Dict[str, Any]:
    metrics = _load_json(run_dir / "metrics.json")
    training_summary = _load_json(run_dir / "training_summary.json") if (run_dir / "training_summary.json").exists() else {}
    test_metrics = dict(metrics.get("test_metrics") or {})
    val_metrics = dict(metrics.get("val_metrics") or {})
    mean_test = dict((metrics.get("mean_baseline") or {}).get("test") or {})
    old_linear_test = dict((metrics.get("old_linear_baseline") or {}).get("test") or {})
    split_summary = dict(metrics.get("split_summary") or {})
    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "modalities": _modalities_label(metrics),
        "train_samples": int((split_summary.get("train") or {}).get("sample_count") or 0),
        "val_samples": int((split_summary.get("val") or {}).get("sample_count") or 0),
        "test_samples": int((split_summary.get("test") or {}).get("sample_count") or 0),
        "best_epoch": _best_epoch(training_summary),
        "val_rmse": _safe_float(val_metrics.get("overall_rmse")),
        "test_rmse": _safe_float(test_metrics.get("overall_rmse")),
        "test_mae": _safe_float(test_metrics.get("overall_mae")),
        "mean_test_rmse": _safe_float(mean_test.get("overall_rmse")),
        "old_linear_test_rmse": _safe_float(old_linear_test.get("overall_rmse")),
    }


def _sort_rows(rows: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
    if sort_by == "name":
        return sorted(rows, key=lambda row: str(row["name"]))
    metric_key = "test_rmse" if sort_by == "test_rmse" else "val_rmse"
    return sorted(
        rows,
        key=lambda row: (
            row[metric_key] if row[metric_key] is not None else float("inf"),
            str(row["name"]),
        ),
    )


def _markdown_table(rows: Iterable[Dict[str, Any]]) -> str:
    header = [
        "run",
        "modalities",
        "train",
        "val",
        "test",
        "best_epoch",
        "val_rmse",
        "test_rmse",
        "test_mae",
        "mean_test_rmse",
        "old_linear_test_rmse",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                [
                    str(row["name"]),
                    str(row["modalities"]),
                    str(row["train_samples"]),
                    str(row["val_samples"]),
                    str(row["test_samples"]),
                    str(row["best_epoch"] if row["best_epoch"] is not None else "-"),
                    _format_float(row["val_rmse"]),
                    _format_float(row["test_rmse"]),
                    _format_float(row["test_mae"]),
                    _format_float(row["mean_test_rmse"]),
                    _format_float(row["old_linear_test_rmse"]),
                ]
            ) + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dirs = _discover_run_dirs(args)
    if not run_dirs:
        raise FileNotFoundError("no run directories found")

    rows = [_row_from_run_dir(run_dir) for run_dir in run_dirs]
    rows = _sort_rows(rows, args.sort_by)
    markdown = _markdown_table(rows)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada_vla_common import (
    ACTION_FIELDS,
    Sample,
    assign_splits,
    discover_session_roots,
    feature_vector_from_record,
    fit_standardizer,
    has_converted_manifests,
    load_jsonl,
    load_session_samples,
    standardize_vector,
)

TARGET_FIELDS = list(ACTION_FIELDS)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate a simple Go2 behavior-cloning baseline")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Converted manifest root or raw session root")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument("--text-dim", type=int, default=64, help="Hashed instruction feature dimension")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio for raw sessions")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio for raw sessions")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio for raw sessions")
    parser.add_argument("--min-episode-length", type=int, default=1, help="Drop raw episodes shorter than this many frames")
    parser.add_argument("--split-mode", choices=["auto", "by_session", "by_trajectory"], default="auto", help="Split mode when reading raw sessions")
    parser.add_argument("--save-model", type=Path, help="Optional path to save the fitted baseline weights as JSON")
    return parser


def _target_vector(record: Dict[str, Any]) -> List[float]:
    action = record.get("control_action") or {}
    return [float(action.get(field, 0.0)) for field in TARGET_FIELDS]


def _mean_vector(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    totals = [0.0] * dim
    for vector in vectors:
        for index, value in enumerate(vector):
            totals[index] += float(value)
    return [value / len(vectors) for value in totals]


def _mse_mae(predictions: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> Dict[str, Any]:
    if not targets:
        return {"count": 0, "mae": [], "rmse": [], "overall_mae": 0.0, "overall_rmse": 0.0}

    dim = len(targets[0])
    abs_sums = [0.0] * dim
    sq_sums = [0.0] * dim
    overall_abs = 0.0
    overall_sq = 0.0
    for pred, target in zip(predictions, targets):
        for index in range(dim):
            err = float(pred[index]) - float(target[index])
            abs_sums[index] += abs(err)
            sq_sums[index] += err * err
            overall_abs += abs(err)
            overall_sq += err * err

    count = len(targets)
    mae = [value / count for value in abs_sums]
    rmse = [math.sqrt(value / count) for value in sq_sums]
    return {
        "count": count,
        "mae": mae,
        "rmse": rmse,
        "overall_mae": overall_abs / (count * dim),
        "overall_rmse": math.sqrt(overall_sq / (count * dim)),
    }


@dataclass
class LinearBaseline:
    weights: List[List[float]]
    feature_means: List[float]
    feature_stds: List[float]
    target_means: List[float]
    target_stds: List[float]

    @classmethod
    def create(cls, feature_dim: int, target_dim: int) -> "LinearBaseline":
        return cls(
            weights=[[0.0] * (feature_dim + 1) for _ in range(target_dim)],
            feature_means=[],
            feature_stds=[],
            target_means=[],
            target_stds=[],
        )

    def predict(self, features: Sequence[float]) -> List[float]:
        x = list(features) + [1.0]
        normalized = []
        for row in self.weights:
            total = 0.0
            for weight, feature in zip(row, x):
                total += weight * feature
            normalized.append(total)
        if not self.target_means or not self.target_stds:
            return normalized
        return [normalized[index] * self.target_stds[index] + self.target_means[index] for index in range(len(normalized))]

    def fit(self, features: Sequence[Sequence[float]], targets: Sequence[Sequence[float]], l2: float) -> None:
        if not features:
            raise ValueError("no features provided")

        self.feature_means, self.feature_stds = fit_standardizer(features)
        standardized_features = [standardize_vector(vector, self.feature_means, self.feature_stds) for vector in features]
        self.target_means, self.target_stds = fit_standardizer(targets)
        standardized_targets = [standardize_vector(vector, self.target_means, self.target_stds) for vector in targets]

        feature_dim = len(standardized_features[0]) + 1
        target_dim = len(standardized_targets[0])
        self.weights = [[0.0] * feature_dim for _ in range(target_dim)]

        gram = [[0.0] * feature_dim for _ in range(feature_dim)]
        rhs = [[0.0] * target_dim for _ in range(feature_dim)]
        for feature_vector, target_vector in zip(standardized_features, standardized_targets):
            x = list(feature_vector) + [1.0]
            for row_index, left_value in enumerate(x):
                for col_index, right_value in enumerate(x):
                    gram[row_index][col_index] += left_value * right_value
                for target_index, target_value in enumerate(target_vector):
                    rhs[row_index][target_index] += left_value * target_value

        for diagonal in range(feature_dim):
            gram[diagonal][diagonal] += l2

        augmented = [gram_row[:] + rhs_row[:] for gram_row, rhs_row in zip(gram, rhs)]
        rhs_start = feature_dim
        for pivot_col in range(feature_dim):
            pivot_row = max(range(pivot_col, feature_dim), key=lambda row: abs(augmented[row][pivot_col]))
            pivot_value = augmented[pivot_row][pivot_col]
            if abs(pivot_value) <= 1e-12:
                continue
            if pivot_row != pivot_col:
                augmented[pivot_row], augmented[pivot_col] = augmented[pivot_col], augmented[pivot_row]

            pivot_value = augmented[pivot_col][pivot_col]
            for col_index in range(pivot_col, feature_dim + target_dim):
                augmented[pivot_col][col_index] /= pivot_value

            for row_index in range(feature_dim):
                if row_index == pivot_col:
                    continue
                factor = augmented[row_index][pivot_col]
                if abs(factor) <= 1e-18:
                    continue
                for col_index in range(pivot_col, feature_dim + target_dim):
                    augmented[row_index][col_index] -= factor * augmented[pivot_col][col_index]

        for feature_index in range(feature_dim):
            for target_index in range(target_dim):
                self.weights[target_index][feature_index] = augmented[feature_index][rhs_start + target_index]

    def save(self, path: Path) -> None:
        payload = {
            "model_type": "linear_bc_go2_v1",
            "target_fields": TARGET_FIELDS,
            "weights": self.weights,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "target_means": self.target_means,
            "target_stds": self.target_stds,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")


def _records_from_samples(samples: Sequence[Sample], dataset_root: Path) -> List[Dict[str, Any]]:
    return [sample.to_manifest_record(dataset_root) for sample in samples]


def _load_split_records(dataset_root: Path, args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    if has_converted_manifests(dataset_root):
        return {
            "train": load_jsonl(dataset_root / "train.jsonl") if (dataset_root / "train.jsonl").exists() else [],
            "val": load_jsonl(dataset_root / "val.jsonl") if (dataset_root / "val.jsonl").exists() else [],
            "test": load_jsonl(dataset_root / "test.jsonl") if (dataset_root / "test.jsonl").exists() else [],
        }

    session_roots = discover_session_roots(dataset_root)
    if not session_roots:
        raise FileNotFoundError(f"no converted manifests or raw session roots found under {dataset_root}")

    samples: List[Sample] = []
    for session_root in session_roots:
        samples.extend(load_session_samples(session_root, min_trajectory_length=args.min_episode_length))
    split_samples = assign_splits(
        samples,
        split_mode=args.split_mode,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    return {name: _records_from_samples(items, dataset_root) for name, items in split_samples.items()}


def _build_features_and_targets(records: Sequence[Dict[str, Any]], dataset_root: Path, text_dim: int) -> Tuple[List[List[float]], List[List[float]]]:
    features: List[List[float]] = []
    targets: List[List[float]] = []
    for record in records:
        features.append(feature_vector_from_record(record, dataset_root, text_dim=text_dim))
        targets.append(_target_vector(record))
    return features, targets


def _predict_many(model: LinearBaseline, features: Sequence[Sequence[float]]) -> List[List[float]]:
    return [model.predict(vector) for vector in features]


def _describe_split(name: str, records: Sequence[Dict[str, Any]]) -> str:
    sessions = len({str(record.get("session_id") or "") for record in records})
    episodes = len({(str(record.get("session_id") or ""), str(record.get("episode_id") or record.get("trajectory_id") or "")) for record in records})
    instructions = len({str(record.get("instruction") or "") for record in records})
    return f"{name}: {len(records)} samples, {sessions} sessions, {episodes} episodes, {instructions} instructions"


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    split_records = _load_split_records(dataset_root, args)

    train_records = split_records["train"]
    val_records = split_records["val"]
    test_records = split_records["test"]
    if not train_records:
        raise RuntimeError("no training records available")

    print(_describe_split("train", train_records))
    print(_describe_split("val", val_records))
    print(_describe_split("test", test_records))

    train_features, train_targets = _build_features_and_targets(train_records, dataset_root, args.text_dim)
    val_features, val_targets = _build_features_and_targets(val_records, dataset_root, args.text_dim)
    test_features, test_targets = _build_features_and_targets(test_records, dataset_root, args.text_dim)

    mean_baseline = _mean_vector(train_targets)
    if val_targets:
        print("mean-baseline val:", _mse_mae([mean_baseline for _ in val_targets], val_targets))
    if test_targets:
        print("mean-baseline test:", _mse_mae([mean_baseline for _ in test_targets], test_targets))

    model = LinearBaseline.create(len(train_features[0]), len(train_targets[0]))
    model.fit(train_features, train_targets, l2=args.l2)
    print("linear-baseline train:", _mse_mae(_predict_many(model, train_features), train_targets))
    if val_targets:
        print("linear-baseline val:", _mse_mae(_predict_many(model, val_features), val_targets))
    if test_targets:
        print("linear-baseline test:", _mse_mae(_predict_many(model, test_features), test_targets))

    if args.save_model is not None:
        model.save(args.save_model.resolve())
        print(f"saved model to {args.save_model.resolve()}")


if __name__ == "__main__":
    main()

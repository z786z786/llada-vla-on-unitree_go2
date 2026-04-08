from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "go2_local_dataset_v1"
STATE_FIELDS = ["vx", "vy", "wz", "yaw"]
ACTION_FIELDS = ["vx", "vy", "wz"]
RAW_ACTION_FIELDS = ["vx", "vy", "wz", "camera_pitch", "keys"]
CONTROLLED_INSTRUCTIONS = [
    "go forward",
    "move backward",
    "strafe left",
    "strafe right",
    "stand up",
    "lie down",
    "turn left",
    "turn right",
    "stay still",
]
KNOWN_TASK_FAMILIES = [
    "legacy_motion",
    "goal_navigation",
    "visual_following",
    "obstacle_aware_navigation",
]
VISUAL_TASK_FAMILIES = [
    "goal_navigation",
    "visual_following",
    "obstacle_aware_navigation",
]
KNOWN_TARGET_TYPES = [
    "door",
    "apple",
    "red_object",
    "box",
    "person",
    "obstacle",
    "landmark",
    "waypoint",
]
TASK_METADATA_FIELDS = [
    "capture_mode",
    "task_family",
    "target_type",
    "target_description",
    "target_instance_id",
    "task_tags",
    "collector_notes",
    "instruction_source",
]
ACTION_ACTIVITY_EPS = 0.05
TEXT_TOKEN_RE = re.compile(r"[a-z0-9]+")
INSTRUCTION_TEMPLATE_PATTERNS = {
    "goal_navigation": [
        re.compile(r"^go to the [a-z0-9 ]+$"),
        re.compile(r"^go to [a-z0-9 ]+$"),
        re.compile(r"^approach the [a-z0-9 ]+$"),
        re.compile(r"^approach [a-z0-9 ]+$"),
    ],
    "visual_following": [
        re.compile(r"^follow the [a-z0-9 ]+$"),
        re.compile(r"^follow [a-z0-9 ]+$"),
    ],
    "obstacle_aware_navigation": [
        re.compile(r"^go around the [a-z0-9 ]+$"),
        re.compile(r"^go around [a-z0-9 ]+$"),
        re.compile(r"^avoid the [a-z0-9 ]+$"),
        re.compile(r"^avoid [a-z0-9 ]+$"),
    ],
}


@dataclass(frozen=True)
class Sample:
    session_id: str
    trajectory_id: str
    trajectory_index: int
    trajectory_step_index: int
    trajectory_length: int
    step_id: int
    timestamp: float
    instruction: str
    image_path: Optional[str]
    source_image_path: Optional[str]
    state: Dict[str, Any]
    raw_action: Dict[str, Any]
    control_action: Dict[str, Any]
    action_chunk: List[Dict[str, Any]]
    raw_record: Dict[str, Any]

    def to_manifest_record(self, dataset_root: Path) -> Dict[str, Any]:
        record = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": f"{self.session_id}:{self.trajectory_id}:{self.step_id:06d}",
            "session_id": self.session_id,
            "episode_id": self.trajectory_id,
            "trajectory_id": self.trajectory_id,
            "trajectory_index": self.trajectory_index,
            "trajectory_step_index": self.trajectory_step_index,
            "trajectory_length": self.trajectory_length,
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "instruction": self.instruction,
            "state": self.state,
            "raw_action": self.raw_action,
            "control_action": self.control_action,
            "action_chunk": self.action_chunk,
            "chunk_length": len(self.action_chunk),
            "scene_id": str(self.raw_record.get("scene_id") or ""),
            "operator_id": str(self.raw_record.get("operator_id") or ""),
            "dataset_root": str(dataset_root),
        }
        for field in TASK_METADATA_FIELDS:
            value = self.raw_record.get(field)
            if value is None:
                continue
            if isinstance(value, str) and not value:
                continue
            if isinstance(value, list) and not value:
                continue
            record[field] = value
        if self.image_path is not None:
            record["image_path"] = self.image_path
        if self.source_image_path is not None:
            record["source_image_path"] = self.source_image_path
        return record


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dump_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def discover_session_roots(raw_root: Path) -> List[Path]:
    if raw_root.exists() and (raw_root / "index.json").exists() and (raw_root / "episodes").is_dir():
        return [raw_root]

    roots: List[Path] = []
    for child in sorted(raw_root.iterdir() if raw_root.exists() else []):
        if child.is_dir() and (child / "index.json").exists() and (child / "episodes").is_dir():
            roots.append(child)
    return roots


def ensure_session_materialized(output_root: Path, session_root: Path, session_id: str) -> Path:
    sessions_dir = output_root / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    target = sessions_dir / session_id
    if target.exists() or target.is_symlink():
        return target

    source = session_root.resolve()
    try:
        target.symlink_to(source, target_is_directory=True)
    except OSError:
        shutil.copytree(source, target)
    return target


def has_converted_manifests(dataset_root: Path) -> bool:
    return (dataset_root / "dataset.jsonl").exists()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def normalize_tags(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def is_controlled_instruction(instruction: str) -> bool:
    return instruction in CONTROLLED_INSTRUCTIONS


def infer_task_family(instruction: str, explicit_task_family: str = "") -> str:
    task_family = _safe_str(explicit_task_family)
    if task_family:
        return task_family
    normalized = _safe_str(instruction).lower()
    if normalized in CONTROLLED_INSTRUCTIONS:
        return "legacy_motion"
    for family, patterns in INSTRUCTION_TEMPLATE_PATTERNS.items():
        if any(pattern.fullmatch(normalized) for pattern in patterns):
            return family
    return ""


def instruction_matches_known_template(instruction: str, task_family: str = "") -> bool:
    normalized = _safe_str(instruction).lower()
    family = infer_task_family(normalized, task_family)
    if family == "legacy_motion":
        return normalized in CONTROLLED_INSTRUCTIONS
    patterns = INSTRUCTION_TEMPLATE_PATTERNS.get(family, [])
    return any(pattern.fullmatch(normalized) for pattern in patterns)


def episode_task_metadata(payload: Dict[str, Any], episode_meta: Dict[str, Any]) -> Dict[str, Any]:
    task_block = payload.get("task")
    if not isinstance(task_block, dict):
        task_block = {}

    instruction = _safe_str(payload.get("instruction"))
    explicit_task_family = (
        _safe_str(payload.get("task_family")) or
        _safe_str(episode_meta.get("task_family")) or
        _safe_str(task_block.get("task_family"))
    )
    task_family = infer_task_family(instruction, explicit_task_family)
    task_tags = normalize_tags(
        payload.get("task_tags")
        or episode_meta.get("task_tags")
        or task_block.get("task_tags")
        or payload.get("tags")
        or episode_meta.get("tags")
    )

    return {
        "capture_mode": (
            _safe_str(payload.get("capture_mode")) or
            _safe_str(episode_meta.get("capture_mode")) or
            _safe_str(task_block.get("capture_mode"))
        ),
        "task_family": task_family,
        "target_type": (
            _safe_str(payload.get("target_type")) or
            _safe_str(episode_meta.get("target_type")) or
            _safe_str(task_block.get("target_type"))
        ),
        "target_description": (
            _safe_str(payload.get("target_description")) or
            _safe_str(episode_meta.get("target_description")) or
            _safe_str(task_block.get("target_description"))
        ),
        "target_instance_id": (
            _safe_str(payload.get("target_instance_id")) or
            _safe_str(episode_meta.get("target_instance_id")) or
            _safe_str(task_block.get("target_instance_id"))
        ),
        "task_tags": task_tags,
        "collector_notes": (
            _safe_str(payload.get("collector_notes")) or
            _safe_str(episode_meta.get("collector_notes")) or
            _safe_str(task_block.get("collector_notes")) or
            _safe_str(payload.get("notes")) or
            _safe_str(episode_meta.get("notes"))
        ),
        "instruction_source": (
            _safe_str(payload.get("instruction_source")) or
            _safe_str(task_block.get("instruction_source")) or
            ("semantic_text" if task_family and task_family != "legacy_motion" else "motion_label")
        ),
    }


def action_bucket(
    action: Dict[str, Any],
    epsilon: float = ACTION_ACTIVITY_EPS,
) -> str:
    vx = _safe_float(action.get("vx"))
    vy = _safe_float(action.get("vy"))
    wz = _safe_float(action.get("wz"))
    active: List[str] = []
    if vx >= epsilon:
        active.append("forward")
    elif vx <= -epsilon:
        active.append("backward")
    if vy >= epsilon:
        active.append("strafe_left")
    elif vy <= -epsilon:
        active.append("strafe_right")
    if wz >= epsilon:
        active.append("turn_left")
    elif wz <= -epsilon:
        active.append("turn_right")
    if not active:
        return "stop"
    if len(active) == 1:
        return active[0]
    return "mixed:" + "+".join(active)


def summarize_trajectory_actions(
    actions: Sequence[Dict[str, Any]],
    timestamps: Optional[Sequence[float]] = None,
    epsilon: float = ACTION_ACTIVITY_EPS,
) -> Dict[str, Any]:
    frame_count = len(actions)
    if frame_count == 0:
        return {
            "frame_count": 0,
            "duration_seconds": 0.0,
            "action_change_count": 0,
            "stop_ratio": 0.0,
            "turn_ratio": 0.0,
            "move_ratio": 0.0,
            "distinct_action_bucket_count": 0,
            "action_buckets": [],
        }

    buckets = [action_bucket(action, epsilon=epsilon) for action in actions]
    compressed: List[str] = []
    for bucket in buckets:
        if not compressed or compressed[-1] != bucket:
            compressed.append(bucket)
    change_count = max(0, len(compressed) - 1)
    stop_frames = sum(1 for bucket in buckets if bucket == "stop")
    turn_frames = sum(1 for action in actions if abs(_safe_float(action.get("wz"))) >= epsilon)
    move_frames = sum(1 for bucket in buckets if bucket != "stop")

    duration_seconds = 0.0
    if timestamps and len(timestamps) >= 2:
        ordered = [_safe_float(value) for value in timestamps]
        duration_seconds = max(0.0, ordered[-1] - ordered[0])

    return {
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "action_change_count": change_count,
        "stop_ratio": stop_frames / frame_count,
        "turn_ratio": turn_frames / frame_count,
        "move_ratio": move_frames / frame_count,
        "distinct_action_bucket_count": len(set(buckets)),
        "action_buckets": compressed,
    }


def summarize_trajectory_metric_series(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "mean": float(sum(ordered) / len(ordered)),
        "median": float(statistics.median(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def action_from_payload(payload: Any, raw: bool = False) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    record: Dict[str, Any] = {field: _safe_float(payload.get(field)) for field in ACTION_FIELDS}
    if raw:
        record["camera_pitch"] = _safe_float(payload.get("camera_pitch"))
        record["keys"] = _safe_int(payload.get("keys"))
    return record


def raw_action_from_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    payload = frame.get("raw_action")
    if payload is None:
        payload = frame.get("action")
    if payload is None:
        payload = frame.get("control_action")
    return action_from_payload(payload, raw=True)


def control_action_from_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    payload = frame.get("control_action")
    if payload is None:
        payload = frame.get("action")
    if payload is None:
        payload = frame.get("raw_action")
    return action_from_payload(payload, raw=False)


def state_from_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    payload = frame.get("state") or {}
    return {field: _safe_float(payload.get(field)) for field in STATE_FIELDS}


def episode_entries(session_root: Path) -> Tuple[str, List[Dict[str, Any]]]:
    index_payload = load_json(session_root / "index.json")
    session_id = str(index_payload.get("session_id") or session_root.name)
    episodes = list(index_payload.get("episodes") or [])
    return session_id, episodes


def load_session_samples(session_root: Path, action_horizon: int = 1, min_trajectory_length: int = 1) -> List[Sample]:
    session_id, entries = episode_entries(session_root)
    samples: List[Sample] = []

    for trajectory_index, episode_meta in enumerate(entries):
        episode_id = str(episode_meta.get("episode_id") or "")
        if not episode_id:
            continue

        episode_path = session_root / "episodes" / f"{episode_id}.json"
        payload = load_json(episode_path)
        instruction = str(payload.get("instruction") or "")
        scene_id = str(payload.get("scene_id") or episode_meta.get("scene_id") or "")
        operator_id = str(payload.get("operator_id") or episode_meta.get("operator_id") or "")
        task_metadata = episode_task_metadata(payload, episode_meta)
        schema_version = str(payload.get("schema_version") or payload.get("meta", {}).get("schema_version") or "")
        frames = list(payload.get("frames") or [])
        trajectory_length = len(frames)
        if trajectory_length < max(1, int(min_trajectory_length)):
            continue

        for step_pos, frame in enumerate(frames):
            image_rel = frame.get("image")
            source_image_path = None if not image_rel else str(session_root / str(image_rel))
            image_path = None if not image_rel else str(Path("sessions") / session_id / str(image_rel))

            action_chunk: List[Dict[str, Any]] = []
            for future_pos in range(step_pos, min(step_pos + action_horizon, trajectory_length)):
                action_chunk.append(control_action_from_frame(frames[future_pos]))

            raw_record = {
                "schema_version": schema_version,
                "episode_id": episode_id,
                "trajectory_id": episode_id,
                "session_id": session_id,
                "instruction": instruction,
                "scene_id": scene_id,
                "operator_id": operator_id,
            }
            raw_record.update(task_metadata)

            samples.append(
                Sample(
                    session_id=session_id,
                    trajectory_id=episode_id,
                    trajectory_index=trajectory_index,
                    trajectory_step_index=step_pos,
                    trajectory_length=trajectory_length,
                    step_id=step_pos + 1,
                    timestamp=_safe_float(frame.get("timestamp")),
                    instruction=instruction,
                    image_path=image_path,
                    source_image_path=source_image_path,
                    state=state_from_frame(frame),
                    raw_action=raw_action_from_frame(frame),
                    control_action=control_action_from_frame(frame),
                    action_chunk=action_chunk,
                    raw_record=raw_record,
                )
            )

    return samples


def _ratio_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, int]:
    if total <= 0:
        return {"train": 0, "val": 0, "test": 0}

    ratios = [max(0.0, train_ratio), max(0.0, val_ratio), max(0.0, test_ratio)]
    if sum(ratios) <= 0.0:
        ratios = [1.0, 0.0, 0.0]
    ratio_sum = sum(ratios)
    ratios = [value / ratio_sum for value in ratios]

    counts = {
        "train": int(total * ratios[0]),
        "val": int(total * ratios[1]),
        "test": total - int(total * ratios[0]) - int(total * ratios[1]),
    }

    if total >= 3:
        if ratios[0] > 0.0 and counts["train"] == 0:
            counts["train"] = 1
        if ratios[1] > 0.0 and counts["val"] == 0:
            counts["val"] = 1
        if ratios[2] > 0.0 and counts["test"] == 0:
            counts["test"] = 1

    while sum(counts.values()) > total:
        for name in ("train", "val", "test"):
            if counts[name] > 0 and sum(counts.values()) > total:
                counts[name] -= 1

    while sum(counts.values()) < total:
        counts["train"] += 1

    return counts


def assign_splits(
    samples: Sequence[Sample],
    split_mode: str = "auto",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, List[Sample]]:
    if not samples:
        return {"train": [], "val": [], "test": []}

    session_groups: Dict[str, List[Sample]] = {}
    trajectory_groups: Dict[Tuple[str, str], List[Sample]] = {}
    for sample in sorted(samples, key=lambda item: (item.session_id, item.trajectory_id, item.step_id)):
        session_groups.setdefault(sample.session_id, []).append(sample)
        trajectory_groups.setdefault((sample.session_id, sample.trajectory_id), []).append(sample)

    if split_mode == "auto":
        split_mode = "by_session" if len(session_groups) >= 2 else "by_trajectory"
    if split_mode not in {"by_session", "by_trajectory"}:
        raise ValueError(f"unsupported split mode: {split_mode}")

    grouped_items: List[List[Sample]]
    if split_mode == "by_session":
        def _session_sort_key(key: str) -> Tuple[int, str]:
            episode_count = len({sample.trajectory_id for sample in session_groups[key]})
            return (-episode_count, key)

        grouped_items = [session_groups[key] for key in sorted(session_groups, key=_session_sort_key)]
    else:
        grouped_items = [trajectory_groups[key] for key in sorted(trajectory_groups)]

    counts = _ratio_counts(len(grouped_items), train_ratio, val_ratio, test_ratio)
    split_names: List[str] = []
    for name in ("train", "val", "test"):
        split_names.extend([name] * counts[name])
    while len(split_names) < len(grouped_items):
        split_names.append("train")

    split_samples = {"train": [], "val": [], "test": []}
    for group, split_name in zip(grouped_items, split_names):
        split_samples[split_name].extend(group)
    return split_samples


def hash_token(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


def tokenize_instruction(text: str) -> List[str]:
    return TEXT_TOKEN_RE.findall(text.lower())


def instruction_feature_vector(text: str, dim: int = 64) -> List[float]:
    tokens = tokenize_instruction(text)
    vector = [0.0] * dim
    if not tokens:
        return vector
    scale = 1.0 / len(tokens)
    for token in tokens:
        vector[hash_token(token, dim)] += scale
    return vector


def state_feature_vector(state: Dict[str, Any]) -> List[float]:
    return [_safe_float(state.get(field)) for field in STATE_FIELDS]


def resolve_image_path(record: Dict[str, Any], dataset_root: Path) -> Optional[Path]:
    source = record.get("source_image_path")
    if source:
        path = Path(str(source))
        if path.exists():
            return path
    image_path = record.get("image_path")
    if image_path:
        path = dataset_root / str(image_path)
        if path.exists():
            return path
    return None


def image_feature_vector(image_path: Optional[Path]) -> List[float]:
    if image_path is None or not image_path.exists():
        return [0.0] * 20

    size = image_path.stat().st_size
    if size <= 0:
        return [0.0] * 20

    with image_path.open("rb") as handle:
        data = handle.read(4096)
    if not data:
        return [0.0] * 20

    hist = [0] * 16
    total = 0
    total_sq = 0
    for byte in data:
        hist[byte // 16] += 1
        total += byte
        total_sq += byte * byte

    mean = total / size
    variance = max(0.0, (total_sq / size) - (mean * mean))
    std = math.sqrt(variance)
    entropy = 0.0
    for count in hist:
        if count == 0:
            continue
        probability = count / size
        entropy -= probability * math.log2(probability)

    vector = [count / size for count in hist]
    vector.extend([math.log1p(size), mean / 255.0, std / 128.0, entropy / 8.0])
    return vector


def feature_vector_from_record(record: Dict[str, Any], dataset_root: Path, text_dim: int = 64) -> List[float]:
    vector: List[float] = []
    vector.extend(instruction_feature_vector(str(record.get("instruction") or ""), text_dim))
    vector.extend(state_feature_vector(record.get("state") or {}))
    vector.extend(image_feature_vector(resolve_image_path(record, dataset_root)))
    return vector


def fit_standardizer(vectors: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    if not vectors:
        return [], []

    dim = len(vectors[0])
    means = [0.0] * dim
    for vector in vectors:
        for index, value in enumerate(vector):
            means[index] += float(value)
    means = [value / len(vectors) for value in means]

    variances = [0.0] * dim
    for vector in vectors:
        for index, value in enumerate(vector):
            diff = float(value) - means[index]
            variances[index] += diff * diff
    stds = [math.sqrt(value / max(1, len(vectors))) for value in variances]
    stds = [std if std > 1e-12 else 1.0 for std in stds]
    return means, stds


def standardize_vector(vector: Sequence[float], means: Sequence[float], stds: Sequence[float]) -> List[float]:
    if not means or not stds:
        return list(vector)
    return [(float(value) - means[index]) / stds[index] for index, value in enumerate(vector)]

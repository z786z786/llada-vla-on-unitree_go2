from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageFile, UnidentifiedImageError

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from llada_vla_common import (
    ACTION_FIELDS,
    Sample,
    assign_splits,
    discover_session_roots,
    fit_standardizer,
    has_converted_manifests,
    load_jsonl,
    load_session_samples,
    resolve_image_path,
    standardize_vector,
    tokenize_instruction,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import models, transforms
except ModuleNotFoundError:
    torch = None
    nn = None
    DataLoader = None
    models = None
    transforms = None

DatasetBase = torch.utils.data.Dataset if torch is not None else object
ModuleBase = nn.Module if nn is not None else object

TARGET_FIELDS = list(ACTION_FIELDS)
SAFE_DEFAULT_STATE_FIELDS = ["yaw"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练并评估 Go2 多模态 VLA 基线 v2")
    parser.add_argument("--dataset-root", type=Path, required=True, help="转换后清单根目录，或原始 session 根目录")
    parser.add_argument("--manifest-path", type=Path, help="可选：转换后数据集内的 manifest 文件路径")
    parser.add_argument("--output-dir", type=Path, required=True, help="checkpoint 与指标输出目录")
    parser.add_argument(
        "--ablation-mode",
        choices=["instruction_only", "image_plus_instruction", "image_plus_instruction_plus_state"],
        default="image_plus_instruction",
        help="启用哪些模态",
    )
    parser.add_argument("--image-size", type=int, default=224, help="视觉编码器使用的方形图像尺寸")
    parser.add_argument("--image-error-mode", choices=["blank", "raise"], default="blank", help="无法读取图像时的处理方式")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW 学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW 权重衰减")
    parser.add_argument("--hidden-dim", type=int, default=256, help="模态投影共享隐藏维度")
    parser.add_argument("--text-embed-dim", type=int, default=64, help="文本 token 投影前的嵌入维度")
    parser.add_argument("--vision-learning-rate", type=float, help="可选：视觉骨干网络单独学习率")
    parser.add_argument("--freeze-vision-backbone", action="store_true", help="冻结 ResNet18 骨干，仅训练投影/融合/输出头")
    parser.add_argument("--use-state", type=_parse_bool, default=False, help="在当前消融模式允许时启用状态分支")
    parser.add_argument("--state-fields", type=str, help="要使用的状态字段，逗号分隔；默认只使用 yaw")
    parser.add_argument("--fusion-type", choices=["mlp", "tiny_transformer"], default="mlp", help="多模态融合类型")
    parser.add_argument(
        "--split-strategy",
        choices=["use_manifest", "auto", "by_session", "by_episode"],
        default="use_manifest",
        help="使用原始 session 时如何划分 train/val/test",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="原始 session 划分训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="原始 session 划分验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="原始 session 划分测试集比例")
    parser.add_argument("--split-seed", type=int, help="可选：划分前随机打散 session/episode 分组的随机种子")
    parser.add_argument("--min-episode-length", type=int, default=1, help="丢弃短于该长度的原始 episode")
    parser.add_argument("--checkpoint-path", type=Path, help="用于恢复训练或仅评估的 checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="跳过训练，直接评估 checkpoint")
    parser.add_argument("--save-predictions", action="store_true", help="保存 val/test 逐样本预测结果")
    parser.add_argument("--compare-old-linear", action="store_true", help="同时报告当前手工线性基线结果")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker 数量")
    parser.add_argument("--device", type=str, help="Torch 设备，例如 cuda、cuda:0、cpu")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no-pretrained-vision", action="store_true", help="禁用预训练 ResNet18 权重")
    return parser


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"非法布尔值：{value}")


def _require_torch() -> None:
    if torch is None or nn is None or DataLoader is None or models is None or transforms is None:
        raise ModuleNotFoundError(
            "PyTorch 基线 v2 依赖 torch 与 torchvision，请先安装后重试。"
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _manifest_root_from_args(dataset_root: Path, manifest_path: Optional[Path]) -> Path:
    if manifest_path is None:
        return dataset_root
    manifest_path = manifest_path.resolve()
    if manifest_path.is_dir():
        return manifest_path
    return manifest_path.parent


def _records_from_samples(samples: Sequence[Sample], dataset_root: Path) -> List[Dict[str, Any]]:
    return [sample.to_manifest_record(dataset_root) for sample in samples]


def _load_split_records(dataset_root: Path, manifest_root: Path, args: argparse.Namespace) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    if has_converted_manifests(manifest_root) and args.split_strategy == "use_manifest":
        split_records = {
            "train": load_jsonl(manifest_root / "train.jsonl") if (manifest_root / "train.jsonl").exists() else [],
            "val": load_jsonl(manifest_root / "val.jsonl") if (manifest_root / "val.jsonl").exists() else [],
            "test": load_jsonl(manifest_root / "test.jsonl") if (manifest_root / "test.jsonl").exists() else [],
        }
        return split_records, {"source": "manifest_files", "strategy": "use_manifest"}

    session_roots = discover_session_roots(dataset_root)
    if not session_roots:
        raise FileNotFoundError(f"no converted manifests or raw session roots found under {dataset_root}")

    samples: List[Sample] = []
    for session_root in session_roots:
        samples.extend(load_session_samples(session_root, min_trajectory_length=args.min_episode_length))

    strategy = args.split_strategy
    mapped = "auto"
    if strategy == "by_episode":
        mapped = "by_trajectory"
    elif strategy in {"auto", "by_session"}:
        mapped = strategy
    elif strategy == "use_manifest":
        mapped = "auto"

    split_samples = assign_splits(
        samples,
        split_mode=mapped,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )
    split_records = {name: _records_from_samples(items, dataset_root) for name, items in split_samples.items()}
    resolved_strategy = strategy if strategy != "use_manifest" else ("by_episode" if mapped == "by_trajectory" else mapped)
    if strategy == "auto":
        resolved_strategy = "by_session" if len({sample.session_id for sample in samples}) >= 2 else "by_episode"
    split_info = {"source": "raw_sessions", "strategy": resolved_strategy}
    if args.split_seed is not None:
        split_info["seed"] = int(args.split_seed)
    return split_records, split_info


def _effective_modalities(ablation_mode: str, use_state: bool) -> Dict[str, bool]:
    use_image = ablation_mode != "instruction_only"
    allow_state = ablation_mode == "image_plus_instruction_plus_state"
    return {"text": True, "image": use_image, "state": allow_state and use_state}


def _resolve_state_fields(args: argparse.Namespace, modalities: Dict[str, bool]) -> List[str]:
    if not modalities["state"]:
        return []
    if args.state_fields:
        return [field.strip() for field in args.state_fields.split(",") if field.strip()]
    return list(SAFE_DEFAULT_STATE_FIELDS)


def _sample_target(record: Dict[str, Any]) -> List[float]:
    action = record.get("control_action") or {}
    return [float(action.get(field, 0.0)) for field in TARGET_FIELDS]


def _describe_split(name: str, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    instruction_counter = Counter(str(record.get("instruction") or "") for record in records)
    sessions = sorted({str(record.get("session_id") or "") for record in records})
    episodes = sorted({f"{record.get('session_id') or ''}:{record.get('episode_id') or record.get('trajectory_id') or ''}" for record in records})
    return {
        "name": name,
        "sample_count": len(records),
        "session_count": len(sessions),
        "episode_count": len(episodes),
        "instruction_count": len(instruction_counter),
        "sessions": sessions,
        "instruction_counts": dict(sorted(instruction_counter.items())),
    }


class InstructionTokenizer:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, stoi: Dict[str, int]) -> None:
        self.stoi = dict(stoi)
        self.itos = [token for token, _ in sorted(self.stoi.items(), key=lambda item: item[1])]
        self.pad_id = self.stoi[self.PAD_TOKEN]
        self.unk_id = self.stoi[self.UNK_TOKEN]

    @classmethod
    def build(cls, texts: Sequence[str]) -> "InstructionTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize_instruction(text))
        stoi = {cls.PAD_TOKEN: 0, cls.UNK_TOKEN: 1}
        for token, _ in sorted(counter.items()):
            stoi[token] = len(stoi)
        return cls(stoi)

    @classmethod
    def from_payload(cls, payload: Dict[str, int]) -> "InstructionTokenizer":
        return cls(payload)

    def encode(self, text: str) -> List[int]:
        tokens = tokenize_instruction(text)
        if not tokens:
            return [self.unk_id]
        return [self.stoi.get(token, self.unk_id) for token in tokens]

    def to_payload(self) -> Dict[str, int]:
        return dict(self.stoi)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


class Go2ManifestDataset(DatasetBase):
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        dataset_root: Path,
        tokenizer: InstructionTokenizer,
        image_transform: Optional[Any],
        state_fields: Sequence[str],
        modalities: Dict[str, bool],
        image_error_mode: str,
    ) -> None:
        self.records = list(records)
        self.dataset_root = dataset_root
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.state_fields = list(state_fields)
        self.modalities = dict(modalities)
        self.image_error_mode = image_error_mode
        self._image_error_count = 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        item: Dict[str, Any] = {
            "record": record,
            "instruction_ids": self.tokenizer.encode(str(record.get("instruction") or "")),
            "target": _sample_target(record),
            "sample_id": str(record.get("sample_id") or ""),
        }
        if self.modalities["image"]:
            image_path = resolve_image_path(record, self.dataset_root)
            if image_path is None:
                raise FileNotFoundError(f"missing image for sample {item['sample_id']}")
            image = self._load_image(image_path=image_path, sample_id=item["sample_id"])
            item["image"] = self.image_transform(image) if self.image_transform is not None else image
        if self.modalities["state"]:
            state = record.get("state") or {}
            item["state"] = [float(state.get(field, 0.0)) for field in self.state_fields]
        return item

    def _blank_image(self) -> Image.Image:
        return Image.new("RGB", (224, 224), color=(0, 0, 0))

    def _load_image(self, image_path: Path, sample_id: str) -> Image.Image:
        try:
            with Image.open(image_path) as source:
                return source.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            if self.image_error_mode == "raise":
                raise RuntimeError(f"加载样本图像失败：{sample_id}: {image_path}: {error}") from error
            self._image_error_count += 1
            if self._image_error_count <= 5:
                print(
                    f"warning: 加载样本图像失败：{sample_id}, "
                    f"改用空白图像回退：{image_path} ({error})"
                )
            return self._blank_image()


class VisionEncoder(ModuleBase):
    def __init__(self, hidden_dim: int, pretrained: bool) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(backbone.fc.in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def backbone_parameters(self) -> List[Any]:
        return list(self.backbone.parameters())

    def forward(self, image: Any) -> Any:
        features = self.backbone(image).flatten(1)
        return self.projector(features)


class TextEncoder(ModuleBase):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_id: int) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

    def forward(self, token_ids: Any) -> Any:
        embeddings = self.embedding(token_ids)
        mask = (token_ids != self.pad_id).unsqueeze(-1)
        masked = embeddings * mask
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = masked.sum(dim=1) / counts
        return self.projector(pooled)


class StateEncoder(ModuleBase):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

    def forward(self, state: Any) -> Any:
        return self.projector(state)


class MultimodalFusionMLP(ModuleBase):
    def __init__(self, token_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_count * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, tokens: Sequence[Any]) -> Any:
        return self.net(torch.cat(list(tokens), dim=-1))


class MultimodalFusionTransformer(ModuleBase):
    def __init__(self, token_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.modality_embed = nn.Parameter(torch.zeros(1, token_count, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, tokens: Sequence[Any]) -> Any:
        stacked = torch.stack(list(tokens), dim=1) + self.modality_embed[:, : len(tokens), :]
        encoded = self.encoder(stacked)
        return encoded.mean(dim=1)


class Go2VLABaselineV2(ModuleBase):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        hidden_dim: int,
        text_embed_dim: int,
        fusion_type: str,
        modalities: Dict[str, bool],
        state_dim: int,
        pretrained_vision: bool,
    ) -> None:
        super().__init__()
        self.modalities = dict(modalities)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=text_embed_dim, hidden_dim=hidden_dim, pad_id=pad_id)
        self.vision_encoder = VisionEncoder(hidden_dim=hidden_dim, pretrained=pretrained_vision) if modalities["image"] else None
        self.state_encoder = StateEncoder(input_dim=state_dim, hidden_dim=hidden_dim) if modalities["state"] else None

        token_count = 1 + int(modalities["image"]) + int(modalities["state"])
        if fusion_type == "tiny_transformer":
            self.fusion = MultimodalFusionTransformer(token_count=token_count, hidden_dim=hidden_dim)
        else:
            self.fusion = MultimodalFusionMLP(token_count=token_count, hidden_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, len(TARGET_FIELDS))

    def forward(self, batch: Dict[str, Any]) -> Any:
        tokens = [self.text_encoder(batch["instruction_ids"])]
        if self.vision_encoder is not None:
            tokens.append(self.vision_encoder(batch["image"]))
        if self.state_encoder is not None:
            tokens.append(self.state_encoder(batch["state"]))
        fused = self.fusion(tokens)
        return self.head(fused)


def _image_transform(image_size: int) -> Any:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _collate_fn(batch: Sequence[Dict[str, Any]], tokenizer: InstructionTokenizer, modalities: Dict[str, bool]) -> Dict[str, Any]:
    max_len = max(len(item["instruction_ids"]) for item in batch)
    instruction_rows = []
    targets = []
    sample_ids = []
    records = []
    for item in batch:
        row = list(item["instruction_ids"])
        row.extend([tokenizer.pad_id] * (max_len - len(row)))
        instruction_rows.append(row)
        targets.append(item["target"])
        sample_ids.append(item["sample_id"])
        records.append(item["record"])

    collated: Dict[str, Any] = {
        "instruction_ids": torch.tensor(instruction_rows, dtype=torch.long),
        "target": torch.tensor(targets, dtype=torch.float32),
        "sample_ids": sample_ids,
        "records": records,
    }
    if modalities["image"]:
        collated["image"] = torch.stack([item["image"] for item in batch], dim=0)
    if modalities["state"]:
        collated["state"] = torch.tensor([item["state"] for item in batch], dtype=torch.float32)
    return collated


def _make_loader(
    records: Sequence[Dict[str, Any]],
    dataset_root: Path,
    tokenizer: InstructionTokenizer,
    image_size: int,
    state_fields: Sequence[str],
    modalities: Dict[str, bool],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    image_error_mode: str,
) -> Any:
    dataset = Go2ManifestDataset(
        records=records,
        dataset_root=dataset_root,
        tokenizer=tokenizer,
        image_transform=_image_transform(image_size) if modalities["image"] else None,
        state_fields=state_fields,
        modalities=modalities,
        image_error_mode=image_error_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: _collate_fn(batch, tokenizer=tokenizer, modalities=modalities),
    )


def _metrics_from_predictions(predictions: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> Dict[str, Any]:
    if not targets:
        return {"count": 0, "per_axis": {}, "overall_mae": 0.0, "overall_rmse": 0.0}

    abs_sums = [0.0] * len(TARGET_FIELDS)
    sq_sums = [0.0] * len(TARGET_FIELDS)
    overall_abs = 0.0
    overall_sq = 0.0
    count = len(targets)
    for pred, target in zip(predictions, targets):
        for index, axis in enumerate(TARGET_FIELDS):
            err = float(pred[index]) - float(target[index])
            abs_sums[index] += abs(err)
            sq_sums[index] += err * err
            overall_abs += abs(err)
            overall_sq += err * err

    per_axis = {
        axis: {"mae": abs_sums[index] / count, "rmse": math.sqrt(sq_sums[index] / count)}
        for index, axis in enumerate(TARGET_FIELDS)
    }
    return {
        "count": count,
        "per_axis": per_axis,
        "overall_mae": overall_abs / (count * len(TARGET_FIELDS)),
        "overall_rmse": math.sqrt(overall_sq / (count * len(TARGET_FIELDS))),
    }


def _mean_baseline_metrics(train_records: Sequence[Dict[str, Any]], eval_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    train_targets = [_sample_target(record) for record in train_records]
    eval_targets = [_sample_target(record) for record in eval_records]
    means, _ = fit_standardizer(train_targets)
    predictions = [means for _ in eval_targets]
    return _metrics_from_predictions(predictions, eval_targets)


def _maybe_old_linear_baseline(
    train_records: Sequence[Dict[str, Any]],
    eval_records: Sequence[Dict[str, Any]],
    dataset_root: Path,
) -> Optional[Dict[str, Any]]:
    try:
        from llada_vla_baseline import LinearBaseline
        from llada_vla_baseline import _build_features_and_targets as build_features_and_targets
        from llada_vla_baseline import _predict_many as predict_many
    except Exception:
        return None

    if not train_records or not eval_records:
        return None
    train_features, train_targets = build_features_and_targets(train_records, dataset_root, text_dim=64)
    eval_features, eval_targets = build_features_and_targets(eval_records, dataset_root, text_dim=64)
    model = LinearBaseline.create(len(train_features[0]), len(train_targets[0]))
    model.fit(train_features, train_targets, l2=1e-4)
    predictions = predict_many(model, eval_features)
    return _metrics_from_predictions(predictions, eval_targets)


def _device_from_args(device_arg: Optional[str]) -> Any:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch_to_device(batch: Dict[str, Any], device: Any, modalities: Dict[str, bool]) -> Dict[str, Any]:
    moved = dict(batch)
    moved["instruction_ids"] = batch["instruction_ids"].to(device)
    moved["target"] = batch["target"].to(device)
    if modalities["image"]:
        moved["image"] = batch["image"].to(device)
    if modalities["state"]:
        moved["state"] = batch["state"].to(device)
    return moved


def _evaluate_model(model: Go2VLABaselineV2, loader: Any, device: Any, modalities: Dict[str, bool]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    predictions: List[List[float]] = []
    targets: List[List[float]] = []
    dumps: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device=device, modalities=modalities)
            outputs = model(batch)
            batch_predictions = outputs.detach().cpu().tolist()
            batch_targets = batch["target"].detach().cpu().tolist()
            predictions.extend(batch_predictions)
            targets.extend(batch_targets)
            for sample_id, record, pred, target in zip(batch["sample_ids"], batch["records"], batch_predictions, batch_targets):
                dumps.append(
                    {
                        "sample_id": sample_id,
                        "session_id": str(record.get("session_id") or ""),
                        "episode_id": str(record.get("episode_id") or ""),
                        "instruction": str(record.get("instruction") or ""),
                        "prediction": {axis: pred[index] for index, axis in enumerate(TARGET_FIELDS)},
                        "target": {axis: target[index] for index, axis in enumerate(TARGET_FIELDS)},
                    }
                )
    return _metrics_from_predictions(predictions, targets), dumps


def _train_one_epoch(model: Go2VLABaselineV2, loader: Any, optimizer: Any, device: Any, modalities: Dict[str, bool]) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        batch = _move_batch_to_device(batch, device=device, modalities=modalities)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss = loss_fn(outputs, batch["target"])
        loss.backward()
        optimizer.step()
        batch_size = batch["target"].shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / max(1, total_samples)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def _save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _build_optimizer(model: Go2VLABaselineV2, args: argparse.Namespace) -> Any:
    parameter_groups: List[Dict[str, Any]] = []
    vision_backbone_ids = set()
    if model.vision_encoder is not None:
        if args.freeze_vision_backbone:
            model.vision_encoder.freeze_backbone()
        else:
            vision_params = [param for param in model.vision_encoder.backbone_parameters() if param.requires_grad]
            if vision_params:
                parameter_groups.append(
                    {
                        "params": vision_params,
                        "lr": args.vision_learning_rate if args.vision_learning_rate is not None else args.learning_rate,
                    }
                )
                vision_backbone_ids = {id(param) for param in vision_params}

    other_params = [param for param in model.parameters() if param.requires_grad and id(param) not in vision_backbone_ids]
    if other_params:
        parameter_groups.append({"params": other_params, "lr": args.learning_rate})
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def _checkpoint_payload(
    model: Go2VLABaselineV2,
    optimizer: Optional[Any],
    epoch: int,
    best_val_rmse: float,
    tokenizer: InstructionTokenizer,
    state_fields: Sequence[str],
    modalities: Dict[str, bool],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "best_val_rmse": best_val_rmse,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "tokenizer": tokenizer.to_payload(),
        "state_fields": list(state_fields),
        "modalities": dict(modalities),
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "text_embed_dim": args.text_embed_dim,
            "fusion_type": args.fusion_type,
            "pretrained_vision": not args.no_pretrained_vision,
            "freeze_vision_backbone": args.freeze_vision_backbone,
        },
    }


def _load_checkpoint(path: Path, device: Any) -> Dict[str, Any]:
    return torch.load(path, map_location=device)


def _build_model_from_checkpoint_or_args(
    args: argparse.Namespace,
    tokenizer: InstructionTokenizer,
    modalities: Dict[str, bool],
    state_fields: Sequence[str],
    checkpoint: Optional[Dict[str, Any]],
) -> Go2VLABaselineV2:
    if checkpoint is not None:
        config = checkpoint.get("model_config") or {}
        hidden_dim = int(config.get("hidden_dim", args.hidden_dim))
        text_embed_dim = int(config.get("text_embed_dim", args.text_embed_dim))
        fusion_type = str(config.get("fusion_type") or args.fusion_type)
        pretrained_vision = bool(config.get("pretrained_vision", not args.no_pretrained_vision))
    else:
        hidden_dim = args.hidden_dim
        text_embed_dim = args.text_embed_dim
        fusion_type = args.fusion_type
        pretrained_vision = not args.no_pretrained_vision

    return Go2VLABaselineV2(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        hidden_dim=hidden_dim,
        text_embed_dim=text_embed_dim,
        fusion_type=fusion_type,
        modalities=modalities,
        state_dim=len(state_fields),
        pretrained_vision=pretrained_vision,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _require_torch()
    _set_seed(args.seed)

    dataset_root = args.dataset_root.resolve()
    manifest_root = _manifest_root_from_args(dataset_root, args.manifest_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modalities = _effective_modalities(args.ablation_mode, args.use_state)
    state_fields = _resolve_state_fields(args, modalities)

    split_records, split_info = _load_split_records(dataset_root=dataset_root, manifest_root=manifest_root, args=args)
    train_records = split_records["train"]
    val_records = split_records["val"]
    test_records = split_records["test"]
    if not train_records:
        raise RuntimeError("no training records available")

    checkpoint = None
    if args.checkpoint_path is not None:
        checkpoint = _load_checkpoint(args.checkpoint_path.resolve(), device=torch.device("cpu"))
        if checkpoint.get("modalities"):
            modalities = dict(checkpoint["modalities"])
        if checkpoint.get("state_fields") is not None:
            state_fields = list(checkpoint["state_fields"])

    tokenizer = (
        InstructionTokenizer.from_payload(checkpoint["tokenizer"])
        if checkpoint is not None and checkpoint.get("tokenizer")
        else InstructionTokenizer.build([str(record.get("instruction") or "") for record in train_records])
    )

    device = _device_from_args(args.device)
    train_loader = _make_loader(
        records=train_records,
        dataset_root=manifest_root,
        tokenizer=tokenizer,
        image_size=args.image_size,
        state_fields=state_fields,
        modalities=modalities,
        batch_size=args.batch_size,
        shuffle=not args.eval_only,
        num_workers=args.num_workers,
        image_error_mode=args.image_error_mode,
    )
    val_loader = _make_loader(
        records=val_records,
        dataset_root=manifest_root,
        tokenizer=tokenizer,
        image_size=args.image_size,
        state_fields=state_fields,
        modalities=modalities,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        image_error_mode=args.image_error_mode,
    )
    test_loader = _make_loader(
        records=test_records,
        dataset_root=manifest_root,
        tokenizer=tokenizer,
        image_size=args.image_size,
        state_fields=state_fields,
        modalities=modalities,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        image_error_mode=args.image_error_mode,
    )

    model = _build_model_from_checkpoint_or_args(
        args=args,
        tokenizer=tokenizer,
        modalities=modalities,
        state_fields=state_fields,
        checkpoint=checkpoint,
    )
    if args.freeze_vision_backbone and model.vision_encoder is not None:
        model.vision_encoder.freeze_backbone()
    model.to(device)

    optimizer = _build_optimizer(model, args)
    start_epoch = 0
    best_val_rmse = float("inf")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state") is not None and not args.eval_only:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_rmse = float(checkpoint.get("best_val_rmse", float("inf")))

    split_summary = {
        name: _describe_split(name, records)
        for name, records in (("train", train_records), ("val", val_records), ("test", test_records))
    }
    print(f"Split source: {split_info['source']} strategy={split_info['strategy']}")
    for name in ("train", "val", "test"):
        summary = split_summary[name]
        print(
            f"{name}: {summary['sample_count']} samples, "
            f"{summary['session_count']} sessions, {summary['episode_count']} episodes, "
            f"{summary['instruction_count']} instructions"
        )

    history: List[Dict[str, Any]] = []
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    if not args.eval_only:
        for epoch in range(start_epoch, args.epochs):
            train_loss = _train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device, modalities=modalities)
            train_metrics, _ = _evaluate_model(model=model, loader=train_loader, device=device, modalities=modalities)
            val_metrics, _ = _evaluate_model(model=model, loader=val_loader, device=device, modalities=modalities)
            epoch_summary = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_overall_rmse": train_metrics["overall_rmse"],
                "val_overall_rmse": val_metrics["overall_rmse"],
            }
            history.append(epoch_summary)
            print(
                f"epoch={epoch + 1} train_loss={train_loss:.6f} "
                f"train_rmse={train_metrics['overall_rmse']:.6f} val_rmse={val_metrics['overall_rmse']:.6f}"
            )
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    best_val_rmse=best_val_rmse,
                    tokenizer=tokenizer,
                    state_fields=state_fields,
                    modalities=modalities,
                    args=args,
                ),
                last_path,
            )
            if val_metrics["count"] > 0 and val_metrics["overall_rmse"] <= best_val_rmse:
                best_val_rmse = val_metrics["overall_rmse"]
                torch.save(
                    _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch + 1,
                        best_val_rmse=best_val_rmse,
                        tokenizer=tokenizer,
                        state_fields=state_fields,
                        modalities=modalities,
                        args=args,
                    ),
                    best_path,
                )

    eval_checkpoint_path = args.checkpoint_path.resolve() if args.eval_only else (best_path if best_path.exists() else last_path)
    if not eval_checkpoint_path.exists():
        raise FileNotFoundError(f"no checkpoint available for evaluation: {eval_checkpoint_path}")
    best_checkpoint = _load_checkpoint(eval_checkpoint_path, device=device)
    model.load_state_dict(best_checkpoint["model_state"])
    model.to(device)

    train_metrics, train_dumps = _evaluate_model(model=model, loader=train_loader, device=device, modalities=modalities)
    val_metrics, val_dumps = _evaluate_model(model=model, loader=val_loader, device=device, modalities=modalities)
    test_metrics, test_dumps = _evaluate_model(model=model, loader=test_loader, device=device, modalities=modalities)
    mean_baseline = {
        "train": _mean_baseline_metrics(train_records, train_records),
        "val": _mean_baseline_metrics(train_records, val_records),
        "test": _mean_baseline_metrics(train_records, test_records),
    }
    old_linear = None
    if args.compare_old_linear:
        old_linear = {
            "val": _maybe_old_linear_baseline(train_records, val_records, manifest_root),
            "test": _maybe_old_linear_baseline(train_records, test_records, manifest_root),
        }

    metrics_payload = {
        "model_name": "go2_vla_baseline_v2",
        "target_fields": TARGET_FIELDS,
        "split_info": split_info,
        "split_summary": split_summary,
        "modalities": modalities,
        "state_fields": state_fields,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "mean_baseline": mean_baseline,
        "old_linear_baseline": old_linear,
        "checkpoint_used": str(eval_checkpoint_path),
    }
    _save_json(output_dir / "metrics.json", metrics_payload)

    training_summary = {
        "model_name": "go2_vla_baseline_v2",
        "ablation_mode": args.ablation_mode,
        "fusion_type": args.fusion_type,
        "hidden_dim": args.hidden_dim,
        "text_embed_dim": args.text_embed_dim,
        "image_size": args.image_size,
        "image_error_mode": args.image_error_mode,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "vision_learning_rate": args.vision_learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
        "pretrained_vision": not args.no_pretrained_vision,
        "freeze_vision_backbone": args.freeze_vision_backbone,
        "modalities": modalities,
        "state_fields": state_fields,
        "split_info": split_info,
        "split_summary": split_summary,
        "history": history,
        "tokenizer_vocab": tokenizer.to_payload(),
        "checkpoint_used": str(eval_checkpoint_path),
    }
    _save_json(output_dir / "training_summary.json", training_summary)

    summary_lines = [
        "Go2 VLA 基线 v2 汇总",
        f"ablation_mode={args.ablation_mode}",
        f"fusion_type={args.fusion_type}",
        f"modalities={modalities}",
        f"state_fields={state_fields}",
        f"checkpoint_used={eval_checkpoint_path}",
        f"val_overall_rmse={val_metrics['overall_rmse']:.6f}",
        f"test_overall_rmse={test_metrics['overall_rmse']:.6f}",
        f"均值基线 test 总体 RMSE={mean_baseline['test']['overall_rmse']:.6f}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    if args.save_predictions:
        _save_jsonl(output_dir / "predictions_val.jsonl", val_dumps)
        _save_jsonl(output_dir / "predictions_test.jsonl", test_dumps)

    print("验证集：", json.dumps(val_metrics, ensure_ascii=True, sort_keys=True))
    print("测试集：", json.dumps(test_metrics, ensure_ascii=True, sort_keys=True))
    print("mean-baseline 测试集：", json.dumps(mean_baseline["test"], ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

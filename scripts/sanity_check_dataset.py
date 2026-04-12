#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from html import escape
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from llada_vla_common import (
    ACTION_FIELDS,
    CONTROLLED_INSTRUCTIONS,
    QUALITY_LABEL_EXCLUDED,
    QUALITY_REVIEW_FILENAME,
    QUALITY_REVIEW_SCHEMA_VERSION,
    VISUAL_TASK_FAMILIES,
    control_action_from_frame,
    discover_session_roots,
    episode_task_metadata,
    infer_task_family,
    load_episode_derived_labels,
    load_quality_review_index,
    load_json,
    resolved_episode_quality,
    summarize_trajectory_actions,
    summarize_trajectory_metric_series,
)


@dataclass
class FrameRecord:
    timestamp: float
    image_rel: str
    image_disk_path: Path
    instruction: str
    control_action: Dict[str, float]


@dataclass
class EpisodeRecord:
    session_id: str
    episode_id: str
    instruction: str
    capture_mode: str
    task_family: str
    target_type: str
    target_label: str
    target_description: str
    derived_target_side: str
    derived_target_distance: str
    derived_label_source: str
    collector_notes: str
    scene_id: str
    operator_id: str
    original_quality_label: int
    quality_label: int
    quality_overridden: bool
    quality_notes: str
    exclude_from_processing: bool
    frames: List[FrameRecord]
    warnings: List[str]
    info: List[str]
    trajectory_metrics: Dict[str, Any]


CSS = """
body { font-family: sans-serif; margin: 16px; background: #f4f1e8; color: #1c1c1c; }
a { color: #005b7f; }
.card { background: #fffaf0; border: 1px solid #d7ccb8; border-radius: 12px; padding: 12px; margin-bottom: 12px; }
.viewer { position: relative; width: min(960px, 100%); }
.viewer img { width: 100%; height: auto; max-height: min(62vh, 720px); object-fit: contain; border-radius: 12px; display: block; background: #111; }
.overlay { position: absolute; left: 16px; top: 16px; background: rgba(0,0,0,0.65); color: white; padding: 10px 12px; border-radius: 10px; }
.controls { display: flex; gap: 8px; margin: 12px 0; flex-wrap: wrap; align-items: center; }
.controls label { display: inline-flex; align-items: center; gap: 6px; }
button, input[type=range], select { font: inherit; }
.playback-stats { color: #5f5f5f; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.grid img { width: 100%; border-radius: 8px; }
.meta { color: #5f5f5f; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid #e2d7c5; padding: 8px; text-align: left; }
code { background: #efe7d8; padding: 2px 4px; border-radius: 4px; }
svg { width: 100%; height: auto; }
.review-toolbar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 12px; }
.review-toolbar .meta { margin: 0; }
.review-summary { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.review-pill { display: inline-flex; align-items: center; border-radius: 999px; padding: 4px 10px; background: #efe7d8; }
.quality-select, .quality-note, .quality-filter, .quality-score-filter { min-width: 120px; }
.quality-note { width: 100%; max-width: 280px; }
.quality-badge { font-weight: 600; }
.quality-badge.label-0 { color: #6f6658; }
.quality-badge.label-1 { color: #1d6f42; }
.quality-badge.label-2 { color: #8a5b00; }
.quality-badge.label-3 { color: #a12626; }
tr.quality-excluded { background: #fff0ee; }
tr.quality-review { background: #fffaed; }
.card.quality-excluded { border-color: #c96b5a; }
.review-file-status { color: #5f5f5f; }
.page-nav { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }
.episode-layout { display: grid; grid-template-columns: minmax(420px, 1.15fr) minmax(320px, 0.85fr); gap: 12px; align-items: start; }
.episode-stack { display: grid; gap: 12px; }
.meta-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 12px; margin-top: 8px; }
.meta-grid .meta { margin: 0; font-size: 14px; }
.compact-card h2 { margin-top: 0; margin-bottom: 8px; }
@media (max-width: 1100px) { .episode-layout { grid-template-columns: 1fr; } }
"""

QUALITY_LABEL_TEXT = {
    0: "0 未知",
    1: "1 好样本",
    2: "2 可用但不完美",
    3: "3 失败但保留 / 质量差",
}


def _quality_badge_text(quality_label: int) -> str:
    return QUALITY_LABEL_TEXT.get(int(quality_label), f"{int(quality_label)} 未知")


def _quality_select_html(select_id: str, quality_label: int, extra_attrs: str = "") -> str:
    options = []
    for value in (1, 2, 3):
        selected = " selected" if int(quality_label) == value else ""
        options.append(f'<option value="{value}"{selected}>{escape(_quality_badge_text(value))}</option>')
    attrs = f" {extra_attrs.strip()}" if extra_attrs.strip() else ""
    return f'<select id="{escape(select_id)}" class="quality-select"{attrs}>{"".join(options)}</select>'


def _initial_review_payload(episodes: Sequence[EpisodeRecord]) -> Dict[str, Dict[str, Any]]:
    payload: Dict[str, Dict[str, Any]] = {}
    for episode in episodes:
        if not episode.quality_overridden and not episode.quality_notes:
            continue
        key = f"{episode.session_id}::{episode.episode_id}"
        payload[key] = {
            "session_id": episode.session_id,
            "episode_id": episode.episode_id,
            "quality_label": int(episode.quality_label),
            "original_quality_label": int(episode.original_quality_label),
            "quality_notes": episode.quality_notes,
            "exclude_from_processing": bool(episode.exclude_from_processing),
        }
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为 Go2 采集 session 生成本地 HTML 回放与体检报告")
    parser.add_argument("--data-root", type=Path, required=True, help="数据集根目录，或包含多个 session 根目录的父目录")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sanity_check", help="HTML 报告输出目录")
    parser.add_argument("--num-samples", type=int, default=4, help="每条 episode 页面中展示的随机静态帧数量")
    parser.add_argument("--max-episodes", type=int, help="可选的 episode 数量上限，用于冒烟测试")
    parser.add_argument("--seed", type=int, default=0, help="预览采样随机种子")
    parser.add_argument("--serve", action="store_true", help="生成报告后启动本地 HTTP 服务，支持页面直接写回 quality_review.json")
    parser.add_argument("--host", default="127.0.0.1", help="本地 HTTP 服务监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="本地 HTTP 服务端口，默认 8000")
    return parser


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _warn_for_episode(payload: Dict[str, Any], frames: Sequence[FrameRecord]) -> List[str]:
    issues: List[str] = []
    instruction = str(payload.get("instruction") or "")
    task_metadata = episode_task_metadata(payload, {})
    task_family = str(task_metadata.get("task_family") or infer_task_family(instruction))
    if not instruction:
        issues.append("missing_instruction")
    elif task_family == "legacy_motion" and instruction not in CONTROLLED_INSTRUCTIONS:
        issues.append("legacy_motion_instruction_unknown")
    if not payload.get("scene_id"):
        issues.append("missing_scene_id")
    if not payload.get("operator_id"):
        issues.append("missing_operator_id")
    if len(frames) < 2:
        issues.append("too_few_frames")
    return sorted(set(issues))


def _info_for_episode(frames: Sequence[FrameRecord]) -> List[str]:
    info: List[str] = []
    for axis in ACTION_FIELDS:
        values = [frame.control_action[axis] for frame in frames]
        if values and max(values) - min(values) < 1e-6:
            info.append(f"flat_{axis}")
    return sorted(set(info))


def load_episodes(
    data_root: Path,
    max_episodes: Optional[int],
    quality_review_index: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> List[EpisodeRecord]:
    session_roots = discover_session_roots(data_root)
    if not session_roots:
        raise FileNotFoundError(f"no session roots found under {data_root}")

    episodes: List[EpisodeRecord] = []
    for session_root in session_roots:
        index_payload = load_json(session_root / "index.json")
        session_id = str(index_payload.get("session_id") or session_root.name)
        for episode_meta in index_payload.get("episodes") or []:
            if max_episodes is not None and len(episodes) >= max_episodes:
                return episodes
            episode_id = str(episode_meta.get("episode_id") or "")
            if not episode_id:
                continue
            payload = load_json(session_root / "episodes" / f"{episode_id}.json")
            task_metadata = episode_task_metadata(payload, episode_meta)
            derived_labels = load_episode_derived_labels(session_root, episode_id)
            quality_review = resolved_episode_quality(payload, episode_meta, quality_review_index, session_id, episode_id)
            frames: List[FrameRecord] = []
            for frame in payload.get("frames") or []:
                image_rel = str(frame.get("image") or "")
                if not image_rel:
                    continue
                image_disk_path = session_root / image_rel
                if not image_disk_path.exists():
                    continue
                frames.append(
                    FrameRecord(
                        timestamp=_safe_float(frame.get("timestamp")),
                        image_rel=image_rel,
                        image_disk_path=image_disk_path,
                        instruction=str(frame.get("instruction") or payload.get("instruction") or ""),
                        control_action=control_action_from_frame(frame),
                    )
                )
            episode = EpisodeRecord(
                session_id=session_id,
                episode_id=episode_id,
                instruction=str(payload.get("instruction") or ""),
                capture_mode=str(task_metadata.get("capture_mode") or ""),
                task_family=str(task_metadata.get("task_family") or ""),
                target_type=str(task_metadata.get("target_type") or ""),
                target_label=str(task_metadata.get("target_label") or ""),
                target_description=str(task_metadata.get("target_description") or ""),
                derived_target_side=str(derived_labels.get("target_side_band") or ""),
                derived_target_distance=str(derived_labels.get("target_distance_band") or ""),
                derived_label_source=str(derived_labels.get("label_source") or ""),
                collector_notes=str(task_metadata.get("collector_notes") or ""),
                scene_id=str(payload.get("scene_id") or episode_meta.get("scene_id") or ""),
                operator_id=str(payload.get("operator_id") or episode_meta.get("operator_id") or ""),
                original_quality_label=int(quality_review.get("original_quality_label") or 0),
                quality_label=int(quality_review.get("quality_label") or 0),
                quality_overridden=bool(quality_review.get("quality_overridden")),
                quality_notes=str(quality_review.get("quality_notes") or ""),
                exclude_from_processing=bool(quality_review.get("exclude_from_processing")),
                frames=frames,
                warnings=[],
                info=[],
                trajectory_metrics={},
            )
            episode.warnings = _warn_for_episode(payload, frames)
            episode.info = _info_for_episode(frames)
            episode.trajectory_metrics = summarize_trajectory_actions(
                [frame.control_action for frame in frames],
                [frame.timestamp for frame in frames],
            )
            episodes.append(episode)
    return episodes


def collect_session_stats(data_root: Path) -> Tuple[int, Dict[str, int]]:
    session_roots = discover_session_roots(data_root)
    session_episode_counts: Dict[str, int] = {}
    empty_session_count = 0
    for session_root in session_roots:
        index_payload = load_json(session_root / "index.json")
        session_id = str(index_payload.get("session_id") or session_root.name)
        episode_count = len(index_payload.get("episodes") or [])
        session_episode_counts[session_id] = episode_count
        if episode_count == 0:
            empty_session_count += 1
    return empty_session_count, dict(sorted(session_episode_counts.items()))


def _default_quality_review_payload(data_root: Path) -> Dict[str, Any]:
    return {
        "schema_version": QUALITY_REVIEW_SCHEMA_VERSION,
        "dataset_root": str(data_root),
        "excluded_quality_label": QUALITY_LABEL_EXCLUDED,
        "review_filename_hint": QUALITY_REVIEW_FILENAME,
        "semantics": "quality_label stores an override for the original recording score (1/2/3)",
        "episodes": [],
    }


def _load_quality_review_payload(data_root: Path) -> Dict[str, Any]:
    review_path = data_root / QUALITY_REVIEW_FILENAME
    if not review_path.exists():
        return _default_quality_review_payload(data_root)
    payload = load_json(review_path)
    if not isinstance(payload, dict):
        return _default_quality_review_payload(data_root)
    result = _default_quality_review_payload(data_root)
    result.update(payload)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        result["episodes"] = []
    return result


def _normalize_quality_review_payload(data_root: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _default_quality_review_payload(data_root)
    normalized["dataset_root"] = str(data_root)
    normalized["episodes"] = []
    for item in payload.get("episodes") or []:
        if not isinstance(item, dict):
            continue
        session_id = str(item.get("session_id") or "").strip()
        episode_id = str(item.get("episode_id") or "").strip()
        if not session_id or not episode_id:
            continue
        quality_label = int(item.get("quality_label") or 0)
        if quality_label < 0:
            quality_label = 0
        if quality_label > 3:
            quality_label = 3
        original_quality_label = int(item.get("original_quality_label") or 0)
        if original_quality_label < 0:
            original_quality_label = 0
        if original_quality_label > 3:
            original_quality_label = 3
        normalized_item = {
            "session_id": session_id,
            "episode_id": episode_id,
            "quality_label": quality_label,
            "original_quality_label": original_quality_label,
            "exclude_from_processing": bool(item.get("exclude_from_processing")) or quality_label == QUALITY_LABEL_EXCLUDED,
        }
        quality_notes = str(item.get("quality_notes") or item.get("notes") or "").strip()
        if quality_notes:
            normalized_item["quality_notes"] = quality_notes
        normalized["episodes"].append(normalized_item)
    normalized["episodes"].sort(key=lambda item: (item["session_id"], item["episode_id"]))
    return normalized


def _write_quality_review_payload(data_root: Path, payload: Dict[str, Any]) -> Path:
    review_path = data_root / QUALITY_REVIEW_FILENAME
    review_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_quality_review_payload(data_root, payload)
    review_path.write_text(json.dumps(normalized, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return review_path


def serve_report(output_dir: Path, data_root: Path, host: str, port: int) -> None:
    output_dir = output_dir.resolve()
    data_root = data_root.resolve()

    class ReportRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(output_dir), **kwargs)

        def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path) -> None:
            mime_type, _ = mimetypes.guess_type(str(path))
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", (mime_type or "application/octet-stream"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/review/load":
                self._send_json(_load_quality_review_payload(data_root))
                return
            if parsed.path == "/api/file":
                query = parse_qs(parsed.query)
                raw_path = str((query.get("path") or [""])[0]).strip()
                try:
                    requested_path = Path(raw_path).resolve(strict=True)
                    requested_path.relative_to(data_root)
                    if not requested_path.is_file():
                        raise FileNotFoundError(str(requested_path))
                    self._send_file(requested_path)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            super().do_GET()

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/review/save":
                self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            content_length = int(self.headers.get("Content-Length", "0") or 0)
            try:
                raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                payload = json.loads(raw_body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("payload must be object")
                review_path = _write_quality_review_payload(data_root, payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True, "path": str(review_path)})

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer((host, port), ReportRequestHandler)
    url = f"http://{host}:{port}/index.html"
    print(f"本地报告服务已启动：{url}")
    print(f"页面修改会直接写回：{data_root / QUALITY_REVIEW_FILENAME}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n收到中断，正在停止本地报告服务...")
    finally:
        server.server_close()


def _polyline(values: Sequence[float], width: int, height: int, minimum: Optional[float] = None, maximum: Optional[float] = None) -> str:
    if not values:
        return ""
    minimum = min(values) if minimum is None else minimum
    maximum = max(values) if maximum is None else maximum
    if math.isclose(minimum, maximum):
        minimum -= 1.0
        maximum += 1.0
    points = []
    for index, value in enumerate(values):
        x = 0.0 if len(values) == 1 else (index / (len(values) - 1)) * width
        y = height - ((value - minimum) / (maximum - minimum)) * height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _plot_svg(episode: EpisodeRecord) -> str:
    width = 720
    height = 150
    colors = {"vx": "#cc5500", "vy": "#007f5f", "wz": "#2b59c3"}
    values = {axis: [frame.control_action[axis] for frame in episode.frames] for axis in ACTION_FIELDS}
    lines = []
    for axis in ACTION_FIELDS:
        lines.append(
            f'<polyline fill="无" stroke="{colors[axis]}" stroke-width="3" points="{_polyline(values[axis], width, height, -1.0, 1.0)}" />'
        )
    legend = " ".join(
        f'<text x="{20 + index * 140}" y="20" fill="{colors[axis]}" font-size="14">{axis}</text>'
        for index, axis in enumerate(ACTION_FIELDS)
    )
    return (
        f'<svg viewBox="0 0 {width} {height + 30}" role="img" aria-label="action plot">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff" stroke="#d7ccb8" />'
        f'<line x1="0" y1="{height / 2:.1f}" x2="{width}" y2="{height / 2:.1f}" stroke="#b8b0a0" stroke-dasharray="4 4" />'
        f'{"".join(lines)}{legend}</svg>'
    )


def _histogram_svg(values: Sequence[float], axis: str) -> str:
    width = 180
    height = 96
    bins = 12
    counts = [0] * bins
    for value in values:
        clamped = max(-1.0, min(1.0, value))
        index = min(bins - 1, max(0, int((clamped + 1.0) / 2.0 * bins)))
        counts[index] += 1
    peak = max(counts) if counts else 1
    bars = []
    bar_width = width / bins
    for index, count in enumerate(counts):
        bar_height = 0 if peak == 0 else (count / peak) * height
        x = index * bar_width
        y = height - bar_height
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 2:.1f}" height="{bar_height:.1f}" fill="#7a9e9f" />')
    return (
        f'<div><strong>{escape(axis)}</strong>'
        f'<svg viewBox="0 0 {width} {height + 20}"><rect x="0" y="0" width="{width}" height="{height}" fill="#fff" stroke="#d7ccb8" />'
        f'{"".join(bars)}<text x="0" y="{height + 16}" font-size="12">-1.0</text>'
        f'<text x="{width - 28}" y="{height + 16}" font-size="12">1.0</text></svg></div>'
    )


def _review_shared_script(data_root: Path, episode_refs_payload: Sequence[Dict[str, Any]], initial_reviews: Dict[str, Dict[str, Any]]) -> str:
    return f"""
    const datasetRoot = {json.dumps(str(data_root), ensure_ascii=True)};
    const qualityReviewFilename = {json.dumps(QUALITY_REVIEW_FILENAME, ensure_ascii=True)};
    const qualityReviewSchemaVersion = {json.dumps(QUALITY_REVIEW_SCHEMA_VERSION, ensure_ascii=True)};
    const excludedQualityLabel = {QUALITY_LABEL_EXCLUDED};
    const episodeRefs = {json.dumps(list(episode_refs_payload), ensure_ascii=True)};
    const initialReviews = {json.dumps(initial_reviews, ensure_ascii=True)};
    const reviewStorageKey = `go2_quality_review::${{datasetRoot}}`;
    const episodeRefsByKey = Object.fromEntries(
      episodeRefs.map((item) => [`${{item.session_id}}::${{item.episode_id}}`, item]),
    );
    let reviewState = {{}};
    let serverAutosaveAvailable = false;
    let autosaveHandle = null;
    let autosaveReady = false;

    function reviewHandleDb() {{
      return new Promise((resolve, reject) => {{
        if (!window.indexedDB) {{
          resolve(null);
          return;
        }}
        const request = window.indexedDB.open('go2_quality_review_handles', 1);
        request.onupgradeneeded = () => {{
          const db = request.result;
          if (!db.objectStoreNames.contains('handles')) {{
            db.createObjectStore('handles');
          }}
        }};
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      }});
    }}

    async function loadStoredHandle() {{
      const db = await reviewHandleDb();
      if (!db) {{
        return null;
      }}
      return await new Promise((resolve, reject) => {{
        const tx = db.transaction('handles', 'readonly');
        const store = tx.objectStore('handles');
        const request = store.get(reviewStorageKey);
        request.onsuccess = () => resolve(request.result || null);
        request.onerror = () => reject(request.error);
      }});
    }}

    async function storeHandle(handle) {{
      const db = await reviewHandleDb();
      if (!db) {{
        return;
      }}
      await new Promise((resolve, reject) => {{
        const tx = db.transaction('handles', 'readwrite');
        const store = tx.objectStore('handles');
        const request = store.put(handle, reviewStorageKey);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      }});
    }}

    async function clearStoredHandle() {{
      const db = await reviewHandleDb();
      if (!db) {{
        return;
      }}
      await new Promise((resolve, reject) => {{
        const tx = db.transaction('handles', 'readwrite');
        const store = tx.objectStore('handles');
        const request = store.delete(reviewStorageKey);
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      }});
    }}

    function makeEpisodeKey(sessionId, episodeId) {{
      return `${{String(sessionId || '').trim()}}::${{String(episodeId || '').trim()}}`;
    }}

    function labelText(label) {{
      switch (Number(label) || 0) {{
        case 1:
          return '1 好样本';
        case 2:
          return '2 可用但不完美';
        case 3:
          return '3 失败但保留 / 质量差';
        default:
          return '0 未知';
      }}
    }}

    function labelClass(label) {{
      return `label-${{Number(label) || 0}}`;
    }}

    function normalizeReview(raw, fallbackRef) {{
      const ref = fallbackRef || {{}};
      const originalQualityLabel = Math.max(0, Math.min(3, Number((raw && raw.original_quality_label) || ref.original_quality_label) || 0));
      const overrideQualityLabel = Math.max(0, Math.min(3, Number(raw && raw.quality_label) || 0));
      const qualityNotes = String((raw && (raw.quality_notes || raw.notes)) || '').trim();
      const effectiveQualityLabel = overrideQualityLabel > 0 ? overrideQualityLabel : originalQualityLabel;
      return {{
        session_id: String((raw && raw.session_id) || ref.session_id || '').trim(),
        episode_id: String((raw && raw.episode_id) || ref.episode_id || '').trim(),
        original_quality_label: originalQualityLabel,
        quality_label: effectiveQualityLabel,
        override_quality_label: overrideQualityLabel,
        quality_overridden: overrideQualityLabel > 0 && overrideQualityLabel !== originalQualityLabel,
        quality_notes: qualityNotes,
        exclude_from_processing: effectiveQualityLabel === excludedQualityLabel || Boolean(raw && raw.exclude_from_processing),
      }};
    }}

    function mergeReviewSource(source) {{
      if (!source || typeof source !== 'object') {{
        return;
      }}
      Object.entries(source).forEach(([key, value]) => {{
        const normalizedKey = key.includes('::')
          ? key
          : makeEpisodeKey(value && value.session_id, value && value.episode_id);
        if (!normalizedKey) {{
          return;
        }}
        const fallbackRef = episodeRefsByKey[normalizedKey] || value || {{}};
        const normalized = normalizeReview(value || {{}}, fallbackRef);
        if (!normalized.session_id || !normalized.episode_id) {{
          return;
        }}
        if (!normalized.quality_overridden && !normalized.quality_notes) {{
          delete reviewState[normalizedKey];
          return;
        }}
        reviewState[normalizedKey] = {{
          session_id: normalized.session_id,
          episode_id: normalized.episode_id,
          original_quality_label: normalized.original_quality_label,
          quality_label: normalized.override_quality_label,
          quality_notes: normalized.quality_notes,
          exclude_from_processing: normalized.exclude_from_processing,
        }};
      }});
    }}

    function payloadEpisodesToSource(payload) {{
      const result = {{}};
      const episodes = Array.isArray(payload && payload.episodes) ? payload.episodes : [];
      episodes.forEach((item) => {{
        const key = makeEpisodeKey(item && item.session_id, item && item.episode_id);
        if (!key) {{
          return;
        }}
        result[key] = item;
      }});
      return result;
    }}

    function loadStoredReviews() {{
      try {{
        const raw = window.localStorage.getItem(reviewStorageKey);
        if (!raw) {{
          return {{}};
        }}
        const payload = JSON.parse(raw);
        return payload && typeof payload === 'object' ? payload : {{}};
      }} catch (error) {{
        return {{}};
      }}
    }}

    function persistReviews() {{
      try {{
        window.localStorage.setItem(reviewStorageKey, JSON.stringify(reviewState));
      }} catch (error) {{
        announceStatus(`保存浏览器缓存失败：${{error}}`, true);
      }}
    }}

    function seedReviews() {{
      reviewState = {{}};
      mergeReviewSource(initialReviews);
      mergeReviewSource(loadStoredReviews());
      persistReviews();
    }}

    function getEpisodeReview(sessionId, episodeId) {{
      const key = makeEpisodeKey(sessionId, episodeId);
      const fallbackRef = episodeRefsByKey[key] || {{ session_id: sessionId, episode_id: episodeId }};
      return normalizeReview(reviewState[key] || {{}}, fallbackRef);
    }}

    function upsertEpisodeReview(sessionId, episodeId, qualityLabel, qualityNotes) {{
      const key = makeEpisodeKey(sessionId, episodeId);
      const fallbackRef = episodeRefsByKey[key] || {{ session_id: sessionId, episode_id: episodeId }};
      const originalQualityLabel = Number(fallbackRef.original_quality_label) || 0;
      const overrideQualityLabel = Number(qualityLabel) || 0;
      const normalized = normalizeReview({{
        session_id: fallbackRef.session_id,
        episode_id: fallbackRef.episode_id,
        original_quality_label: originalQualityLabel,
        quality_label: overrideQualityLabel,
        quality_notes: qualityNotes,
      }}, fallbackRef);
      if (!normalized.quality_overridden && !normalized.quality_notes) {{
        delete reviewState[key];
      }} else {{
        reviewState[key] = {{
          session_id: normalized.session_id,
          episode_id: normalized.episode_id,
          original_quality_label: normalized.original_quality_label,
          quality_label: normalized.override_quality_label,
          quality_notes: normalized.quality_notes,
          exclude_from_processing: normalized.exclude_from_processing,
        }};
      }}
      persistReviews();
      return getEpisodeReview(sessionId, episodeId);
    }}

    function clearEpisodeOverride(sessionId, episodeId) {{
      const key = makeEpisodeKey(sessionId, episodeId);
      delete reviewState[key];
      persistReviews();
      return getEpisodeReview(sessionId, episodeId);
    }}

    function exportableReviews() {{
      return Object.values(reviewState)
        .filter((item) => item.session_id && item.episode_id && ((Number(item.quality_label) || 0) > 0 || item.quality_notes))
        .sort((left, right) => {{
          const sessionCmp = left.session_id.localeCompare(right.session_id);
          if (sessionCmp !== 0) {{
            return sessionCmp;
          }}
          return left.episode_id.localeCompare(right.episode_id);
        }});
    }}

    function buildExportPayload() {{
      return {{
        schema_version: qualityReviewSchemaVersion,
        dataset_root: datasetRoot,
        excluded_quality_label: excludedQualityLabel,
        review_filename_hint: qualityReviewFilename,
        semantics: 'quality_label stores an override for the original recording score (1/2/3)',
        episodes: exportableReviews(),
      }};
    }}

    async function loadServerReviewPayload() {{
      const response = await fetch('/api/review/load', {{
        method: 'GET',
        cache: 'no-store',
        headers: {{ 'Accept': 'application/json' }},
      }});
      if (!response.ok) {{
        throw new Error(`server load failed: ${{response.status}}`);
      }}
      return await response.json();
    }}

    async function saveServerReviewPayload(announceMessage = '') {{
      const response = await fetch('/api/review/save', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(buildExportPayload()),
      }});
      if (!response.ok) {{
        let errorMessage = `server save failed: ${{response.status}}`;
        try {{
          const payload = await response.json();
          if (payload && payload.error) {{
            errorMessage = String(payload.error);
          }}
        }} catch (error) {{
        }}
        throw new Error(errorMessage);
      }}
      serverAutosaveAvailable = true;
      updateBoundFileStatus();
      if (announceMessage) {{
        announceStatus(announceMessage);
      }}
      return true;
    }}

    async function ensureHandlePermission(handle) {{
      if (!handle || typeof handle.queryPermission !== 'function') {{
        return false;
      }}
      if ((await handle.queryPermission({{ mode: 'readwrite' }})) === 'granted') {{
        return true;
      }}
      return (await handle.requestPermission({{ mode: 'readwrite' }})) === 'granted';
    }}

    function updateBoundFileStatus() {{
      const node = document.getElementById('reviewFileStatus');
      if (!node) {{
        return;
      }}
      if (serverAutosaveAvailable) {{
        node.textContent = '已连接本地报告服务，修改后会直接写回 quality_review.json';
        return;
      }}
      if (autosaveHandle && autosaveReady) {{
        node.textContent = `已绑定本地文件：${{autosaveHandle.name}}，修改后自动写回`;
        return;
      }}
      if (!window.showSaveFilePicker) {{
        node.textContent = '当前浏览器不支持直接写回本地文件，只能先缓存到浏览器。';
        return;
      }}
      node.textContent = '当前仅写入浏览器缓存；点击“绑定回调文件”后会自动写回本地 quality_review.json。';
    }}

    async function flushReviewsToBoundFile(announceMessage = '') {{
      if (!autosaveHandle) {{
        updateBoundFileStatus();
        return false;
      }}
      if (!(await ensureHandlePermission(autosaveHandle))) {{
        autosaveReady = false;
        updateBoundFileStatus();
        announceStatus('本地文件写权限未授权，当前只保存在浏览器缓存中。', true);
        return false;
      }}
      const text = `${{JSON.stringify(buildExportPayload(), null, 2)}}\\n`;
      const writable = await autosaveHandle.createWritable();
      await writable.write(text);
      await writable.close();
      autosaveReady = true;
      updateBoundFileStatus();
      if (announceMessage) {{
        announceStatus(announceMessage);
      }}
      return true;
    }}

    async function bindReviewFile() {{
      if (serverAutosaveAvailable) {{
        announceStatus('当前已连接本地报告服务，页面修改会直接写回源文件。');
        return;
      }}
      if (!window.showSaveFilePicker) {{
        announceStatus('当前浏览器不支持直接写回本地文件，请改用导出 JSON。', true);
        return;
      }}
      const handle = await window.showSaveFilePicker({{
        suggestedName: qualityReviewFilename,
        types: [{{ description: 'JSON', accept: {{ 'application/json': ['.json'] }} }}],
      }});
      autosaveHandle = handle;
      autosaveReady = false;
      await storeHandle(handle);
      await flushReviewsToBoundFile(`已绑定并写回 ${{handle.name}}`);
    }}

    async function restoreBoundFile() {{
      try {{
        const handle = await loadStoredHandle();
        if (!handle) {{
          updateBoundFileStatus();
          return;
        }}
        autosaveHandle = handle;
        autosaveReady = (await ensureHandlePermission(handle));
        updateBoundFileStatus();
      }} catch (error) {{
        autosaveHandle = null;
        autosaveReady = false;
        await clearStoredHandle();
        updateBoundFileStatus();
      }}
    }}

    async function persistReviewState(announceMessage = '') {{
      if (serverAutosaveAvailable) {{
        try {{
          await saveServerReviewPayload(announceMessage);
          return;
        }} catch (error) {{
          serverAutosaveAvailable = false;
          updateBoundFileStatus();
          announceStatus(`本地服务写回失败：${{error}}`, true);
        }}
      }}
      if (autosaveHandle) {{
        try {{
          const saved = await flushReviewsToBoundFile(announceMessage);
          if (saved) {{
            return;
          }}
        }} catch (error) {{
          autosaveReady = false;
          updateBoundFileStatus();
          announceStatus(`自动写回失败：${{error}}`, true);
          return;
        }}
      }}
      if (announceMessage) {{
        announceStatus(`${{announceMessage}}（当前仅保存在浏览器缓存）`);
      }}
    }}

    async function exportReviewFile() {{
      const text = `${{JSON.stringify(buildExportPayload(), null, 2)}}\\n`;
      if (window.showSaveFilePicker) {{
        const handle = await window.showSaveFilePicker({{
          suggestedName: qualityReviewFilename,
          types: [{{ description: 'JSON', accept: {{ 'application/json': ['.json'] }} }}],
        }});
        const writable = await handle.createWritable();
        await writable.write(text);
        await writable.close();
        announceStatus(`已保存回调结果，建议放到 ${{datasetRoot}}/${{qualityReviewFilename}}`);
        return;
      }}
      const blob = new Blob([text], {{ type: 'application/json' }});
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = qualityReviewFilename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      announceStatus(`已导出回调结果，请将下载的文件保存到 ${{datasetRoot}}/${{qualityReviewFilename}}`);
    }}

    async function importReviewFile(file) {{
      const text = await file.text();
      const payload = JSON.parse(text);
      const episodes = Array.isArray(payload && payload.episodes) ? payload.episodes : [];
      const imported = {{}};
      episodes.forEach((item) => {{
        const key = makeEpisodeKey(item && item.session_id, item && item.episode_id);
        if (!key) {{
          return;
        }}
        imported[key] = item;
      }});
      mergeReviewSource(imported);
      persistReviews();
      announceStatus(`已导入 ${{episodes.length}} 条回调记录`);
    }}

    function clearReviewCache() {{
      reviewState = {{}};
      persistReviews();
    }}

    function announceStatus(message, isError = false) {{
      const node = document.getElementById('reviewStatus');
      if (!node) {{
        return;
      }}
      node.textContent = message;
      node.style.color = isError ? '#a12626' : '#5f5f5f';
    }}

    async function initializeServerAutosave() {{
      if (!String(window.location.protocol || '').startsWith('http')) {{
        updateBoundFileStatus();
        return;
      }}
      try {{
        const payload = await loadServerReviewPayload();
        reviewState = {{}};
        mergeReviewSource(payloadEpisodesToSource(payload));
        persistReviews();
        serverAutosaveAvailable = true;
      }} catch (error) {{
        serverAutosaveAvailable = false;
      }}
      updateBoundFileStatus();
    }}

    seedReviews();
    initializeServerAutosave()
      .catch(() => updateBoundFileStatus())
      .finally(() => {{
        if (!serverAutosaveAvailable) {{
          restoreBoundFile().catch(() => updateBoundFileStatus());
        }}
      }});
    """


def _episode_page(
    episode: EpisodeRecord,
    page_path: Path,
    index_path: Path,
    rng: random.Random,
    num_samples: int,
    data_root: Path,
    episode_refs_payload: Sequence[Dict[str, Any]],
    initial_reviews: Dict[str, Dict[str, Any]],
    prev_page_name: Optional[str],
    next_page_name: Optional[str],
) -> str:
    frames_payload = []
    for frame in episode.frames:
        frames_payload.append(
            {
                "image": os.path.relpath(frame.image_disk_path, page_path.parent),
                "disk_path": str(frame.image_disk_path),
                "timestamp": round(frame.timestamp, 6),
                "instruction": frame.instruction,
                "vx": frame.control_action["vx"],
                "vy": frame.control_action["vy"],
                "wz": frame.control_action["wz"],
            }
        )
    histogram_html = "".join(_histogram_svg([frame.control_action[axis] for frame in episode.frames], axis) for axis in ACTION_FIELDS)
    warnings = "无" if not episode.warnings else ", ".join(episode.warnings)
    info = "无" if not episode.info else ", ".join(episode.info)
    index_rel = os.path.relpath(index_path, page_path.parent)
    current_quality_text = _quality_badge_text(episode.quality_label)
    original_quality_text = _quality_badge_text(episode.original_quality_label)
    current_processing_text = "已排除" if episode.exclude_from_processing else "保留"
    review_script = _review_shared_script(data_root, episode_refs_payload, initial_reviews)
    review_select_html = _quality_select_html("qualityLabel", episode.quality_label)
    review_panel_class = "card quality-excluded" if episode.exclude_from_processing else "card"
    prev_link_html = f'<a href="{escape(prev_page_name)}">Prev Episode</a>' if prev_page_name else '<span class="meta">Prev Episode</span>'
    next_link_html = f'<a href="{escape(next_page_name)}">Next Episode</a>' if next_page_name else '<span class="meta">Next Episode</span>'

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{escape(episode.session_id)} {escape(episode.episode_id)}</title>
  <style>{CSS}</style>
</head>
<body>
  <div class=\"page-nav\">
    <a href=\"{escape(index_rel)}\">Back to summary</a>
    {prev_link_html}
    {next_link_html}
  </div>
  <div class=\"episode-layout\">
    <div class=\"episode-stack\">
      <div class=\"{review_panel_class} compact-card\" id=\"reviewPanel\">
        <h1>{escape(episode.session_id)} / {escape(episode.episode_id)}</h1>
        <p class=\"meta\"><strong>instruction</strong> <code>{escape(episode.instruction)}</code></p>
        <div class=\"meta-grid\">
          <p class=\"meta\">capture_mode=<code>{escape(episode.capture_mode or '-')}</code></p>
          <p class=\"meta\">task_family=<code>{escape(episode.task_family or '-')}</code></p>
          <p class=\"meta\">target_type=<code>{escape(episode.target_type or '-')}</code></p>
          <p class=\"meta\">target_label=<code>{escape(episode.target_label or '-')}</code></p>
          <p class=\"meta\">derived_side=<code>{escape(episode.derived_target_side or '-')}</code></p>
          <p class=\"meta\">derived_distance=<code>{escape(episode.derived_target_distance or '-')}</code></p>
          <p class=\"meta\">scene=<code>{escape(episode.scene_id or '-')}</code></p>
          <p class=\"meta\">operator=<code>{escape(episode.operator_id or '-')}</code></p>
          <p class=\"meta\">duration_s=<code>{episode.trajectory_metrics.get('duration_seconds', 0.0):.3f}</code></p>
          <p class=\"meta\">action_changes=<code>{episode.trajectory_metrics.get('action_change_count', 0)}</code></p>
          <p class=\"meta\">stop_ratio=<code>{episode.trajectory_metrics.get('stop_ratio', 0.0):.1%}</code></p>
          <p class=\"meta\">turn_ratio=<code>{episode.trajectory_metrics.get('turn_ratio', 0.0):.1%}</code></p>
          <p class=\"meta\">warnings=<code>{escape(warnings)}</code></p>
          <p class=\"meta\">info=<code>{escape(info)}</code></p>
        </div>
        <h2>录制分数回调</h2>
        <p class=\"meta\">当前页面直接回调录制时分数。<code>3</code> 会从后续处理里排除；如需恢复，只要改回 <code>2</code> 并重新导出 <code>{escape(str(data_root / QUALITY_REVIEW_FILENAME))}</code>。</p>
        <div class=\"review-toolbar\">
          <label>当前分数 {review_select_html}</label>
          <label>备注 <input id=\"qualityNotes\" class=\"quality-note\" type=\"text\" value=\"{escape(episode.quality_notes)}\" placeholder=\"可选：记录回调原因\" /></label>
          <button id=\"bindReviewFile\">绑定回调文件</button>
          <button id=\"exportReview\">导出回调 JSON</button>
          <button id=\"resetReview\">恢复原始分数</button>
          <label>导入回调 JSON <input id=\"importReviewFile\" type=\"file\" accept=\".json,application/json\" /></label>
          <span id=\"reviewFileStatus\" class=\"review-file-status\"></span>
        </div>
        <div class=\"review-summary\">
          <span class=\"review-pill\">原始分数 <span id=\"originalQualityBadge\" class=\"quality-badge {escape(f'label-{episode.original_quality_label}')}">{escape(original_quality_text)}</span></span>
          <span class=\"review-pill\">当前分数 <span id=\"currentQualityBadge\" class=\"quality-badge {escape(f'label-{episode.quality_label}')}">{escape(current_quality_text)}</span></span>
          <span class=\"review-pill\">后续处理 <span id=\"currentProcessingBadge\">{escape(current_processing_text)}</span></span>
        </div>
        <p id=\"reviewStatus\" class=\"meta\">修改后会立即尝试写回绑定的本地文件；未绑定时先保存在浏览器缓存中。</p>
      </div>
      <div class=\"card compact-card\">
        <div class=\"viewer\">
          <img id=\"frameImage\" src=\"\" alt=\"episode replay frame\" />
          <div class=\"overlay\" id=\"overlay\"></div>
        </div>
        <div class=\"controls\">
          <button id=\"playPause\">Play</button>
          <button id=\"prevFrame\">Prev Frame</button>
          <button id=\"nextFrame\">Next Frame</button>
          <label>Frame <input id=\"frameSlider\" type=\"range\" min=\"0\" max=\"{max(0, len(frames_payload) - 1)}\" value=\"0\" /></label>
          <label>Speed
            <select id=\"speedSelect\">
              <option value=\"0.5\">0.5x</option>
              <option value=\"0.75\">0.75x</option>
              <option value=\"1\" selected>1.0x</option>
              <option value=\"1.5\">1.5x</option>
              <option value=\"2\">2.0x</option>
            </select>
          </label>
          <span class=\"playback-stats\" id=\"playbackStats\"></span>
        </div>
      </div>
    </div>
    <div class=\"episode-stack\">
      <div class=\"card compact-card\">
        <h2>Action vs time</h2>
        {_plot_svg(episode)}
      </div>
      <div class=\"card compact-card\">
        <h2>Action histograms</h2>
        <div class=\"grid\">{histogram_html}</div>
      </div>
    </div>
  </div>
  <script>
    {review_script}
    const frames = {json.dumps(frames_payload, ensure_ascii=True)};
    const currentEpisode = {json.dumps({"session_id": episode.session_id, "episode_id": episode.episode_id}, ensure_ascii=True)};
    const serverFileMode = String(window.location.protocol || '').startsWith('http');
    let index = 0;
    let timer = null;
    let playing = false;
    let playbackSpeed = 1.0;
    const image = document.getElementById('frameImage');
    const overlay = document.getElementById('overlay');
    const slider = document.getElementById('frameSlider');
    const playPause = document.getElementById('playPause');
    const speedSelect = document.getElementById('speedSelect');
    const playbackStats = document.getElementById('playbackStats');
    const qualityLabel = document.getElementById('qualityLabel');
    const qualityNotes = document.getElementById('qualityNotes');
    const originalQualityBadge = document.getElementById('originalQualityBadge');
    const currentQualityBadge = document.getElementById('currentQualityBadge');
    const currentProcessingBadge = document.getElementById('currentProcessingBadge');
    const reviewPanel = document.getElementById('reviewPanel');

    image.addEventListener('load', () => {{
      if (image.naturalWidth > 0 && image.naturalHeight > 0) {{
        image.style.aspectRatio = `${{image.naturalWidth}} / ${{image.naturalHeight}}`;
        image.style.height = 'auto';
      }}
    }});

    function renderCurrentReview() {{
      const review = getEpisodeReview(currentEpisode.session_id, currentEpisode.episode_id);
      qualityLabel.value = String(review.quality_label);
      qualityNotes.value = review.quality_notes;
      originalQualityBadge.textContent = labelText(review.original_quality_label);
      originalQualityBadge.className = `quality-badge ${{labelClass(review.original_quality_label)}}`;
      currentQualityBadge.textContent = labelText(review.quality_label);
      currentQualityBadge.className = `quality-badge ${{labelClass(review.quality_label)}}`;
      currentProcessingBadge.textContent = review.exclude_from_processing ? '已排除' : '保留';
      reviewPanel.classList.toggle('quality-excluded', review.exclude_from_processing);
    }}

    function saveCurrentReview() {{
      const review = upsertEpisodeReview(
        currentEpisode.session_id,
        currentEpisode.episode_id,
        Number(qualityLabel.value) || 0,
        qualityNotes.value,
      );
      renderCurrentReview();
      persistReviewState(`已更新 ${{currentEpisode.session_id}}/${{currentEpisode.episode_id}} -> ${{labelText(review.quality_label)}}`)
        .catch((error) => announceStatus(`写回失败：${{error}}`, true));
    }}

    function clamp(value, minValue, maxValue) {{
      return Math.min(maxValue, Math.max(minValue, value));
    }}

    function frameImageUrl(frame) {{
      if (serverFileMode && frame.disk_path) {{
        return `/api/file?path=${{encodeURIComponent(frame.disk_path)}}`;
      }}
      return frame.image;
    }}

    function computeNominalFrameGapMs() {{
      const gaps = [];
      for (let i = 1; i < frames.length; i += 1) {{
        const gapMs = (frames[i].timestamp - frames[i - 1].timestamp) * 1000;
        if (Number.isFinite(gapMs) && gapMs > 1) {{
          gaps.push(gapMs);
        }}
      }}
      if (!gaps.length) {{
        return 100;
      }}
      gaps.sort((left, right) => left - right);
      return clamp(gaps[Math.floor(gaps.length / 2)], 33, 250);
    }}

    const nominalFrameGapMs = computeNominalFrameGapMs();

    function preloadAround(currentIndex) {{
      for (let offset = 0; offset <= 2; offset += 1) {{
        const nextIndex = currentIndex + offset;
        if (nextIndex >= frames.length) {{
          break;
        }}
        const preload = new Image();
        preload.src = frameImageUrl(frames[nextIndex]);
      }}
    }}

    function nextFrameDelayMs(currentIndex) {{
      if (frames.length <= 1) {{
        return nominalFrameGapMs / playbackSpeed;
      }}
      const nextFrame = frames[currentIndex + 1];
      if (!nextFrame) {{
        return nominalFrameGapMs / playbackSpeed;
      }}
      const gapMs = (nextFrame.timestamp - frames[currentIndex].timestamp) * 1000;
      const safeGapMs = Number.isFinite(gapMs) && gapMs > 1 ? gapMs : nominalFrameGapMs;
      return clamp(safeGapMs / playbackSpeed, 16, 250);
    }}

    function updatePlaybackStats() {{
      if (!frames.length) {{
        playbackStats.textContent = '0 frames';
        return;
      }}
      const nominalFps = 1000 / nominalFrameGapMs;
      playbackStats.textContent = `frame ${{index + 1}}/${{frames.length}} | nominal ${{nominalFps.toFixed(1)}} FPS | speed ${{playbackSpeed.toFixed(2)}}x`;
    }}

    function stopPlayback() {{
      playing = false;
      if (timer !== null) {{
        clearTimeout(timer);
        timer = null;
      }}
      playPause.textContent = 'Play';
    }}

    function scheduleNextFrame() {{
      if (!playing || !frames.length) {{
        return;
      }}
      if (timer !== null) {{
        clearTimeout(timer);
      }}
      timer = window.setTimeout(() => {{
        timer = null;
        if (!playing) {{
          return;
        }}
        index = (index + 1) % frames.length;
        render();
        scheduleNextFrame();
      }}, nextFrameDelayMs(index));
    }}

    function render() {{
      if (!frames.length) {{
        overlay.textContent = 'No frames available';
        updatePlaybackStats();
        return;
      }}
      const frame = frames[index];
      image.src = frameImageUrl(frame);
      overlay.innerHTML = `指令: ${{frame.instruction}}<br />vx: ${{frame.vx.toFixed(3)}} vy: ${{frame.vy.toFixed(3)}} wz: ${{frame.wz.toFixed(3)}}<br />t: ${{frame.timestamp.toFixed(3)}}`;
      slider.value = String(index);
      preloadAround(index);
      updatePlaybackStats();
    }}

    function step(delta) {{
      if (!frames.length) {{
        return;
      }}
      index = (index + delta + frames.length) % frames.length;
      render();
      if (playing) {{
        scheduleNextFrame();
      }}
    }}

    document.getElementById('prevFrame').addEventListener('click', () => step(-1));
    document.getElementById('nextFrame').addEventListener('click', () => step(1));
    qualityLabel.addEventListener('change', saveCurrentReview);
    qualityNotes.addEventListener('change', saveCurrentReview);
    document.getElementById('bindReviewFile').addEventListener('click', () => {{
      bindReviewFile().catch((error) => announceStatus(`绑定文件失败：${{error}}`, true));
    }});
    document.getElementById('exportReview').addEventListener('click', () => {{
      exportReviewFile().catch((error) => announceStatus(`导出失败：${{error}}`, true));
    }});
    document.getElementById('resetReview').addEventListener('click', () => {{
      clearEpisodeOverride(currentEpisode.session_id, currentEpisode.episode_id);
      renderCurrentReview();
      persistReviewState(`已恢复 ${{currentEpisode.session_id}}/${{currentEpisode.episode_id}} 的原始录制分数`)
        .catch((error) => announceStatus(`写回失败：${{error}}`, true));
    }});
    document.getElementById('importReviewFile').addEventListener('change', async (event) => {{
      const [file] = Array.from(event.target.files || []);
      if (!file) {{
        return;
      }}
      try {{
        await importReviewFile(file);
        renderCurrentReview();
        await persistReviewState(`已导入并更新当前页面分数`);
      }} catch (error) {{
        announceStatus(`导入失败：${{error}}`, true);
      }} finally {{
        event.target.value = '';
      }}
    }});
    slider.addEventListener('input', (event) => {{
      index = Number(event.target.value);
      render();
      if (playing) {{
        scheduleNextFrame();
      }}
    }});
    speedSelect.addEventListener('change', (event) => {{
      playbackSpeed = Math.max(0.25, Number(event.target.value) || 1.0);
      updatePlaybackStats();
      if (playing) {{
        scheduleNextFrame();
      }}
    }});
    playPause.addEventListener('click', () => {{
      if (playing) {{
        stopPlayback();
        return;
      }}
      if (!frames.length) {{
        return;
      }}
      playing = true;
      playPause.textContent = 'Pause';
      scheduleNextFrame();
    }});
    window.addEventListener('keydown', (event) => {{
      if (event.key === ' ') {{
        event.preventDefault();
        playPause.click();
      }} else if (event.key === 'ArrowLeft') {{
        event.preventDefault();
        step(-1);
      }} else if (event.key === 'ArrowRight') {{
        event.preventDefault();
        step(1);
      }}
    }});
    renderCurrentReview();
    render();
  </script>
</body>
</html>
"""


def write_reports(episodes: Sequence[EpisodeRecord], output_dir: Path, data_root: Path, seed: int, num_samples: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.html"
    rng = random.Random(seed)
    episode_refs_payload = [
        {
            "session_id": episode.session_id,
            "episode_id": episode.episode_id,
            "original_quality_label": episode.original_quality_label,
        }
        for episode in episodes
    ]
    initial_reviews = _initial_review_payload(episodes)
    page_names = [f"{episode.session_id}_{episode.episode_id}.html" for episode in episodes]

    rows = []
    for row_index, episode in enumerate(episodes):
        page_name = page_names[row_index]
        page_path = episode_dir / page_name
        page_path.write_text(
            _episode_page(
                episode,
                page_path,
                index_path,
                rng,
                num_samples,
                data_root,
                episode_refs_payload,
                initial_reviews,
                page_names[row_index - 1] if row_index > 0 else None,
                page_names[row_index + 1] if row_index + 1 < len(page_names) else None,
            ),
            encoding="utf-8",
        )
        review_row_class = "quality-excluded" if episode.exclude_from_processing else ("quality-review" if episode.quality_overridden else "")
        rows.append(
            f"<tr class=\"{review_row_class}\" data-review-row=\"1\" data-session-id=\"{escape(episode.session_id)}\" data-episode-id=\"{escape(episode.episode_id)}\">"
            f"<td><a href=\"episodes/{escape(page_name)}\">{escape(episode.session_id)} / {escape(episode.episode_id)}</a></td>"
            f"<td>{escape(episode.instruction)}</td><td>{escape(episode.capture_mode or '-')}</td><td>{escape(episode.task_family or '-')}</td><td>{escape(episode.target_label or episode.target_type or '-')}</td><td>{escape(episode.derived_target_side or '-')}</td><td>{escape(episode.derived_target_distance or '-')}</td><td>{len(episode.frames)}</td>"
            f"<td>{episode.trajectory_metrics.get('duration_seconds', 0.0):.2f}</td><td>{int(episode.trajectory_metrics.get('action_change_count', 0))}</td><td>{episode.trajectory_metrics.get('stop_ratio', 0.0):.1%}</td><td>{episode.trajectory_metrics.get('turn_ratio', 0.0):.1%}</td>"
            f"<td>{escape(episode.scene_id or '-')}</td><td>{escape(episode.operator_id or '-')}</td>"
            f"<td>{escape(', '.join(episode.warnings) or '无')}</td>"
            f"<td>{escape(', '.join(episode.info) or '无')}</td>"
            f"<td><span class=\"quality-badge label-{episode.original_quality_label}\">{escape(_quality_badge_text(episode.original_quality_label))}</span></td>"
            f"<td><span class=\"quality-badge label-{episode.quality_label}\" data-review-badge>{escape(_quality_badge_text(episode.quality_label))}</span></td>"
            f"<td>{_quality_select_html(f'quality-select-{row_index}', episode.quality_label, 'data-review-select=\"1\"')}</td>"
            f"<td><input class=\"quality-note\" data-review-note=\"1\" type=\"text\" value=\"{escape(episode.quality_notes)}\" placeholder=\"可选：记录排除原因\" /></td></tr>"
        )

    instruction_episode_counts: Dict[str, int] = {}
    instruction_frame_counts: Dict[str, int] = {}
    scene_counts: Dict[str, int] = {}
    operator_counts: Dict[str, int] = {}
    capture_mode_counts: Dict[str, int] = {}
    task_family_counts: Dict[str, int] = {}
    target_type_counts: Dict[str, int] = {}
    target_label_counts: Dict[str, int] = {}
    target_description_counts: Dict[str, int] = {}
    derived_target_side_counts: Dict[str, int] = {}
    derived_target_distance_counts: Dict[str, int] = {}
    quality_label_counts: Dict[str, int] = {}
    instruction_scene_sets: Dict[str, set] = {}
    instruction_target_sets: Dict[str, set] = {}
    zero_action_frame_count = 0
    frame_lengths: List[int] = []
    duration_values: List[float] = []
    action_change_values: List[float] = []
    stop_ratio_values: List[float] = []
    turn_ratio_values: List[float] = []
    trajectory_episodes: List[EpisodeRecord] = []
    for episode in episodes:
        instruction_episode_counts[episode.instruction] = instruction_episode_counts.get(episode.instruction, 0) + 1
        instruction_frame_counts[episode.instruction] = instruction_frame_counts.get(episode.instruction, 0) + len(episode.frames)
        scene_key = episode.scene_id or "-"
        operator_key = episode.operator_id or "-"
        capture_mode_key = episode.capture_mode or "-"
        scene_counts[scene_key] = scene_counts.get(scene_key, 0) + 1
        operator_counts[operator_key] = operator_counts.get(operator_key, 0) + 1
        capture_mode_counts[capture_mode_key] = capture_mode_counts.get(capture_mode_key, 0) + 1
        task_family_key = episode.task_family or "-"
        target_type_key = episode.target_type or "-"
        target_label_key = episode.target_label or "-"
        target_description_key = episode.target_description or "-"
        derived_target_side_key = episode.derived_target_side or "-"
        derived_target_distance_key = episode.derived_target_distance or "-"
        quality_label_key = _quality_badge_text(episode.quality_label)
        task_family_counts[task_family_key] = task_family_counts.get(task_family_key, 0) + 1
        target_type_counts[target_type_key] = target_type_counts.get(target_type_key, 0) + 1
        target_label_counts[target_label_key] = target_label_counts.get(target_label_key, 0) + 1
        target_description_counts[target_description_key] = target_description_counts.get(target_description_key, 0) + 1
        derived_target_side_counts[derived_target_side_key] = derived_target_side_counts.get(derived_target_side_key, 0) + 1
        derived_target_distance_counts[derived_target_distance_key] = derived_target_distance_counts.get(derived_target_distance_key, 0) + 1
        quality_label_counts[quality_label_key] = quality_label_counts.get(quality_label_key, 0) + 1
        instruction_scene_sets.setdefault(episode.instruction, set()).add(scene_key)
        derived_target_key = (
            f"{episode.derived_target_side or 'unknown'}:{episode.derived_target_distance or 'unknown'}"
            if (episode.derived_target_side or episode.derived_target_distance)
            else "-"
        )
        instruction_target_sets.setdefault(episode.instruction, set()).add(
            episode.target_label or episode.target_type or derived_target_key
        )
        frame_lengths.append(len(episode.frames))
        duration_values.append(float(episode.trajectory_metrics.get("duration_seconds", 0.0)))
        action_change_values.append(float(episode.trajectory_metrics.get("action_change_count", 0.0)))
        stop_ratio_values.append(float(episode.trajectory_metrics.get("stop_ratio", 0.0)))
        turn_ratio_values.append(float(episode.trajectory_metrics.get("turn_ratio", 0.0)))
        if episode.capture_mode == "trajectory":
            trajectory_episodes.append(episode)
        for frame in episode.frames:
            magnitude = sum(abs(frame.control_action[axis]) for axis in ACTION_FIELDS)
            if magnitude <= 1e-6:
                zero_action_frame_count += 1

    empty_session_count, session_episode_counts = collect_session_stats(data_root)
    total_frames = sum(len(episode.frames) for episode in episodes)
    length_min = min(frame_lengths) if frame_lengths else 0
    length_max = max(frame_lengths) if frame_lengths else 0
    length_median = sorted(frame_lengths)[len(frame_lengths) // 2] if frame_lengths else 0

    summary = {
        "episode_count": len(episodes),
        "frame_count": total_frames,
        "warning_episode_count": sum(1 for episode in episodes if episode.warnings),
        "info_episode_count": sum(1 for episode in episodes if episode.info),
        "empty_session_count": empty_session_count,
        "session_episode_counts": session_episode_counts,
        "instructions": sorted({episode.instruction for episode in episodes}),
        "instruction_episode_counts": dict(sorted(instruction_episode_counts.items())),
        "instruction_frame_counts": dict(sorted(instruction_frame_counts.items())),
        "scene_episode_counts": dict(sorted(scene_counts.items())),
        "operator_episode_counts": dict(sorted(operator_counts.items())),
        "capture_mode_episode_counts": dict(sorted(capture_mode_counts.items())),
        "task_family_episode_counts": dict(sorted(task_family_counts.items())),
        "target_type_episode_counts": dict(sorted(target_type_counts.items())),
        "target_label_episode_counts": dict(sorted(target_label_counts.items(), key=lambda item: (-item[1], item[0]))),
        "target_description_episode_counts": dict(sorted(target_description_counts.items(), key=lambda item: (-item[1], item[0]))),
        "derived_target_side_episode_counts": dict(sorted(derived_target_side_counts.items(), key=lambda item: (-item[1], item[0]))),
        "derived_target_distance_episode_counts": dict(sorted(derived_target_distance_counts.items(), key=lambda item: (-item[1], item[0]))),
        "quality_label_episode_counts": dict(sorted(quality_label_counts.items(), key=lambda item: (-item[1], item[0]))),
        "quality_filtered_episode_count": sum(1 for episode in episodes if episode.exclude_from_processing),
        "instructions_with_multiple_scenes": {
            instruction: sorted(scene_set)
            for instruction, scene_set in sorted(instruction_scene_sets.items())
            if len(scene_set - {"-"}) >= 2
        },
        "instructions_with_multiple_targets": {
            instruction: sorted(target_set)
            for instruction, target_set in sorted(instruction_target_sets.items())
            if len(target_set - {"-"}) >= 2
        },
        "zero_action_frame_count": zero_action_frame_count,
        "zero_action_frame_ratio": 0.0 if total_frames == 0 else zero_action_frame_count / total_frames,
        "episode_length_min": length_min,
        "episode_length_median": length_median,
        "episode_length_max": length_max,
        "trajectory_metrics": {
            "duration_seconds": summarize_trajectory_metric_series(duration_values),
            "action_change_count": summarize_trajectory_metric_series(action_change_values),
            "stop_ratio": summarize_trajectory_metric_series(stop_ratio_values),
            "turn_ratio": summarize_trajectory_metric_series(turn_ratio_values),
        },
        "trajectory_episode_summary": {
            "episode_count": len(trajectory_episodes),
            "duration_seconds": summarize_trajectory_metric_series(
                [float(episode.trajectory_metrics.get("duration_seconds", 0.0)) for episode in trajectory_episodes]
            ),
            "action_change_count": summarize_trajectory_metric_series(
                [float(episode.trajectory_metrics.get("action_change_count", 0.0)) for episode in trajectory_episodes]
            ),
            "stop_ratio": summarize_trajectory_metric_series(
                [float(episode.trajectory_metrics.get("stop_ratio", 0.0)) for episode in trajectory_episodes]
            ),
            "turn_ratio": summarize_trajectory_metric_series(
                [float(episode.trajectory_metrics.get("turn_ratio", 0.0)) for episode in trajectory_episodes]
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    instruction_rows = []
    total_episodes = max(1, summary["episode_count"])
    total_frames = max(1, summary["frame_count"])
    for instruction in summary["instructions"]:
        episode_count = summary["instruction_episode_counts"].get(instruction, 0)
        frame_count = summary["instruction_frame_counts"].get(instruction, 0)
        instruction_rows.append(
            f"<tr><td><code>{escape(instruction)}</code></td>"
            f"<td>{episode_count}</td>"
            f"<td>{episode_count / total_episodes:.1%}</td>"
            f"<td>{frame_count}</td>"
            f"<td>{frame_count / total_frames:.1%}</td></tr>"
        )

    scene_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["scene_episode_counts"].items()
    )
    operator_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["operator_episode_counts"].items()
    )
    capture_mode_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["capture_mode_episode_counts"].items()
    )
    task_family_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["task_family_episode_counts"].items()
    )
    target_type_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["target_type_episode_counts"].items()
    )
    target_label_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["target_label_episode_counts"].items()
    )
    derived_target_side_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["derived_target_side_episode_counts"].items()
    )
    derived_target_distance_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["derived_target_distance_episode_counts"].items()
    )
    quality_label_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["quality_label_episode_counts"].items()
    )
    session_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["session_episode_counts"].items()
    )
    multi_scene_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{escape(', '.join(values))}</td></tr>"
        for name, values in summary["instructions_with_multiple_scenes"].items()
    )
    multi_target_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{escape(', '.join(values))}</td></tr>"
        for name, values in summary["instructions_with_multiple_targets"].items()
    )
    review_script = _review_shared_script(data_root, episode_refs_payload, initial_reviews)

    index_html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Go2 数据集体检报告</title>
  <style>{CSS}</style>
</head>
<body>
  <div class=\"card\">
    <h1>Go2 数据集体检报告</h1>
    <p class=\"meta\">episodes={summary['episode_count']} frames={summary['frame_count']} warning_episodes={summary['warning_episode_count']} info_episodes={summary['info_episode_count']}</p>
    <p>指令s: <code>{escape(', '.join(summary['instructions']) or '无')}</code></p>
  </div>
  <div class=\"card\">
    <h2>录制分数回调</h2>
    <p class=\"meta\">这里显示的是录制时的原始分数 `1/2/3`。如果要把某些 `3` 回调成 `2`，可先用“指定分数”筛出，再修改后导出到 <code>{escape(str(data_root / QUALITY_REVIEW_FILENAME))}</code>。</p>
    <div class=\"review-toolbar\">
      <label>筛选
        <select id=\"qualityFilter\" class=\"quality-filter\">
          <option value=\"all\">全部样本</option>
          <option value=\"excluded\">仅质量差（3）</option>
          <option value=\"overridden\">仅已回调</option>
          <option value=\"original\">仅未回调</option>
        </select>
      </label>
      <label>指定分数
        <select id=\"qualityScoreFilter\" class=\"quality-score-filter\">
          <option value=\"all\">全部分数</option>
          <option value=\"1\">仅 1 好样本</option>
          <option value=\"2\">仅 2 可用但不完美</option>
          <option value=\"3\">仅 3 失败但保留 / 质量差</option>
        </select>
      </label>
      <button id=\"bindReviewFile\">绑定回调文件</button>
      <button id=\"exportReview\">导出回调 JSON</button>
      <button id=\"resetAllReviews\">清空本地回调缓存</button>
      <label>导入回调 JSON <input id=\"importReviewFile\" type=\"file\" accept=\".json,application/json\" /></label>
      <span id=\"reviewFileStatus\" class=\"review-file-status\"></span>
    </div>
    <div class=\"review-summary\">
      <span class=\"review-pill\">1 好样本 <strong id=\"summaryLabel1\">0</strong></span>
      <span class=\"review-pill\">2 可用但不完美 <strong id=\"summaryLabel2\">0</strong></span>
      <span class=\"review-pill\">3 失败但保留 / 质量差 <strong id=\"summaryLabel3\">0</strong></span>
      <span class=\"review-pill\">已回调 <strong id=\"summaryOverridden\">0</strong></span>
      <span class=\"review-pill\">后续处理排除 <strong id=\"summaryExcluded\">0</strong></span>
    </div>
    <p id=\"reviewStatus\" class=\"meta\">修改后会立即尝试写回绑定的本地文件；未绑定时先保存在浏览器缓存中。</p>
  </div>
  <div class=\"grid\">
    <div class=\"card\">
      <h2>Dataset Overview</h2>
      <table>
        <tbody>
          <tr><th>Total episodes</th><td>{summary['episode_count']}</td></tr>
          <tr><th>Total frames</th><td>{summary['frame_count']}</td></tr>
          <tr><th>Empty sessions</th><td>{summary['empty_session_count']}</td></tr>
          <tr><th>Zero-action frames</th><td>{summary['zero_action_frame_count']}</td></tr>
          <tr><th>Zero-action ratio</th><td>{summary['zero_action_frame_ratio']:.1%}</td></tr>
          <tr><th>Quality-filtered episodes</th><td>{summary['quality_filtered_episode_count']}</td></tr>
          <tr><th>Episode length min / median / max</th><td>{summary['episode_length_min']} / {summary['episode_length_median']} / {summary['episode_length_max']}</td></tr>
          <tr><th>平均时长（秒）</th><td>{summary['trajectory_metrics']['duration_seconds']['mean']:.2f}</td></tr>
          <tr><th>平均动作切换次数</th><td>{summary['trajectory_metrics']['action_change_count']['mean']:.2f}</td></tr>
          <tr><th>平均停止占比</th><td>{summary['trajectory_metrics']['stop_ratio']['mean']:.1%}</td></tr>
          <tr><th>平均转向占比</th><td>{summary['trajectory_metrics']['turn_ratio']['mean']:.1%}</td></tr>
        </tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>Session 覆盖情况</h2>
      <table>
        <thead><tr><th>Session</th><th>Episode 数</th></tr></thead>
        <tbody>{session_rows or '<tr><td colspan="2">没有可用的 session 元数据。</td></tr>'}</tbody>
      </table>
    </div>
  </div>
  <div class=\"card\">
    <h2>指令统计</h2>
    <table>
      <thead><tr><th>指令</th><th>Episode 数</th><th>Episode 占比</th><th>帧数</th><th>帧占比</th></tr></thead>
      <tbody>{''.join(instruction_rows) or '<tr><td colspan="5">没有可用的指令统计。</td></tr>'}</tbody>
    </table>
  </div>
  <div class=\"grid\">
    <div class=\"card\">
      <h2>场景覆盖情况</h2>
      <table>
        <thead><tr><th>场景</th><th>Episode 数</th></tr></thead>
        <tbody>{scene_rows or '<tr><td colspan="2">没有可用的场景元数据。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>操作员覆盖情况</h2>
      <table>
        <thead><tr><th>操作员</th><th>Episode 数</th></tr></thead>
        <tbody>{operator_rows or '<tr><td colspan="2">没有可用的操作员元数据。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>采集模式统计</h2>
      <table>
        <thead><tr><th>采集模式</th><th>Episode 数</th></tr></thead>
        <tbody>{capture_mode_rows or '<tr><td colspan="2">没有可用的采集模式元数据。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>任务族统计</h2>
      <table>
        <thead><tr><th>任务族</th><th>Episode 数</th></tr></thead>
        <tbody>{task_family_rows or '<tr><td colspan="2">没有可用的任务族元数据。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>目标类型统计</h2>
      <table>
        <thead><tr><th>目标类型</th><th>Episode 数</th></tr></thead>
        <tbody>{target_type_rows or '<tr><td colspan="2">没有可用的目标元数据。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>语义目标统计</h2>
      <table>
        <thead><tr><th>target_label</th><th>Episode 数</th></tr></thead>
        <tbody>{target_label_rows or '<tr><td colspan="2">没有可用的语义目标标签。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>派生左右标签统计</h2>
      <table>
        <thead><tr><th>target_side_band</th><th>Episode 数</th></tr></thead>
        <tbody>{derived_target_side_rows or '<tr><td colspan="2">还没有派生左右标签。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>派生远近标签统计</h2>
      <table>
        <thead><tr><th>target_distance_band</th><th>Episode 数</th></tr></thead>
        <tbody>{derived_target_distance_rows or '<tr><td colspan="2">还没有派生远近标签。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>分数统计</h2>
      <table>
        <thead><tr><th>当前分数</th><th>Episode 数</th></tr></thead>
        <tbody>{quality_label_rows or '<tr><td colspan="2">当前还没有分数统计。</td></tr>'}</tbody>
      </table>
    </div>
  </div>
  <div class=\"grid\">
    <div class=\"card\">
      <h2>轨迹统计指标</h2>
      <table>
        <thead><tr><th>指标</th><th>均值</th><th>中位数</th><th>最小值</th><th>最大值</th></tr></thead>
        <tbody>
          <tr><td>时长（秒）</td><td>{summary['trajectory_metrics']['duration_seconds']['mean']:.2f}</td><td>{summary['trajectory_metrics']['duration_seconds']['median']:.2f}</td><td>{summary['trajectory_metrics']['duration_seconds']['min']:.2f}</td><td>{summary['trajectory_metrics']['duration_seconds']['max']:.2f}</td></tr>
          <tr><td>动作切换次数</td><td>{summary['trajectory_metrics']['action_change_count']['mean']:.2f}</td><td>{summary['trajectory_metrics']['action_change_count']['median']:.2f}</td><td>{summary['trajectory_metrics']['action_change_count']['min']:.0f}</td><td>{summary['trajectory_metrics']['action_change_count']['max']:.0f}</td></tr>
          <tr><td>停止占比</td><td>{summary['trajectory_metrics']['stop_ratio']['mean']:.1%}</td><td>{summary['trajectory_metrics']['stop_ratio']['median']:.1%}</td><td>{summary['trajectory_metrics']['stop_ratio']['min']:.1%}</td><td>{summary['trajectory_metrics']['stop_ratio']['max']:.1%}</td></tr>
          <tr><td>转向占比</td><td>{summary['trajectory_metrics']['turn_ratio']['mean']:.1%}</td><td>{summary['trajectory_metrics']['turn_ratio']['median']:.1%}</td><td>{summary['trajectory_metrics']['turn_ratio']['min']:.1%}</td><td>{summary['trajectory_metrics']['turn_ratio']['max']:.1%}</td></tr>
        </tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>仅 trajectory 模式</h2>
      <table>
        <tbody>
          <tr><th>trajectory episode 数</th><td>{summary['trajectory_episode_summary']['episode_count']}</td></tr>
          <tr><th>平均时长（秒）</th><td>{summary['trajectory_episode_summary']['duration_seconds']['mean']:.2f}</td></tr>
          <tr><th>平均动作切换次数</th><td>{summary['trajectory_episode_summary']['action_change_count']['mean']:.2f}</td></tr>
          <tr><th>平均停止占比</th><td>{summary['trajectory_episode_summary']['stop_ratio']['mean']:.1%}</td></tr>
          <tr><th>平均转向占比</th><td>{summary['trajectory_episode_summary']['turn_ratio']['mean']:.1%}</td></tr>
        </tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>跨场景重复指令</h2>
      <table>
        <thead><tr><th>指令</th><th>场景s</th></tr></thead>
        <tbody>{multi_scene_rows or '<tr><td colspan="2">当前还没有跨多个场景重复出现的指令。</td></tr>'}</tbody>
      </table>
    </div>
    <div class=\"card\">
      <h2>跨目标重复指令</h2>
      <table>
        <thead><tr><th>指令</th><th>目标s</th></tr></thead>
        <tbody>{multi_target_rows or '<tr><td colspan="2">当前还没有跨多个目标重复出现的指令。</td></tr>'}</tbody>
      </table>
    </div>
  </div>
  <div class=\"card\">
    <table>
      <thead><tr><th>Episode</th><th>指令</th><th>采集模式</th><th>任务族</th><th>目标</th><th>左右</th><th>远近</th><th>帧数</th><th>Duration(s)</th><th>Action Changes</th><th>Stop Ratio</th><th>Turn Ratio</th><th>场景</th><th>操作员</th><th>警告</th><th>信息</th><th>原始分数</th><th>当前分数</th><th>设置分数</th><th>备注</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
  <script>
    {review_script}
    const qualityFilter = document.getElementById('qualityFilter');
    const qualityScoreFilter = document.getElementById('qualityScoreFilter');
    const reviewRows = Array.from(document.querySelectorAll('[data-review-row]'));

    function renderRow(row) {{
      const sessionId = row.dataset.sessionId || '';
      const episodeId = row.dataset.episodeId || '';
      const review = getEpisodeReview(sessionId, episodeId);
      const badge = row.querySelector('[data-review-badge]');
      const select = row.querySelector('[data-review-select]');
      const note = row.querySelector('[data-review-note]');
      if (badge) {{
        badge.textContent = labelText(review.quality_label);
        badge.className = `quality-badge ${{labelClass(review.quality_label)}}`;
      }}
      if (select) {{
        select.value = String(review.quality_label);
      }}
      if (note) {{
        note.value = review.quality_notes;
      }}
      row.classList.toggle('quality-excluded', review.exclude_from_processing);
      row.classList.toggle('quality-review', !review.exclude_from_processing && review.quality_overridden);
    }}

    function updateReviewSummary() {{
      const counts = {{ 1: 0, 2: 0, 3: 0 }};
      let excluded = 0;
      let overridden = 0;
      episodeRefs.forEach((item) => {{
        const review = getEpisodeReview(item.session_id, item.episode_id);
        counts[review.quality_label] = (counts[review.quality_label] || 0) + 1;
        if (review.exclude_from_processing) {{
          excluded += 1;
        }}
        if (review.quality_overridden) {{
          overridden += 1;
        }}
      }});
      document.getElementById('summaryLabel1').textContent = String(counts[1] || 0);
      document.getElementById('summaryLabel2').textContent = String(counts[2] || 0);
      document.getElementById('summaryLabel3').textContent = String(counts[3] || 0);
      document.getElementById('summaryOverridden').textContent = String(overridden);
      document.getElementById('summaryExcluded').textContent = String(excluded);
    }}

    function applyRowFilter() {{
      const mode = qualityFilter.value;
      const scoreMode = qualityScoreFilter.value;
      reviewRows.forEach((row) => {{
        const sessionId = row.dataset.sessionId || '';
        const episodeId = row.dataset.episodeId || '';
        const review = getEpisodeReview(sessionId, episodeId);
        let visible = true;
        if (mode === 'excluded') {{
          visible = review.exclude_from_processing;
        }} else if (mode === 'overridden') {{
          visible = review.quality_overridden || Boolean(review.quality_notes);
        }} else if (mode === 'original') {{
          visible = !review.quality_overridden && !review.quality_notes;
        }}
        if (visible && scoreMode !== 'all') {{
          visible = String(review.quality_label) === scoreMode;
        }}
        row.hidden = !visible;
      }});
    }}

    function saveRowReview(row) {{
      const sessionId = row.dataset.sessionId || '';
      const episodeId = row.dataset.episodeId || '';
      const select = row.querySelector('[data-review-select]');
      const note = row.querySelector('[data-review-note]');
      const review = upsertEpisodeReview(sessionId, episodeId, Number(select && select.value) || 0, note ? note.value : '');
      renderRow(row);
      updateReviewSummary();
      applyRowFilter();
      persistReviewState(`已更新 ${{sessionId}}/${{episodeId}} -> ${{labelText(review.quality_label)}}`)
        .catch((error) => announceStatus(`写回失败：${{error}}`, true));
    }}

    reviewRows.forEach((row) => {{
      renderRow(row);
      const select = row.querySelector('[data-review-select]');
      const note = row.querySelector('[data-review-note]');
      if (select) {{
        select.addEventListener('change', () => saveRowReview(row));
      }}
      if (note) {{
        note.addEventListener('change', () => saveRowReview(row));
      }}
    }});

    qualityFilter.addEventListener('change', applyRowFilter);
    qualityScoreFilter.addEventListener('change', applyRowFilter);
    document.getElementById('bindReviewFile').addEventListener('click', () => {{
      bindReviewFile().catch((error) => announceStatus(`绑定文件失败：${{error}}`, true));
    }});
    document.getElementById('exportReview').addEventListener('click', () => {{
      exportReviewFile().catch((error) => announceStatus(`导出失败：${{error}}`, true));
    }});
    document.getElementById('resetAllReviews').addEventListener('click', () => {{
      clearReviewCache();
      reviewRows.forEach(renderRow);
      updateReviewSummary();
      applyRowFilter();
      persistReviewState('已清空当前数据集的回调缓存')
        .catch((error) => announceStatus(`写回失败：${{error}}`, true));
    }});
    document.getElementById('importReviewFile').addEventListener('change', async (event) => {{
      const [file] = Array.from(event.target.files || []);
      if (!file) {{
        return;
      }}
      try {{
        await importReviewFile(file);
        reviewRows.forEach(renderRow);
        updateReviewSummary();
        applyRowFilter();
        await persistReviewState('已导入并写回回调结果');
      }} catch (error) {{
        announceStatus(`导入失败：${{error}}`, true);
      }} finally {{
        event.target.value = '';
      }}
    }});

    updateReviewSummary();
    applyRowFilter();
  </script>
</body>
</html>
"""
    index_path.write_text(index_html, encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    data_root = args.data_root.resolve()
    quality_review_index = load_quality_review_index(data_root)
    output_dir = args.output_dir.resolve()
    episodes = load_episodes(data_root, args.max_episodes, quality_review_index)
    write_reports(episodes, output_dir, data_root, args.seed, args.num_samples)
    print(f"已为 {len(episodes)} 条 episode 生成体检报告，输出目录：{output_dir}")
    if args.serve:
        serve_report(output_dir, data_root, args.host, args.port)


if __name__ == "__main__":
    main()

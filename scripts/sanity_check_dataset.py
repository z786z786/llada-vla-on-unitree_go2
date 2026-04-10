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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from llada_vla_common import (
    ACTION_FIELDS,
    CONTROLLED_INSTRUCTIONS,
    VISUAL_TASK_FAMILIES,
    control_action_from_frame,
    discover_session_roots,
    episode_task_metadata,
    infer_task_family,
    load_episode_derived_labels,
    load_json,
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
    target_description: str
    derived_target_side: str
    derived_target_distance: str
    derived_label_source: str
    collector_notes: str
    scene_id: str
    operator_id: str
    frames: List[FrameRecord]
    warnings: List[str]
    info: List[str]
    trajectory_metrics: Dict[str, Any]


CSS = """
body { font-family: sans-serif; margin: 24px; background: #f4f1e8; color: #1c1c1c; }
a { color: #005b7f; }
.card { background: #fffaf0; border: 1px solid #d7ccb8; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
.viewer { position: relative; width: min(960px, 100%); }
.viewer img { width: 100%; border-radius: 12px; display: block; background: #111; }
.overlay { position: absolute; left: 16px; top: 16px; background: rgba(0,0,0,0.65); color: white; padding: 10px 12px; border-radius: 10px; }
.controls { display: flex; gap: 8px; margin: 12px 0; flex-wrap: wrap; }
button, input[type=range] { font: inherit; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.grid img { width: 100%; border-radius: 8px; }
.meta { color: #5f5f5f; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid #e2d7c5; padding: 8px; text-align: left; }
code { background: #efe7d8; padding: 2px 4px; border-radius: 4px; }
svg { width: 100%; height: auto; }
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为 Go2 采集 session 生成本地 HTML 回放与体检报告")
    parser.add_argument("--data-root", type=Path, required=True, help="数据集根目录，或包含多个 session 根目录的父目录")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sanity_check", help="HTML 报告输出目录")
    parser.add_argument("--num-samples", type=int, default=4, help="每条 episode 页面中展示的随机静态帧数量")
    parser.add_argument("--max-episodes", type=int, help="可选的 episode 数量上限，用于冒烟测试")
    parser.add_argument("--seed", type=int, default=0, help="预览采样随机种子")
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


def load_episodes(data_root: Path, max_episodes: Optional[int]) -> List[EpisodeRecord]:
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
                target_description=str(task_metadata.get("target_description") or ""),
                derived_target_side=str(derived_labels.get("target_side_band") or ""),
                derived_target_distance=str(derived_labels.get("target_distance_band") or ""),
                derived_label_source=str(derived_labels.get("label_source") or ""),
                collector_notes=str(task_metadata.get("collector_notes") or ""),
                scene_id=str(payload.get("scene_id") or episode_meta.get("scene_id") or ""),
                operator_id=str(payload.get("operator_id") or episode_meta.get("operator_id") or ""),
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
    height = 220
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
    width = 220
    height = 140
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


def _episode_page(episode: EpisodeRecord, page_path: Path, index_path: Path, rng: random.Random, num_samples: int) -> str:
    frames_payload = []
    for frame in episode.frames:
        frames_payload.append(
            {
                "image": os.path.relpath(frame.image_disk_path, page_path.parent),
                "timestamp": round(frame.timestamp, 6),
                "instruction": frame.instruction,
                "vx": frame.control_action["vx"],
                "vy": frame.control_action["vy"],
                "wz": frame.control_action["wz"],
            }
        )

    sample_frames = rng.sample(episode.frames, min(num_samples, len(episode.frames))) if episode.frames else []
    sample_html = "".join(
        f'<div><img src="{escape(os.path.relpath(frame.image_disk_path, page_path.parent))}" alt="sample frame" />'
        f'<div class="meta">t={frame.timestamp:.3f} vx={frame.control_action["vx"]:.3f} vy={frame.control_action["vy"]:.3f} wz={frame.control_action["wz"]:.3f}</div></div>'
        for frame in sample_frames
    )
    histogram_html = "".join(_histogram_svg([frame.control_action[axis] for frame in episode.frames], axis) for axis in ACTION_FIELDS)
    warnings = "无" if not episode.warnings else ", ".join(episode.warnings)
    info = "无" if not episode.info else ", ".join(episode.info)
    index_rel = os.path.relpath(index_path, page_path.parent)

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{escape(episode.session_id)} {escape(episode.episode_id)}</title>
  <style>{CSS}</style>
</head>
<body>
  <p><a href=\"{escape(index_rel)}\">Back to summary</a></p>
  <div class=\"card\">
    <h1>{escape(episode.session_id)} / {escape(episode.episode_id)}</h1>
    <p class=\"meta\">instruction=<code>{escape(episode.instruction)}</code> capture_mode=<code>{escape(episode.capture_mode or '-')}</code> task_family=<code>{escape(episode.task_family or '-')}</code> target_type=<code>{escape(episode.target_type or '-')}</code> target=<code>{escape(episode.target_description or '-')}</code> derived_side=<code>{escape(episode.derived_target_side or '-')}</code> derived_distance=<code>{escape(episode.derived_target_distance or '-')}</code> derived_source=<code>{escape(episode.derived_label_source or '-')}</code> scene=<code>{escape(episode.scene_id or '-')}</code> operator=<code>{escape(episode.operator_id or '-')}</code> warnings=<code>{escape(warnings)}</code> info=<code>{escape(info)}</code></p>
    <p class=\"meta\">duration_s=<code>{episode.trajectory_metrics.get('duration_seconds', 0.0):.3f}</code> action_changes=<code>{episode.trajectory_metrics.get('action_change_count', 0)}</code> stop_ratio=<code>{episode.trajectory_metrics.get('stop_ratio', 0.0):.1%}</code> turn_ratio=<code>{episode.trajectory_metrics.get('turn_ratio', 0.0):.1%}</code> buckets=<code>{escape(', '.join(episode.trajectory_metrics.get('action_buckets') or []) or '-')}</code></p>
  </div>
  <div class=\"card\">
    <div class=\"viewer\">
      <img id=\"frameImage\" src=\"\" alt=\"episode replay frame\" />
      <div class=\"overlay\" id=\"overlay\"></div>
    </div>
    <div class=\"controls\">
      <button id=\"playPause\">Play</button>
      <button id=\"prevFrame\">Prev</button>
      <button id=\"nextFrame\">Next</button>
      <label>Frame <input id=\"frameSlider\" type=\"range\" min=\"0\" max=\"{max(0, len(frames_payload) - 1)}\" value=\"0\" /></label>
    </div>
  </div>
  <div class=\"card\">
    <h2>Action vs time</h2>
    {_plot_svg(episode)}
  </div>
  <div class=\"card\">
    <h2>Action histograms</h2>
    <div class=\"grid\">{histogram_html}</div>
  </div>
  <div class=\"card\">
    <h2>Random sample inspect</h2>
    <div class=\"grid\">{sample_html or '<p>No frames available.</p>'}</div>
  </div>
  <script>
    const frames = {json.dumps(frames_payload, ensure_ascii=True)};
    let index = 0;
    let timer = null;
    const image = document.getElementById('frameImage');
    const overlay = document.getElementById('overlay');
    const slider = document.getElementById('frameSlider');
    const playPause = document.getElementById('playPause');

    function render() {{
      if (!frames.length) {{
        overlay.textContent = 'No frames available';
        return;
      }}
      const frame = frames[index];
      image.src = frame.image;
      overlay.innerHTML = `指令: ${{frame.instruction}}<br />vx: ${{frame.vx.toFixed(3)}} vy: ${{frame.vy.toFixed(3)}} wz: ${{frame.wz.toFixed(3)}}<br />t: ${{frame.timestamp.toFixed(3)}}`;
      slider.value = String(index);
    }}

    function step(delta) {{
      if (!frames.length) {{
        return;
      }}
      index = (index + delta + frames.length) % frames.length;
      render();
    }}

    document.getElementById('prevFrame').addEventListener('click', () => step(-1));
    document.getElementById('nextFrame').addEventListener('click', () => step(1));
    slider.addEventListener('input', (event) => {{
      index = Number(event.target.value);
      render();
    }});
    playPause.addEventListener('click', () => {{
      if (timer !== null) {{
        clearInterval(timer);
        timer = null;
        playPause.textContent = 'Play';
        return;
      }}
      timer = window.setInterval(() => step(1), 100);
      playPause.textContent = 'Pause';
    }});
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

    rows = []
    for episode in episodes:
        page_name = f"{episode.session_id}_{episode.episode_id}.html"
        page_path = episode_dir / page_name
        page_path.write_text(_episode_page(episode, page_path, index_path, rng, num_samples), encoding="utf-8")
        rows.append(
            f"<tr><td><a href=\"episodes/{escape(page_name)}\">{escape(episode.session_id)} / {escape(episode.episode_id)}</a></td>"
            f"<td>{escape(episode.instruction)}</td><td>{escape(episode.capture_mode or '-')}</td><td>{escape(episode.task_family or '-')}</td><td>{escape(episode.target_description or episode.target_type or '-')}</td><td>{escape(episode.derived_target_side or '-')}</td><td>{escape(episode.derived_target_distance or '-')}</td><td>{len(episode.frames)}</td>"
            f"<td>{episode.trajectory_metrics.get('duration_seconds', 0.0):.2f}</td><td>{int(episode.trajectory_metrics.get('action_change_count', 0))}</td><td>{episode.trajectory_metrics.get('stop_ratio', 0.0):.1%}</td><td>{episode.trajectory_metrics.get('turn_ratio', 0.0):.1%}</td>"
            f"<td>{escape(episode.scene_id or '-')}</td><td>{escape(episode.operator_id or '-')}</td>"
            f"<td>{escape(', '.join(episode.warnings) or '无')}</td>"
            f"<td>{escape(', '.join(episode.info) or '无')}</td></tr>"
        )

    instruction_episode_counts: Dict[str, int] = {}
    instruction_frame_counts: Dict[str, int] = {}
    scene_counts: Dict[str, int] = {}
    operator_counts: Dict[str, int] = {}
    capture_mode_counts: Dict[str, int] = {}
    task_family_counts: Dict[str, int] = {}
    target_type_counts: Dict[str, int] = {}
    target_description_counts: Dict[str, int] = {}
    derived_target_side_counts: Dict[str, int] = {}
    derived_target_distance_counts: Dict[str, int] = {}
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
        target_description_key = episode.target_description or "-"
        derived_target_side_key = episode.derived_target_side or "-"
        derived_target_distance_key = episode.derived_target_distance or "-"
        task_family_counts[task_family_key] = task_family_counts.get(task_family_key, 0) + 1
        target_type_counts[target_type_key] = target_type_counts.get(target_type_key, 0) + 1
        target_description_counts[target_description_key] = target_description_counts.get(target_description_key, 0) + 1
        derived_target_side_counts[derived_target_side_key] = derived_target_side_counts.get(derived_target_side_key, 0) + 1
        derived_target_distance_counts[derived_target_distance_key] = derived_target_distance_counts.get(derived_target_distance_key, 0) + 1
        instruction_scene_sets.setdefault(episode.instruction, set()).add(scene_key)
        derived_target_key = (
            f"{episode.derived_target_side or 'unknown'}:{episode.derived_target_distance or 'unknown'}"
            if (episode.derived_target_side or episode.derived_target_distance)
            else "-"
        )
        instruction_target_sets.setdefault(episode.instruction, set()).add(
            episode.target_description or episode.target_type or derived_target_key
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
        "target_description_episode_counts": dict(sorted(target_description_counts.items(), key=lambda item: (-item[1], item[0]))),
        "derived_target_side_episode_counts": dict(sorted(derived_target_side_counts.items(), key=lambda item: (-item[1], item[0]))),
        "derived_target_distance_episode_counts": dict(sorted(derived_target_distance_counts.items(), key=lambda item: (-item[1], item[0]))),
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
    derived_target_side_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["derived_target_side_episode_counts"].items()
    )
    derived_target_distance_rows = "".join(
        f"<tr><td><code>{escape(name)}</code></td><td>{count}</td></tr>"
        for name, count in summary["derived_target_distance_episode_counts"].items()
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
      <thead><tr><th>Episode</th><th>指令</th><th>采集模式</th><th>任务族</th><th>目标</th><th>左右</th><th>远近</th><th>帧数</th><th>Duration(s)</th><th>Action Changes</th><th>Stop Ratio</th><th>Turn Ratio</th><th>场景</th><th>操作员</th><th>警告</th><th>信息</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</body>
</html>
"""
    index_path.write_text(index_html, encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    data_root = args.data_root.resolve()
    episodes = load_episodes(data_root, args.max_episodes)
    write_reports(episodes, args.output_dir.resolve(), data_root, args.seed, args.num_samples)
    print(f"已为 {len(episodes)} 条 episode 生成体检报告，输出目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

# Sanity Check Report

Use `scripts/sanity_check_dataset.py` to produce a local HTML report for collected sessions.

## Command

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 scripts/sanity_check_dataset.py --data-root data --output-dir outputs/sanity_check
```

Optional flags:

- `--max-episodes N`: quick smoke test on the first `N` episodes
- `--num-samples N`: number of random still frames per episode page
- `--seed N`: random seed for sample selection

## Output

The report directory contains:

- `index.html`: dataset summary and episode table
- `episodes/*.html`: one replay page per episode
- `summary.json`: machine-readable summary

## What The Replay Shows

Each episode page includes:

- image playback like a video replay
- overlay of `instruction`
- overlay of `vx`, `vy`, `wz`
- action-vs-time plot
- action histograms
- random still-frame inspection

This is the primary human check for whether the collected data looks like a coherent control episode.

## What To Look For

Replay:

- wrong frames, blank frames, or image lag
- wrong instruction for the visual scene
- actions that clearly disagree with the image sequence

Action plot:

- long flat zeros
- obvious jumps or clipping
- unexpected oscillation

Histograms:

- extreme dataset bias
- missing turns or lateral movement
- values outside the expected range

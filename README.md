# Go2 Local Data Pipeline

This workspace now has one supported path:

- collect with `native/build/go2_collector`
- collect first-stage motion-only data with `native/build/go2_collector_stage1`
- inspect with `scripts/sanity_check_dataset.py`
- validate with `tools/validate_bc_dataset.py`
- convert with `tools/convert_llada_vla_dataset.py`
- benchmark with `tools/llada_vla_baseline.py`
- benchmark with `tools/llada_vla_baseline_v2.py`

The goal is to validate the task boundary for a future LLaDA-VLA deployment on Go2 while keeping the current baseline simple.

The project now has two dataset regimes:

- stage 1: canonical motion mapping such as `go forward -> forward`
- stage 2: visually necessary tasks where `instruction + current image` must jointly determine `[vx, vy, wz]`

Stage 2 is now the priority. The baseline remains a diagnostic tool.

## Task Definition

Current task interface:

- input: `instruction + front RGB + optional state`
- target: `control_action(vx, vy, wz)`

The collector stores two action views per frame:

- `raw_action`: the raw controller signal recorded at collection time
- `control_action`: the training and deployment target

In this version, `control_action == raw_action` on `vx/vy/wz`. The split is kept so later deployment-oriented remapping can happen without changing the rest of the pipeline.

## Session Schema

Each general collector run creates a new session under `go2_vla_collector/data/`.
The dedicated stage-1 collector writes to `go2_vla_collector/data_stage1/` by default.

Example session layout:

```text
data/<session_id>/
  index.json
  episodes/
    ep_000001.json
  images/
    ep_000001/
      20260405_144137_001.jpg
```

Each episode contains:

- `schema_version`
- `instruction`, `scene_id`, `operator_id`
- optional task metadata such as `task_family`, `target_type`, `target_description`, `target_instance_id`, `task_tags`, `collector_notes`, `instruction_source`
- `frames[]`
- per-frame `state`, `raw_action`, `control_action`, `image`, and timestamps

## Build

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
cmake -S native -B native/build
cmake --build native/build -j
```

## Collect

Use the native collector directly on the local machine connected to the robot.

For first-stage motion-only collection, use the dedicated stage-1 program:

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
./native/build/go2_collector_stage1 --network-interface eno1 --scene-id corridor_a --operator-id op_01
```

Stage-1 collector notes:

- fixed to `single_action`
- rejects `--instruction`, `--task-family`, and other stage-2 semantic metadata flags
- auto-labels each episode from the locked motion key
- keeps first-stage data separate in `data_stage1/`

The general collector still supports legacy motion-only collection and stage-2 semantic tasks:

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
./native/build/go2_collector --network-interface eno1 --instruction "go forward" --scene-id corridor_a --operator-id op_01
```

For visually necessary data, pass semantic task metadata explicitly for the current collection run:

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
./native/build/go2_collector \
  --network-interface eno1 \
  --scene-id corridor_a \
  --operator-id op_01 \
  --instruction "go to the door" \
  --task-family goal_navigation \
  --target-type door \
  --target-description "glass door at corridor end" \
  --target-instance-id corridor_a_door_01 \
  --task-tags indoor,bright_light
```

Additional examples:

```bash
./native/build/go2_collector \
  --network-interface eno1 \
  --scene-id hall_b \
  --operator-id op_01 \
  --instruction "follow the person in front of you" \
  --task-family visual_following \
  --target-type person \
  --target-description "adult with black jacket"
```

```bash
./native/build/go2_collector \
  --network-interface eno1 \
  --scene-id lab_boxes \
  --operator-id op_01 \
  --instruction "go around the obstacle and continue forward" \
  --task-family obstacle_aware_navigation \
  --target-type obstacle \
  --target-description "cardboard box blocking center path"
```

Collector notes:

- one collector run should use one stable semantic task configuration
- if `--instruction` is omitted, the collector falls back to the legacy motion label
- `scene_id` and `operator_id` remain required
- `task_tags` is optional CSV metadata for later filtering
- keyboard control still records high-level `control_action = [vx, vy, wz]`

Capture modes:

- `single_action`:
  - current legacy behavior
  - press `R` to arm
  - first single motion key locks the motion label
  - logging starts after 0.5s
  - mixed input or direction changes discard the segment
  - key release auto-ends the segment
- `trajectory`:
  - new long-trajectory mode for visually necessary tasks
  - press `R` to start recording immediately
  - direction changes, turning, strafing, and pauses are allowed
  - press `T` to end and save
  - press `ESC` to discard
  - `--instruction` is required because the episode label is trajectory-level semantic text

Example for long-trajectory collection:

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
./native/build/go2_collector \
  --network-interface eno1 \
  --capture-mode trajectory \
  --scene-id corridor_a \
  --operator-id op_01 \
  --instruction "go to the door" \
  --task-family goal_navigation \
  --target-type door \
  --target-description "glass door at corridor end"
```

Keyboard shortcuts:

- `W/S`: forward or backward
- `A/D`: strafe left or right
- `Q/E`: turn left or right
- `R`: start the selected capture flow
- `T`: stop the current capture segment and save
- `ESC`: cancel the current armed or active segment
- `Space`: emergency stop
- `C`: clear fault or toggle stand up/down
- `P`: print status
- `H`: print help
- `X`: quit

## Next-Stage Task Families

The next dataset should focus on the following visually necessary task families:

1. `goal_navigation`
   - examples: `go to the door`, `go to the apple`, `approach the box`
   - image is necessary because the same instruction maps to different steering actions depending on where the target appears
2. `visual_following`
   - examples: `follow the person`, `follow the person in front of you`
   - image is necessary because the target position changes frame to frame
3. `obstacle_aware_navigation`
   - examples: `avoid the object in front of you`, `go around the obstacle and continue forward`
   - image is necessary because obstacle geometry and free space determine the action

See `docs/visually_necessary_collection_plan.md` for the concrete collection plan.

## Inspect And Validate

Generate an HTML replay report with action overlays, action curves, histograms, and random sample views:

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 scripts/sanity_check_dataset.py --data-root data --output-dir outputs/sanity_check
```

Validate whether sessions are trainable:

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/validate_bc_dataset.py --dataset-root data --report-path outputs/validate_report.json
```

The validator now also reports whether image could matter in practice:

- `task_family_counts`
- `target_type_counts`
- `target_description_counts`
- instructions spanning multiple scenes
- instructions spanning multiple targets
- count of visually necessary episodes

The sanity-check HTML now surfaces task metadata per episode so it is easier to catch dataset regimes that are still instruction-only in disguise.

## Convert And Train

Convert raw sessions into train/val/test manifests:

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/convert_llada_vla_dataset.py --raw-root data --output-root data/llada_vla_converted --overwrite
```

To reproduce the old "min10" style filtered conversion in the same script, drop episodes shorter than 10 frames. Empty sessions are skipped automatically:

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/convert_llada_vla_dataset.py --raw-root data --output-root data_min10_converted --min-episode-length 10 --overwrite
```

When raw sessions are split by session, the converter now orders sessions by episode count descending before assigning train/val/test, using `session_id` only as a stable tie-breaker.

The converted manifest preserves the next-stage fields when present:

- `task_family`
- `target_type`
- `target_description`
- `target_instance_id`
- `task_tags`
- `collector_notes`
- `instruction_source`

Run the lightweight baseline on either converted manifests or raw sessions:

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/llada_vla_baseline.py --dataset-root data/llada_vla_converted --save-model data/llada_vla_converted/baseline_model.json
```

The baseline is intentionally simple. Its job is to validate that the current data definition is learnable and already matches the future deployment interface.

## PyTorch Baseline V2

The new v2 baseline is a lightweight multimodal PyTorch model:

- text: small trainable vocabulary + embedding + mean pooling
- vision: pretrained `torchvision` ResNet18 by default
- state: optional, leakage-safe by default with `yaw` only
- fusion: `mlp` by default, `tiny_transformer` optional
- head: MSE regression to `control_action(vx, vy, wz)`

Use this baseline to answer whether the current dataset is learnable and whether image adds value. Do not treat it as the main research target.

Install the PyTorch dependencies before training v2:

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 -m pip install -r requirements.txt
```

Train each ablation mode on converted manifests:

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/llada_vla_baseline_v2.py \
  --dataset-root data/llada_vla_converted \
  --output-dir outputs/baseline_v2_instruction_only \
  --ablation-mode instruction_only
```

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/llada_vla_baseline_v2.py \
  --dataset-root data/llada_vla_converted \
  --output-dir outputs/baseline_v2_image_instruction \
  --ablation-mode image_plus_instruction
```

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/llada_vla_baseline_v2.py \
  --dataset-root data/llada_vla_converted \
  --output-dir outputs/baseline_v2_image_instruction_state \
  --ablation-mode image_plus_instruction_plus_state \
  --use-state true
```

Evaluate a saved checkpoint:

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/llada_vla_baseline_v2.py \
  --dataset-root data/llada_vla_converted \
  --output-dir outputs/baseline_v2_eval \
  --checkpoint-path outputs/baseline_v2_image_instruction/best.pt \
  --eval-only
```

Useful options:

- `--fusion-type mlp|tiny_transformer`
- `--split-strategy use_manifest|auto|by_session|by_episode`
- `--freeze-vision-backbone`
- `--vision-learning-rate 1e-5`
- `--save-predictions`
- `--compare-old-linear`
- `--no-pretrained-vision`

The v2 output directory contains:

- `best.pt` and `last.pt`
- `metrics.json`
- `training_summary.json`
- `summary.txt`
- optional `predictions_val.jsonl` and `predictions_test.jsonl`

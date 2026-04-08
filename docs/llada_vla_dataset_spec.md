# Go2 Collector Dataset Spec

## Purpose

This spec defines the local session schema and the converted manifest schema used by the current Go2 data pipeline.

The design goal is:

- keep the model simple now
- keep the task interface aligned with future LLaDA-VLA deployment

The next data regime is no longer limited to canonical motion labels. The target is visually necessary language-conditioned locomotion:

- input: `instruction + front RGB + optional state`
- target: `control_action(vx, vy, wz)`
- key property: the same instruction must require different actions in different visual scenes

## Next-Stage Task Families

### 1. `goal_navigation`

- instruction format:
  - `go to the door`
  - `go to the apple`
  - `go to the red object`
  - `approach the box`
- observation dependence:
  - the robot must localize the target in the current image
- action output:
  - high-level `control_action = [vx, vy, wz]`
- why image is necessary:
  - the same instruction implies turning left, turning right, or moving forward depending on target position and distance

### 2. `visual_following`

- instruction format:
  - `follow the person`
  - `follow the person in front of you`
- observation dependence:
  - the robot must track the target as it moves through the scene
- action output:
  - high-level `control_action = [vx, vy, wz]`
- why image is necessary:
  - the target location changes over time even when the instruction stays fixed

### 3. `obstacle_aware_navigation`

- instruction format:
  - `go around the obstacle and continue forward`
  - `avoid the object in front of you`
- observation dependence:
  - the robot must infer free space and obstacle geometry from the image
- action output:
  - high-level `control_action = [vx, vy, wz]`
- why image is necessary:
  - the correct steering action depends on obstacle layout, not just the text

## Raw Session Layout

```text
go2_vla_collector/data/<session_id>/
  index.json
  episodes/
    ep_000001.json
  images/
    ep_000001/
      20260405_144137_001.jpg
```

## Raw Episode Shape

Each episode stores:

- `schema_version`
- `episode_id`
- `instruction`
- optional task metadata:
  - `task_family`
  - `target_type`
  - `target_description`
  - `target_instance_id`
  - `task_tags`
  - `collector_notes`
  - `instruction_source`
- `scene_id`
- `operator_id`
- `frames`

Each frame stores:

- `timestamp`
- `image`
- `instruction`
- `state`
- `raw_action`
- `control_action`
- `meta`

The frame-level `meta` block keeps timing fields plus task metadata copied down for debugging and replay alignment.

## Action Semantics

### `raw_action`

The raw controller input captured during collection.

Current fields:

- `vx`
- `vy`
- `wz`
- `camera_pitch`
- `keys`

### `control_action`

The training and deployment target.

Current fields:

- `vx`
- `vy`
- `wz`

Current rule:

- `control_action(vx, vy, wz)` is copied from `raw_action(vx, vy, wz)`

This separation is intentional. Later versions can remap `control_action` without changing the rest of the pipeline.

## Task Metadata Semantics

The raw schema and converted manifest now preserve these optional fields:

- `task_family`
  - expected values:
    - `legacy_motion`
    - `goal_navigation`
    - `visual_following`
    - `obstacle_aware_navigation`
- `target_type`
  - examples:
    - `door`
    - `apple`
    - `object`
    - `person`
    - `obstacle`
    - `box`
- `target_description`
  - operator-facing free text for the visible target, for example `red cone near left wall`
- `target_instance_id`
  - stable identifier within a scene if the same category appears multiple times
- `task_tags`
  - optional list for quick filtering such as `occluded`, `left_bias`, `low_light`
- `collector_notes`
  - optional free-text note for special cases
- `instruction_source`
  - `semantic_text` when the collector is run with an explicit semantic instruction
  - `motion_label` when the collector falls back to the legacy motion label

These fields are intentionally lightweight. They add enough semantics for visual task collection without redesigning the whole project.

## Converted Manifest Layout

```text
<output_root>/
  dataset.jsonl
  train.jsonl
  val.jsonl
  test.jsonl
  stats.json
  sessions/
    <session_id> -> raw session symlink or copy
```

Each manifest record contains:

- `schema_version`
- `sample_id`
- `session_id`
- `episode_id`
- `trajectory_index`
- `trajectory_step_index`
- `trajectory_length`
- `step_id`
- `timestamp`
- `instruction`
- `task_family`
- `target_type`
- `target_description`
- `target_instance_id`
- `task_tags`
- `collector_notes`
- `instruction_source`
- `state`
- `raw_action`
- `control_action`
- `action_chunk`
- `chunk_length`
- `scene_id`
- `operator_id`
- `image_path`
- `source_image_path`
- `split`

## Training Contract

Current baseline contract:

- input: `instruction + image + optional state`
- target: `control_action(vx, vy, wz)`

This contract is unchanged for the next-stage tasks. Only the data regime changes: image should become necessary to solve the task.

## Validation Contract

The validator and sanity-check tooling should now answer practical questions such as:

- does the dataset contain visually necessary task families at all
- does the same instruction appear across multiple scenes
- does the same instruction appear with multiple targets
- are target annotations present for visual tasks
- is the dataset still effectively instruction-only despite the new labels

This is the main diagnostic loop before attempting any stronger VLA model.

Future extension:

- replace the simple baseline with LLaDA-VLA
- keep the same observation and action interface
- optionally switch supervision from single-step action to `action_chunk`

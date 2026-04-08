# Go2 Visually Necessary Collection Plan

## Goal

Move from:

- `instruction -> canonical action`

to:

- `instruction + current visual scene -> control_action(vx, vy, wz)`

The collection target is not bigger model complexity. The collection target is a dataset where image is actually required.

## Collection Rule

For every new task family, the same instruction should appear in multiple scenes and require different actions because the image is different.

Good pattern:

- `go to the door` in scene A where the door is on the left
- `go to the door` in scene B where the door is on the right
- `go to the door` in scene C where the door is far ahead and partially occluded

Bad pattern:

- `go to the door` always recorded in one corridor with one fixed starting pose

## Phase 1: Goal-Directed Local Navigation

Priority: highest

### Recommended instruction templates

- `go to the door`
- `go to the apple`
- `go to the red object`
- `approach the box`

### Recommended metadata

- `task_family=goal_navigation`
- `target_type=door|apple|object|box`
- `target_description` describing appearance or location
- `target_instance_id` when multiple similar targets exist in one scene

### Minimum collection target

- episodes: at least 150 to 300
- scene diversity: at least 8 to 12 distinct scenes
- per instruction: at least 40 to 60 episodes
- target pose variation: target should appear left, right, center, near, far

### Trajectory guidance

- target length: about 2 to 8 seconds
- include approach, minor steering correction, and stop or near-stop at the end when practical
- vary start pose deliberately

### Failure modes to avoid

- target always centered before motion starts
- one instruction tightly bound to one scene
- target descriptions missing for multi-object scenes
- collecting only short straight-line approaches

## Phase 2: Visual Following

Priority: medium

### Recommended instruction templates

- `follow the person`
- `follow the person in front of you`

### Recommended metadata

- `task_family=visual_following`
- `target_type=person`
- `target_description` describing clothing or identity cue

### Minimum collection target

- episodes: at least 120 to 200
- scene diversity: at least 5 to 8 scenes
- target motion diversity: slow walk, slight turns, short stops, partial occlusion

### Trajectory guidance

- target length: about 4 to 12 seconds
- keep the person visible most of the time, but include realistic brief partial occlusion
- vary following distance and lateral offset

### Failure modes to avoid

- person always centered and walking straight
- one person, one background, one camera angle only
- target exits the frame for most of the trajectory
- operator uses text labels that do not distinguish multiple people

## Phase 3: Simple Obstacle-Aware Navigation

Priority: medium after phase 1 is stable

### Recommended instruction templates

- `avoid the object in front of you`
- `go around the obstacle and continue forward`

### Recommended metadata

- `task_family=obstacle_aware_navigation`
- `target_type=obstacle`
- `target_description` describing obstacle type and rough placement

### Minimum collection target

- episodes: at least 120 to 200
- scene diversity: at least 6 to 10 scenes
- layout diversity: obstacle left bias, right bias, center block, narrow gap

### Trajectory guidance

- target length: about 3 to 10 seconds
- include both avoidance steering and forward recovery
- vary obstacle size, distance, and free-space side

### Failure modes to avoid

- all obstacles appear in one fixed center location
- the same avoidance direction is always used
- no examples where free space is ambiguous
- obstacle metadata missing or too generic to audit later

## Practical Collection Workflow

One collector run should use one stable semantic task configuration:

```bash
./native/build/go2_collector \
  --network-interface eno1 \
  --scene-id corridor_a \
  --operator-id op_01 \
  --instruction "go to the door" \
  --task-family goal_navigation \
  --target-type door \
  --target-description "glass door at corridor end" \
  --target-instance-id corridor_a_door_01
```

Then repeat across more scenes, targets, and start poses with the same instruction template.

## What To Check After Every Collection Batch

Run:

```bash
python3 scripts/sanity_check_dataset.py --data-root data --output-dir outputs/sanity_check
python3 tools/validate_bc_dataset.py --dataset-root data --report-path outputs/validate_report.json
```

The batch is not ready if:

- `no_visually_necessary_task_family` appears
- instructions do not span multiple scenes
- instructions do not span multiple targets where applicable
- target metadata is missing for visual tasks
- replay pages show one canonical action pattern for each instruction regardless of image

## Baseline Usage

Use `tools/llada_vla_baseline_v2.py` only as a diagnostic:

- if `instruction_only ~= image_plus_instruction`, the dataset is still too text-dominant
- if image starts helping on these new tasks, the collection regime is moving in the right direction

Do not spend the main effort on more model complexity until phase 1 data shows real visual dependence.

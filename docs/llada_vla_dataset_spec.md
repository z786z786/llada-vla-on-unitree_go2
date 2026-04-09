# Go2 采集数据规范

## 目标

本文档定义当前 Go2 本地数据流水线使用的：

- 原始 session 结构
- 转换后的 manifest 结构

设计目标是：

- 当前阶段保持模型与接口尽量简单
- 与未来 LLaDA-VLA 部署侧任务接口保持一致

下一阶段的数据重点不再是单纯的规范动作标签，而是“视觉必需”的语言条件运动控制：

- 输入：`instruction + front RGB + optional state`
- 输出：`control_action(vx, vy, wz)`
- 关键性质：同一句 instruction 在不同视觉场景下应当对应不同动作

## 下一阶段任务族

### 1. `goal_navigation`

- 指令模板：
  - `go to the door`
  - `go to the apple`
  - `go to the red object`
  - `approach the box`
- 观测依赖：
  - 机器人必须在当前图像中定位目标
- 动作输出：
  - 高层 `control_action = [vx, vy, wz]`
- 为什么图像是必要的：
  - 同一句指令会因为目标位置和距离不同而需要左转、右转或直行

### 2. `visual_following`

- 指令模板：
  - `follow the person`
  - `follow the person in front of you`
- 观测依赖：
  - 机器人必须持续跟踪运动中的目标
- 动作输出：
  - 高层 `control_action = [vx, vy, wz]`
- 为什么图像是必要的：
  - 即使 instruction 不变，目标位置仍会随时间变化

### 3. `obstacle_aware_navigation`

- 指令模板：
  - `go around the obstacle and continue forward`
  - `avoid the object in front of you`
- 观测依赖：
  - 机器人必须从图像中判断可通行空间与障碍布局
- 动作输出：
  - 高层 `control_action = [vx, vy, wz]`
- 为什么图像是必要的：
  - 正确转向依赖障碍位置，而不只是文本本身

## 原始 Session 目录结构

```text
go2_vla_collector/data/<session_id>/
  index.json
  episodes/
    ep_000001.json
  images/
    ep_000001/
      20260405_144137_001.jpg
```

## 原始 Episode 结构

每条 episode 保存：

- `schema_version`
- `episode_id`
- `instruction`
- 可选任务元数据：
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

每一帧保存：

- `timestamp`
- `image`
- `instruction`
- `state`
- `raw_action`
- `control_action`
- `meta`

帧级 `meta` 会保留时间对齐信息与任务元数据，方便排查与回放。

## 动作语义

### `raw_action`

采集时记录的原始控制输入。

当前字段：

- `vx`
- `vy`
- `wz`
- `camera_pitch`
- `keys`

### `control_action`

训练与部署阶段使用的目标动作。

当前字段：

- `vx`
- `vy`
- `wz`

当前规则：

- `control_action(vx, vy, wz)` 直接拷贝自 `raw_action(vx, vy, wz)`

保留这层拆分是有意设计。后续如果需要重映射 `control_action`，不必改整个数据流水线。

## 任务元数据语义

原始 schema 与转换后的 manifest 会保留这些可选字段：

- `task_family`
  - 期望值：
    - `legacy_motion`
    - `goal_navigation`
    - `visual_following`
    - `obstacle_aware_navigation`
- `target_type`
  - 示例：
    - `door`
    - `apple`
    - `object`
    - `person`
    - `obstacle`
    - `box`
- `target_description`
  - 面向操作员的自由文本描述，例如 `red cone near left wall`
- `target_instance_id`
  - 同一 scene 中多目标场景下的稳定标识
- `task_tags`
  - 便于筛选的轻量标签，例如 `occluded`、`left_bias`、`low_light`
- `collector_notes`
  - 特殊情况备注
- `instruction_source`
  - 当显式传入语义 instruction 时为 `semantic_text`
  - 当回退为传统动作标签时为 `motion_label`

这些字段刻意保持轻量，只补充足够的语义信息，而不重做整套工程。

## 转换后的 Manifest 结构

```text
<output_root>/
  dataset.jsonl
  train.jsonl
  val.jsonl
  test.jsonl
  stats.json
  sessions/
    <session_id> -> 原始 session 的软链接或拷贝
```

每条 manifest record 包含：

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

## 训练接口约定

当前基线的接口约定是：

- 输入：`instruction + image + optional state`
- 输出：`control_action(vx, vy, wz)`

下一阶段不改接口本身，只改数据分布：图像必须真正变得“有用”。

## 验证约定

验证器与回放工具应当回答下面这些实际问题：

- 数据里是否真的包含视觉必需任务族
- 同一句 instruction 是否覆盖多个 scene
- 在适用场景下，同一句 instruction 是否覆盖多个 target
- 图像是否与动作、时间戳、instruction 对齐
- 当前数据是否仍然只是“文本到固定动作模板”的伪多模态

# Go2 视觉必需数据采集计划

## 目标

我们要从：

- `instruction -> canonical action`

转向：

- `instruction + 当前视觉场景 -> control_action(vx, vy, wz)`

采集重点不是一开始就追求更复杂的模型，而是先拿到一批“图像确实必需”的数据。

## 采集原则

对每个新任务族，都应该让“同一句 instruction 在多个 scene 中出现，并因为图像不同而对应不同动作”。

好的模式：

- 场景 A：`go to the door`，门在左侧
- 场景 B：`go to the door`，门在右侧
- 场景 C：`go to the door`，门在正前方且部分遮挡

坏的模式：

- `go to the door` 永远只在同一条走廊、同一起始位姿下采集

## 阶段 1：局部目标导航

优先级：最高

### 推荐指令模板

- `go to the door`
- `go to the apple`
- `go to the red object`
- `approach the box`

### 推荐元数据

- `task_family=goal_navigation`
- `target_type=door|apple|object|box`
- `target_description` 描述外观或位置
- 多目标场景下补充 `target_instance_id`

### 最低采集目标

- episode 数：至少 150 到 300
- scene 多样性：至少 8 到 12 个不同 scene
- 每个 instruction：至少 40 到 60 条 episode
- 目标位置变化：要覆盖左、右、中、近、远

### 轨迹建议

- 目标时长：约 2 到 8 秒
- 条件允许时，包含接近、轻微修正、结束前停下或接近停下
- 有意识地改变起始位姿

### 要避免的失败模式

- 动作开始前目标总是已经居中
- 某条 instruction 被死绑定到某个固定 scene
- 多目标场景缺失 `target_description`
- 只采短直线靠近轨迹

## 阶段 2：视觉跟随

优先级：中

### 推荐指令模板

- `follow the person`
- `follow the person in front of you`

### 推荐元数据

- `task_family=visual_following`
- `target_type=person`
- `target_description` 描述衣着或身份特征

### 最低采集目标

- episode 数：至少 120 到 200
- scene 多样性：至少 5 到 8 个 scene
- 目标运动多样性：慢走、轻微转向、短暂停顿、局部遮挡

### 轨迹建议

- 目标时长：约 4 到 12 秒
- 大多数时间保持目标可见，但允许真实短时遮挡
- 改变跟随距离与横向偏置

### 要避免的失败模式

- 人始终在画面中心并直线行走
- 只有一个人、一个背景、一个视角
- 目标大部分时间都跑出画面
- 文本标签无法区分多人场景中的目标

## 阶段 3：简单避障导航

优先级：在阶段 1 稳定后进行

### 推荐指令模板

- `avoid the object in front of you`
- `go around the obstacle and continue forward`

### 推荐元数据

- `task_family=obstacle_aware_navigation`
- `target_type=obstacle`
- `target_description` 描述障碍物类型与大致位置

### 最低采集目标

- episode 数：至少 120 到 200
- scene 多样性：至少 6 到 10 个 scene
- 布局多样性：左偏、右偏、中间堵塞、窄通道

### 轨迹建议

- 目标时长：约 3 到 10 秒
- 同时包含绕障动作与回正前进
- 改变障碍物大小、距离与空旷侧方向

### 要避免的失败模式

- 障碍总在同一固定中心位置
- 始终只往同一侧绕行
- 没有自由空间模糊的样本
- 障碍元数据缺失或过于泛化，后期无法审计

## 实际采集工作流

建议一次 collector 运行只对应一个稳定的语义任务配置：

```bash
./native/build/go2_collector   --network-interface eno1   --scene-id corridor_a   --operator-id op_01   --instruction "go to the door"   --task-family goal_navigation   --target-type door   --target-description "glass door at corridor end"   --target-instance-id corridor_a_door_01
```

然后在更多 scene、更多目标和更多起始位姿上重复相同 instruction 模板。

## 每一批采集后要检查什么

运行：

```bash
python3 scripts/sanity_check_dataset.py --data-root data --output-dir outputs/sanity_check
python3 tools/validate_bc_dataset.py --dataset-root data --report-path outputs/validate_report.json
```

若出现以下情况，说明这一批数据还不能直接用于训练：

- 出现 `no_visually_necessary_task_family`
- 同一句 instruction 没有跨多个 scene
- 适用时，同一句 instruction 没有跨多个 target
- 视觉任务缺少 target 元数据
- 回放页面显示“每条指令始终只有一种固定动作模板”

## 基线使用建议

`tools/llada_vla_baseline_v2.py` 只建议作为诊断工具：

- 如果 `instruction_only ~= image_plus_instruction`，说明数据仍然过于依赖文本
- 如果图像开始在这些新任务上提供帮助，说明采集方向是对的

在阶段 1 数据尚未体现真实视觉依赖之前，不建议把主要精力放在增加模型复杂度上。

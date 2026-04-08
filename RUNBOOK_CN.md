# Go2 本地采集联调手册

当前只保留一条本地主路径：

- `native/build/go2_collector` 采集
- `native/build/go2_collector_stage1` 第一阶段简单指令采集
- `scripts/sanity_check_dataset.py` 回放检查
- `tools/validate_bc_dataset.py` 验证可训练性
- `tools/convert_llada_vla_dataset.py` 转换训练清单
- `tools/llada_vla_baseline.py` 跑最小 baseline
- `tools/llada_vla_baseline_v2.py` 跑多模态诊断 baseline

## 1. 构建

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
cmake -S native -B native/build
cmake --build native/build -j
```

## 2. 启动 collector

第一阶段简单指令采集程序：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
./native/build/go2_collector_stage1 --network-interface eno1 --scene-id lab_test --operator-id wxh
```

说明：

- 这个程序固定为 `single_action`
- 不接受 `--instruction` / `--task-family` / `--target-*` 这类第二阶段语义任务参数
- episode 会自动按动作键标成 `go forward` / `turn left` 这类受控指令
- 默认输出目录是 `data_stage1/`，避免和后续长轨迹数据混在一起

通用 collector（保留第二阶段与视觉任务能力）：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
./native/build/go2_collector --network-interface eno1 --scene-id lab_test --operator-id wxh
```

注意：

- `scene_id`、`operator_id` 必须在启动时传入
- 现在支持在启动时直接传入语义任务元数据
- 程序常驻运行，但只有显式截取区间内的帧才会写盘
- 当前数据协议里每帧继续写 `raw_action` 和 `control_action`
- 键盘命令在发送前会做限幅：
  - `vx <= 0.5`
  - `vy <= 0.3`
  - `wz <= 0.8`

现在 collector 有两种采集模式：

- `--capture-mode single_action`
  - 旧模式，默认值
  - 按 `R` 后进入 arm
  - 检测到第一个单一运动键后，等待 0.5 秒再开始录制
  - 中途换向、混合按键会丢弃该段
  - 松键自动结束
- `--capture-mode trajectory`
  - 新模式，适合视觉必需任务
  - 按 `R` 立即开始录制
  - episode 内允许前进、转向、横移、停顿、多阶段变化
  - 按 `T` 结束并保存
  - 按 `ESC` 丢弃
  - 该模式下必须传 `--instruction`

如果你在采集“视觉必需”任务，建议每次启动 collector 固定一个任务配置，例如：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
./native/build/go2_collector \
  --network-interface eno1 \
  --scene-id corridor_a \
  --operator-id wxh \
  --instruction "go to the door" \
  --task-family goal_navigation \
  --target-type door \
  --target-description "corridor end glass door" \
  --target-instance-id corridor_a_door_01 \
  --task-tags indoor,daylight
```

其他两个典型例子：

```bash
./native/build/go2_collector \
  --network-interface eno1 \
  --scene-id hall_b \
  --operator-id wxh \
  --instruction "follow the person in front of you" \
  --task-family visual_following \
  --target-type person \
  --target-description "person with black jacket"
```

```bash
./native/build/go2_collector \
  --network-interface eno1 \
  --scene-id lab_boxes \
  --operator-id wxh \
  --instruction "go around the obstacle and continue forward" \
  --task-family obstacle_aware_navigation \
  --target-type obstacle \
  --target-description "cardboard box blocking the center path"
```

说明：

- 一次 collector 运行，建议只对应一个稳定的语义任务配置
- 如果不传 `--instruction`，collector 会退回旧的运动标签模式
- `task_tags` 是可选的 CSV 标签，方便后续筛选
- 视觉任务里尽量补齐 `target_type` 和 `target_description`
- 真正要采“go to door / follow person / avoid obstacle”这类长轨迹，建议明确使用 `--capture-mode trajectory`

长轨迹采集示例：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
./native/build/go2_collector \
  --network-interface eno1 \
  --capture-mode trajectory \
  --scene-id corridor_a \
  --operator-id wxh \
  --instruction "go to the door" \
  --task-family goal_navigation \
  --target-type door \
  --target-description "corridor end glass door"
```

## 3. 键盘操作

运动控制：

- `W/S`：前进 / 后退
- `A/D`：左移 / 右移
- `Q/E`：左转 / 右转

采集控制：

- `R`：按当前 `capture_mode` 启动采集
- `T`：如果当前正在记录，则结束并保存当前区间
- `ESC`：取消当前已 arm 或正在进行的区间

安全与辅助：

- `Space`：急停，并 latch safety fault
- `C`：清除已恢复的 safety fault，或切换 stand up / down
- `P`：打印当前状态
- `H`：打印帮助
- `X`：退出程序

## 4. 下一阶段采集目标

当前项目已经证明：

- 指令到 canonical action 的映射是可学的
- 但现有数据还不足以证明图像是必要输入

下一阶段要采的不是更多“go forward”，而是视觉必需任务。优先顺序：

1. `goal_navigation`
   - 例子：`go to the door`、`go to the apple`、`approach the box`
2. `visual_following`
   - 例子：`follow the person`
3. `obstacle_aware_navigation`
   - 例子：`avoid the object in front of you`、`go around the obstacle and continue forward`

核心要求：

- 同一句 instruction，必须出现在多个 scene
- 同一句 instruction，必须因为视觉场景不同而对应不同动作
- 不要把 instruction 和单一固定场景绑定死

详细采集计划见 `docs/visually_necessary_collection_plan.md`。

推荐操作流程：

- 如果是旧的单动作短片段数据，继续用 `single_action`
- 如果是视觉必需任务，优先改用 `trajectory`
- 在 `trajectory` 模式下，一条 episode 可以包含：
  - 起步
  - 轻微对准
  - 转向修正
  - 绕障
  - 再次前进
  - 短暂停顿

## 5. Safety 规则

当前会触发 fault 的条件：

- 状态流超时
- `roll` 超限
- `pitch` 超限
- 手动急停

fault 行为：

- 立即 `StopMove()`
- 当前采集区间作废
- 程序进入 fault 状态
- 只有在机体恢复稳定后按 `C` 才能继续采集

## 6. 数据回放检查

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 scripts/sanity_check_dataset.py --data-root data --output-dir outputs/sanity_check
```

生成后的 `index.html` 和 `episodes/*.html` 用来检查：

- 图像是否正常
- instruction 是否和画面匹配
- `task_family / target_type / target_description` 是否填完整
- `vx/vy/wz` 是否和回放语义一致
- 动作曲线是否抖动、断裂、全零
- 同一句指令是否出现在不同 scene / 不同 target 下

## 7. 数据验证

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/validate_bc_dataset.py --dataset-root data --report-path outputs/validate_report.json
```

这里重点看：

- `missing_scene_id`
- `missing_operator_id`
- `task_family_metadata_missing`
- `no_visually_necessary_task_family`
- `instruction_scene_overlap_low`
- `instruction_target_variation_low`
- 图像文件是否都存在
- `state_action_delta` / `state_image_delta` 是否异常
- 视觉任务是否缺少 `target_type` / `target_description`

## 8. 转换与 baseline

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/convert_llada_vla_dataset.py --raw-root data --output-root data/llada_vla_converted --overwrite
python3 tools/llada_vla_baseline.py --dataset-root data/llada_vla_converted --save-model data/llada_vla_converted/baseline_model.json
```

baseline 的目的不是追求高性能，而是先确认：

- 数据是不是可学
- 输入输出定义是不是合理
- 未来接 LLaDA-VLA 时能不能直接复用这套接口

PyTorch baseline v2 用于多模态诊断：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/llada_vla_baseline_v2.py \
  --dataset-root data/llada_vla_converted \
  --output-dir outputs/baseline_v2_goal_nav \
  --ablation-mode image_plus_instruction
```

如果未来在新采集数据上仍然出现 `instruction_only >= image_plus_instruction`，优先先检查数据分布，而不是先加大模型复杂度。

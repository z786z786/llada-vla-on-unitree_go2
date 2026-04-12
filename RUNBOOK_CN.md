# Go2 本地采集联调手册

当前只保留一条本地主路径：

- `native/build/go2_collector` 采集
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

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
./native/build/go2_collector --network-interface eno1 --scene-id lab_test --operator-id wxh --instruction "go to the door"
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

现在 collector 固定使用 `trajectory` 长轨迹采集流程：

- 适合视觉必需任务
- 按 `R` 立即进入采集流程
- episode 内允许前进、转向、横移、停顿、多阶段变化
- 按 `T` 请求结束，等待动作回落后进入标注
- 按 `ESC` 丢弃
- 该模式下必须传 `--instruction`

状态机说明见 `docs/collector_state_machine.md`。

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
  --target-description "corridor end glass door"
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
- `--instruction` 现在始终必填
- 视觉任务里尽量补齐 `target_type` 和 `target_description`
- `--capture-mode trajectory` 仍可传入做兼容校验，但已经不是必需项

长轨迹采集示例：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
./native/build/go2_collector \
  --network-interface eno1 \
  --scene-id corridor_a \
  --operator-id wxh \
  --instruction "go to the door" \
  --task-family goal_navigation \
  --target-type door \
  --target-description "corridor end glass door"
```

离线预览界面时，可直接启动：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
./native/build/go2_collector --preview-ui --web-ui
```

说明：

- 预览模式会自动启用 Web UI，并跳过 Go2 DDS / sport / video 初始化
- 默认切到 `tty` 输入后端，可直接用键盘预览 `R/T/ESC/1-4`
- 不需要传 `--network-interface`
- 如需模拟不同页面文案，可继续传 `--scene-id`、`--instruction`、`--task-family` 等参数

## 3. 操作设备

当前 collector 默认使用 `wireless_controller` 输入后端，直接读取 Go2 原生手柄 DDS 话题 `rt/wirelesscontroller`。

手柄控制：

- `Start`：保持 Go2 原生行为，并在首次按下后解除 collector 启动门控
- 左摇杆 `ly/lx`：前进 / 后退、左移 / 右移
- 右摇杆 `rx`：左转 / 右转
- `A`：启动 trajectory 采集
- `B`：如果当前正在记录，则请求结束并进入标注
- `X`：丢弃当前区间
- `R2`：急停，并 latch safety fault
- `Y`：清除已恢复的 safety fault，或切换 stand up / down
- 方向键 `上/右/下/左`：待标注时提交 `1/2/3/4`

键盘回退控制：

运动控制：

- `W/S`：前进 / 后退
- `A/D`：左移 / 右移
- `Q/E`：左转 / 右转

采集控制：

- `R`：启动 trajectory 采集
- `T`：如果当前正在记录，则请求结束并进入标注
- `ESC`：丢弃当前区间

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

- 统一使用 `trajectory`
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

## 6. 从 Orin 同步原始 session 到本机

默认把 Orin 上 `go2_vla_collector/data/` 里的原始采集 session 增量拉到当前机器的 `go2_vla_collector/data/`。

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
./scripts/sync_orin_sessions.sh --dry-run
./scripts/sync_orin_sessions.sh
```

说明：

- 默认源主机是 `unitree@192.168.123.18`
- 默认源目录是 `/home/unitree/unitree_go2/go2_vla_collector/data`
- 默认目标目录是 `/home/xiaohui/unitree_go2/go2_vla_collector/data`
- 只同步顶层形如 `YYYYMMDD_HHMMSS` 的 collector session
- 不会同步 `llada_vla_converted`、`outputs`、`build`、`single_action` 等目录
- SSH / rsync 需要时会交互输入密码，脚本不会保存密码

如果只想拉一个 session：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
./scripts/sync_orin_sessions.sh --session-id 20260411_171528
```

这个脚本只负责把 raw session 拉回本机；后续回放检查、转换和上传服务器训练仍按下面流程继续。

## 7. 数据回放检查

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 scripts/sanity_check_dataset.py --data-root data --output-dir outputs/sanity_check
```

如果希望页面改分后直接写回 `data/quality_review.json`，建议直接这样启动：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 scripts/sanity_check_dataset.py --data-root data --output-dir outputs/sanity_check --serve
```

生成后的 `index.html` 和 `episodes/*.html` 用来检查：

- 图像是否正常
- instruction 是否和画面匹配
- `task_family / target_type / target_description` 是否填完整
- `vx/vy/wz` 是否和回放语义一致
- 动作曲线是否抖动、断裂、全零
- 同一句指令是否出现在不同 scene / 不同 target 下

现在报告页还支持对录制分数做回调：

- `1` 好样本
- `2` 可用但不完美
- `3` 失败但保留 / 质量差，后续处理排除

建议操作：

1. 在 `index.html` 先用顶部“指定分数”筛选器快速找到目标样本
2. 需要排除的样本保持或改成 `3`
3. 如果是 `--serve` 打开的页面，改分后会直接写回 `data/quality_review.json`
4. 如果是静态 HTML，点击“绑定回调文件”，选择 `data/quality_review.json`
5. 如果后面要回调，把顶部“指定分数”切到 `3`，就能集中把这批样本改回 `2`

## 8. 数据验证

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/validate_bc_dataset.py --dataset-root data --report-path outputs/validate_report.json
```

如果筛检文件不在默认位置，可以显式传入：

```bash
python3 tools/validate_bc_dataset.py \
  --dataset-root data \
  --report-path outputs/validate_report.json \
  --quality-review-path data/quality_review.json
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

## 9. 转换与 baseline

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/convert_llada_vla_dataset.py --raw-root data --output-root data/llada_vla_converted --overwrite
python3 tools/llada_vla_baseline.py --dataset-root data/llada_vla_converted --save-model data/llada_vla_converted/baseline_model.json
```

如果 `data/quality_review.json` 已存在，转换、分布标签生成和从 raw session 直接读取的 baseline 会自动跳过 `quality_label=3` 的 episode；也可以手动指定：

```bash
python3 tools/convert_llada_vla_dataset.py \
  --raw-root data \
  --output-root data/llada_vla_converted \
  --quality-review-path data/quality_review.json \
  --overwrite
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

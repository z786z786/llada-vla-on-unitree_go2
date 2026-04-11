# llada-vla-on-unitree_go2

这是一个面向 Go2 的本地数据采集与离线验证流水线，用于分阶段验证 LLaDA-VLA 风格任务接口。

当前仓库同时覆盖两类数据范式：

- 第一类：规范动作映射，例如 `go forward -> [vx, vy, wz]`
- 第二类：视觉必需任务，即必须结合 `instruction + image (+ optional state)` 才能决定动作

当前工作区提供：

- `native/build/go2_collector`：主采集程序，仅支持 `trajectory` 长轨迹语义任务
- `scripts/sanity_check_dataset.py`：生成 HTML 回放与数据体检报告
- `tools/validate_bc_dataset.py`：离线可训练性验证工具
- `tools/derive_distribution_labels.py`：按轨迹属性分布生成左右/远近标签
- `tools/convert_llada_vla_dataset.py`：原始 session 转训练清单工具
- `tools/llada_vla_baseline.py`：轻量行为克隆基线
- `tools/llada_vla_baseline_v2.py`：多模态诊断基线
- `tools/summarize_baseline_v2_runs.py`：汇总多个 baseline_v2 实验目录的关键指标

## 仓库结构

```text
go2_vla_collector/
  native/
  scripts/
  tools/
  docs/
  data/          # 由 collector 生成的原始采集数据
  outputs/       # 报告、验证结果与基线输出
```

生成目录默认不纳入 git 管理。

## 任务接口

当前学习目标定义为：

- 输入：`instruction + 前视 RGB + 可选机器人状态`
- 输出：`control_action(vx, vy, wz)`

采集器会同时保存两类动作：

- `raw_action`：采集时记录的原始操作信号
- `control_action`：训练与部署阶段使用的目标动作

当前版本里，两者在 `vx / vy / wz` 上保持一致；保留这种拆分是为了后续改动作映射时，不需要重写整个数据流水线。

## Session 结构

collector 默认把每次运行写到 `data/` 下的新 session：

```text
data/<session_id>/
  index.json
  episodes/
    ep_000001.json
  images/
    ep_000001/
      20260405_144137_001.jpg
```

每条 episode 包含：

- `schema_version`
- `instruction`、`scene_id`、`operator_id`
- 可选任务元数据：`task_family`、`target_type`、`target_description`、`target_instance_id`、`task_tags`、`collector_notes`、`instruction_source`
- `frames[]`
- 每帧的 `state`、`raw_action`、`control_action`、`image` 与时间戳

## 构建

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
cmake -S native -B native/build
cmake --build native/build -j
```

构建后主要可执行文件为：

- `./native/build/go2_collector`

## 采集流程

当前 collector 仅保留面向视觉任务的 `trajectory` 长轨迹采集。

长轨迹语义任务示例：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
./native/build/go2_collector   --network-interface eno1   --capture-mode trajectory   --scene-id corridor_a   --operator-id op_01   --instruction "go to the door"   --task-family goal_navigation   --target-type door   --target-description "glass door at corridor end"
```

其他语义任务示例：

```bash
./native/build/go2_collector   --network-interface eno1   --scene-id hall_b   --operator-id op_01   --instruction "follow the person in front of you"   --task-family visual_following   --target-type person   --target-description "adult with black jacket"
```

```bash
./native/build/go2_collector   --network-interface eno1   --scene-id lab_boxes   --operator-id op_01   --instruction "go around the obstacle and continue forward"   --task-family obstacle_aware_navigation   --target-type obstacle   --target-description "cardboard box blocking center path"
```

采集注意事项：

- `scene_id` 与 `operator_id` 始终必填
- `--instruction` 现在始终必填
- 同一轮语义任务采集应保持任务配置稳定

### 采集模式

- `trajectory`
  - 按 `R` 立即开始录制
  - 允许多阶段动作变化
  - 按 `T` 请求结束，等待动作回落后进入标注
  - 按 `ESC` 丢弃
  - 必须提供 `--instruction`

状态机说明见 `docs/collector_state_machine.md`。

## 操作设备

当前默认输入后端是 `wireless_controller`，直接读取 Go2 原生手柄 DDS 话题 `rt/wirelesscontroller`。

手柄映射：

- `Start`：保持 Go2 原生行为，并在首次按下后解除 collector 启动门控
- 左摇杆 `ly/lx`：前进后退 / 左右平移
- 右摇杆 `rx`：左转 / 右转
- `A`：启动 trajectory 采集
- `B`：请求结束当前采集段并进入标注
- `X`：丢弃当前区间
- `R2`：急停
- `Y`：清除故障或切换站立 / 趴下
- 方向键 `上/右/下/左`：待标注时提交 `1/2/3/4`

键盘仍可作为回退输入：

- `W/S`：前进 / 后退
- `A/D`：左移 / 右移
- `Q/E`：左转 / 右转
- `R`：启动 trajectory 采集
- `T`：请求结束当前采集段并进入标注
- `ESC`：丢弃当前区间
- `Space`：急停
- `C`：清除故障或切换站立 / 趴下
- `P`：打印状态
- `H`：打印帮助
- `X`：退出

## 离线检查与验证

生成 HTML 回放报告：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 scripts/sanity_check_dataset.py --data-root data --output-dir outputs/sanity_check
```

验证数据是否适合训练：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/validate_bc_dataset.py --dataset-root data --report-path outputs/validate_report.json
```

验证器会关注例如：

- 元数据缺失
- 图像或时间戳错位
- 指令分布失衡
- 同一句指令跨场景 / 跨目标覆盖不足
- 是否缺少真正的视觉必需任务族

按当前数据分布生成 `左/中/右` 和 `近/中/远` 标签：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/derive_distribution_labels.py \
  --dataset-root data \
  --output-dirname derived_labels \
  --summary-path outputs/distribution_label_summary.json
```

说明：

- 默认按全体 episode 的三分位数切分
- 左右默认使用 `lateral_displacement`
- 远近默认使用 `integrated_planar_distance`
- 若本地坐标方向和你的语义相反，可加 `--invert-side-sign`

## 转换与训练

把原始 session 转成训练清单：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/convert_llada_vla_dataset.py --raw-root data --output-root data/llada_vla_converted --overwrite
```

如果希望转换前自动生成这批 `derived_labels`：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/convert_llada_vla_dataset.py \
  --raw-root data \
  --output-root data/llada_vla_converted \
  --derive-labels distribution \
  --derive-summary-path outputs/distribution_label_summary.json \
  --overwrite
```

运行轻量基线：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/llada_vla_baseline.py   --dataset-root data/llada_vla_converted   --save-model data/llada_vla_converted/baseline_model.json
```

运行多模态诊断基线：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/llada_vla_baseline_v2.py   --dataset-root data/llada_vla_converted   --output-dir outputs/baseline_v2_goal_nav   --ablation-mode image_plus_instruction
```

汇总多个 baseline_v2 实验目录：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 tools/summarize_baseline_v2_runs.py \
  --outputs-root outputs \
  --output-path outputs/baseline_v2_summary.md
```

## 推荐检查命令

Python 工具链静态检查：

```bash
python3 -m py_compile tools/llada_vla_common.py   tools/derive_distribution_labels.py   tools/convert_llada_vla_dataset.py   tools/validate_bc_dataset.py   tools/llada_vla_baseline.py   scripts/sanity_check_dataset.py
```

检查原生 collector 参数：

```bash
./native/build/go2_collector --help
```

## 更多文档

- `RUNBOOK_CN.md`
- `docs/pilot_collection_plan.md`
- `docs/visually_necessary_collection_plan.md`
- `docs/seed5_visual_followup_plan.md`
- `docs/llada_vla_dataset_spec.md`

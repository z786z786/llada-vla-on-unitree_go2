# Seed5 实验复盘与下一轮视觉必要采集计划

## 当前实验结论

基于 `split_seed=5` 的随机 trajectory 划分：

- `image_plus_instruction`
  - `val_rmse = 0.0851`
  - `test_rmse = 0.1537`
- `instruction_only`
  - `val_rmse = 0.0922`
  - `test_rmse = 0.1421`
- `mean baseline`
  - `test_rmse = 0.1461`

当前最重要的信号不是“图像完全没用”，而是：

- 图像分支在验证集上有帮助，但在测试集上没稳定转化成收益
- 文本单模态已经足够解释当前大部分动作变化
- 这批数据还没有把“同一句 instruction 因视觉差异而必须输出不同动作”压得足够强

## 为什么会这样

### 1. instruction 语义仍然过强

当前主指令几乎都为：

- `go to the door`

模型只要学会一个默认动作模板，就能得到不差的分数。

### 2. 目标外观与场景变化还不够“对抗”

即使做了左右/远近覆盖，很多 episode 仍然可能共享：

- 相近场景
- 相近门外观
- 相近起始朝向
- 相近的接近方式

这样视觉分支学到的是“熟悉样子”，不是“必须看图才能决策”。

### 3. 动作目标过于平滑

现在监督目标是逐帧 `vx / vy / wz`，而很多轨迹主要变化集中在：

- `vx` 稳定前进
- `wz` 小幅修正

如果图像差异只对应轻微转向，文本或平均先验也能吃到大部分收益。

## 下一轮采集目标

下一轮不是先追求“更多”，而是追求“更难让 instruction_only 蒙对”。

最低目标：

- 总 episode：`180 ~ 240`
- `goal_navigation` 占主力：`120 ~ 160`
- `obstacle_aware_navigation`：`40 ~ 60`
- `visual_following`：`20 ~ 40`

每个任务族都要保证：

- 同一句 instruction 跨多个 scene
- 同一句 instruction 跨多个目标位置
- 同一句 instruction 跨多个目标外观
- 同一句 instruction 覆盖多个明显不同的控制响应

## 优先级 1：把 `go to the door` 采成真正视觉必需

### 推荐固定 instruction

只保留一句：

- `go to the door`

但要刻意让这句话对应明显不同的动作。

### 每个 scene 要覆盖的起始构型

每个门至少采：

- 左近：门在左侧且较近
- 左远：门在左侧且较远
- 中近：门基本居中且较近
- 中远：门基本居中且较远
- 右近：门在右侧且较近
- 右远：门在右侧且较远

每种构型至少：

- `3 ~ 5` 条有效 episode

如果一个 scene 做不全，就换 scene 补齐。

### 强制加入“易混淆负例”

每条 `go to the door` 的 scene 里尽量引入：

- 两扇门或门状结构
- 门旁边的高反光板/柜门
- 相似颜色的非目标大平面

并用 `target_description` 写清楚，例如：

- `gray iron door near left wall`
- `white glass door at corridor end`

这样模型必须依赖视觉定位目标，而不是只学“朝一个固定大轮廓前进”。

## 优先级 2：加入更强的避障视觉依赖

推荐固定 instruction：

- `go around the obstacle and continue forward`

这一类样本最容易拉开：

- `instruction_only`
- `image_plus_instruction`

因为同一句话下：

- 障碍在左，就该往右绕
- 障碍在右，就该往左绕
- 障碍居中但左边更空，就应优先左绕
- 障碍居中但右边更空，就应优先右绕

### 最低覆盖要求

每个障碍物布局至少采：

- 左偏障碍
- 右偏障碍
- 中间障碍但左侧可通
- 中间障碍但右侧可通

每种布局至少：

- `8 ~ 12` 条 episode

## 优先级 3：减少轨迹中“纯模板前进”的比例

下一轮采集时，尽量减少下面这种过易样本：

- 起始时目标已经接近中心
- 全程只要稳定前进，最多轻微回正

建议在采集时增加：

- 更偏的起始角度
- 更大的横向偏移
- 起点到目标之间的中间遮挡
- 需要先明显转向再前进的路径

## 操作员执行规范

每次开始前先问自己两个问题：

- 如果只看 instruction，不看图，我能猜中大部分动作吗？
- 这个 episode 的第一秒动作，是否会因为目标位置不同而明显改变？

如果答案分别是：

- 能
- 不会

那这一条大概率不够“视觉必要”。

## 推荐采集配额

### A. 门导航

- `6 ~ 8` 个 scene
- 每个 scene `12 ~ 20` 条
- 合计 `100 ~ 140` 条

### B. 避障前行

- `4 ~ 6` 个 scene
- 每个 scene `8 ~ 12` 条
- 合计 `40 ~ 60` 条

### C. 视觉跟随

- `3 ~ 4` 个 scene
- 每个 scene `8 ~ 10` 条
- 合计 `24 ~ 40` 条

## 每批采集后必须做的检查

### 1. 数据分布检查

运行：

```bash
python3 go2_vla_collector/tools/derive_distribution_labels.py \
  --dataset-root go2_vla_collector/data \
  --output-dirname derived_labels \
  --summary-path go2_vla_collector/outputs/distribution_label_summary_next.json
```

要求：

- 左中右不要严重失衡
- 近中远不要严重失衡

### 2. 训练前切分检查

运行：

```bash
python3 go2_vla_collector/tools/convert_llada_vla_dataset.py \
  --raw-root go2_vla_collector/data \
  --output-root go2_vla_collector/data/llada_vla_converted \
  --split-mode by_trajectory \
  --split-seed 5 \
  --derive-labels distribution \
  --overwrite
```

要求：

- `train / val / test` 都有样本
- `val / test` 尽量都覆盖多个 session

### 3. 对照实验检查

必须至少同时跑：

- `instruction_only`
- `image_plus_instruction`

如果下一轮仍然出现：

- `instruction_only <= image_plus_instruction`

说明图像必要性仍然不够，优先继续修数据，而不是继续堆模型。

## 这轮实验后的判断标准

下一轮数据如果有效，至少应该出现下面两个信号中的一个：

- `image_plus_instruction` 的 `test_rmse` 明显低于 `instruction_only`
- 图像模型在 `wz` 维度上比文本模型更稳，尤其在避障和左右偏目标上

若二者都没有出现，优先继续增强：

- 场景差异
- 起始位姿差异
- 目标混淆度
- 障碍布局差异

而不是先改更大的网络。

# 数据回放检查报告

使用 `scripts/sanity_check_dataset.py` 为已采集 session 生成本地 HTML 报告。

## 命令

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 scripts/sanity_check_dataset.py --data-root data --output-dir outputs/sanity_check
```

如果希望页面改分后直接写回 `data/quality_review.json`，推荐直接启动本地服务：

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 scripts/sanity_check_dataset.py --data-root data --output-dir outputs/sanity_check --serve
```

可选参数：

- `--max-episodes N`：只检查前 `N` 条 episode，适合快速冒烟测试
- `--num-samples N`：每条 episode 页面展示的随机静态帧数量
- `--seed N`：随机抽样种子
- `--serve`：生成报告后启动本地 HTTP 服务，页面改分可直接写回 `quality_review.json`
- `--host/--port`：本地服务监听地址与端口

## 输出内容

报告目录包含：

- `index.html`：数据集总览与 episode 列表
- `episodes/*.html`：每条 episode 的独立回放页面
- `summary.json`：机器可读的汇总信息
- `quality_review.json`：建议通过页面“绑定回调文件”直接写到 `data/quality_review.json`，用于记录对原始录制分数的回调结果

## 回放页面会展示什么

每条 episode 页面通常包含：

- 类似视频的图像回放
- `instruction` 叠加层
- `vx`、`vy`、`wz` 叠加层
- 动作随时间变化曲线
- 动作直方图
- 随机静态帧抽查区域
- 录制分数回调区：可查看原始 `1/2/3` 分数、按需要改分、填写备注、导入/导出回调 JSON

这是人工检查“数据是否像一条完整控制轨迹”的第一入口。

## 录制分数回调工作流

建议在 `index.html` 或 `episodes/*.html` 中直接回调录制时的分数：

- `1`：好样本
- `2`：可用但不完美
- `3`：失败但保留 / 质量差，后续处理排除

推荐流程：

1. 在页面中用“指定分数”筛出目标样本
2. 把需要排除的样本保持或改成 `3`
3. 若你是用 `--serve` 打开的页面，修改会直接写回 `data/quality_review.json`
4. 若你是直接打开静态 HTML，再点击“绑定回调文件”，选择 `data/quality_review.json`
5. 如需手动备份，也可以点击“导出回调 JSON”
6. 之后运行验证、标签派生、转换和 baseline 时，当前分数为 `3` 的样本会自动跳过

如果后续回调时要把 `3` 改回 `2`：

- 在 `index.html` 顶部使用“指定分数 -> 仅 3 失败但保留 / 质量差”快速筛出所有差样本
- 逐条改成 `2`
- 再次导出并覆盖 `data/quality_review.json`

## 重点检查项

回放画面：

- 错帧、空白帧或图像延迟
- instruction 与视觉场景不匹配
- 动作与图像变化明显矛盾

动作曲线：

- 长时间全零
- 突兀跳变或明显截断
- 不合理振荡

直方图：

- 数据分布极端偏置
- 缺少转向或横移样本
- 超出预期范围的异常值

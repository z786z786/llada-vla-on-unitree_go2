# 数据回放检查报告

使用 `scripts/sanity_check_dataset.py` 为已采集 session 生成本地 HTML 报告。

## 命令

```bash
cd /home/xiaohui/unitree_go2/go2_vla_collector
python3 scripts/sanity_check_dataset.py --data-root data --output-dir outputs/sanity_check
```

可选参数：

- `--max-episodes N`：只检查前 `N` 条 episode，适合快速冒烟测试
- `--num-samples N`：每条 episode 页面展示的随机静态帧数量
- `--seed N`：随机抽样种子

## 输出内容

报告目录包含：

- `index.html`：数据集总览与 episode 列表
- `episodes/*.html`：每条 episode 的独立回放页面
- `summary.json`：机器可读的汇总信息

## 回放页面会展示什么

每条 episode 页面通常包含：

- 类似视频的图像回放
- `instruction` 叠加层
- `vx`、`vy`、`wz` 叠加层
- 动作随时间变化曲线
- 动作直方图
- 随机静态帧抽查区域

这是人工检查“数据是否像一条完整控制轨迹”的第一入口。

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

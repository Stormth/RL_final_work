# RL_final_work · 执行反馈驱动的代码修复强化学习

强化学习课程设计。用**隐藏测试作为可验证奖励**，对 `Qwen2.5-Coder-7B` 做 LoRA + GRPO/GDPO 训练：模型先"一次生成"，失败后把执行反馈回灌，再"修复一轮"；用 GDPO 解耦多奖励归一化，解决"做对"与"省一轮 / 不超时"之间的尺度冲突。

> 状态：**开题阶段**（方案 v2）。代码尚未开始，W1 起逐步落地。

## 文档

- [课程设计方案 v2](docs/课程设计方案.md) — 问题定义、方法（GRPO/GDPO）、奖励设计、数据、实验与五周计划
- 附录 A 保留了 v1 的"编排层 RL"方案：作为降级预案与对照基线

## 核心设计

1. **工具链是 RL 环境，不是被调度的对象。** 沙箱 + 逐 assert 判定 + 反馈构造，构成可验证奖励的全部来源。
2. **两轮，而不是 8 步 Agent。** `生成 → 修复` 已能验证"模型学会用执行反馈"，而多轮 rollout 的基建成本是两轮的 5 倍以上；W4 有余力再扩到 3 轮。
3. **GDPO 而不是直接加权求和。** 正确性是 0/1 稀疏信号，成本是稠密惩罚；加权求和后组内标准差被成本主导，模型会学"少调一轮"而不是"改对"。
4. **数据侧两道保险。** 时间切分（测试题发布时间晚于模型截止日期）+ 难度校准（只保留 base 通过率 `0.1 < p < 0.7` 的题），前者防背题，后者保证组内有梯度。

## 计划与 Gate

| 周 | 目标 | Gate |
|---|---|---|
| W1 | 数据清洗/难度校准、沙箱与判定；1.5B 单轮 GRPO 冒烟 | G1：reward 曲线能涨 |
| W2 | 7B 单轮 GRPO + SFT 基线 | **G2：pass@1 ≥ base +5pt** |
| W3 | 两轮 rollout，GRPO-2turn | G3：修复轮 ≥ +3pt |
| W4 | 三奖励 + GDPO，与 GRPO 对照 | G4：出主结果表与 Pareto |
| W5 | 一键复现、报告、答辩 | — |
| W6 | 缓冲；（可选）T=3 扩展 | — |

**G2 不过就降级**：切回 v1 的编排层 Q-learning（环境已就绪），训练线作为探索性实验写入报告。

## 目录（计划）

```
.
├── docs/课程设计方案.md
├── env/          # sandbox / judge / 反馈模板
├── data/         # 清洗、难度校准、时间切分
├── train/        # GRPO / GDPO、奖励函数、配置
├── eval/         # 跑分、图表
└── scripts/
```

## 相关方法

[GDPO](https://github.com/NVlabs/GDPO) · [GD²PO](https://aclanthology.org/2026.findings-emnlp.1070/) · [TRL](https://github.com/huggingface/trl) · [AlphaCodium](https://github.com/Codium-ai/AlphaCodium) · [LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset)

## 许可证

待定。

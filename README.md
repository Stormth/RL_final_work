# RL_final_work · 轻量大模型工具链求解算法题

强化学习课程设计。核心问题：**在固定的模型调用预算下，学会"下一步该向哪个工具要反馈"**——让本地 Qwen 反复经历「写码 → 被工具打回 → 改码」，用 RL 决定工具调用顺序与止损时机，在未见过的 LeetCode 题上以更低成本取得更高通过率。

> 状态：**开题阶段**。方案 v1 已完成，代码尚未开始（W1 起逐步落地）。

## 文档

- [课程设计方案 v1](docs/课程设计方案.md) — 问题定义、系统设计、数据、实验设计与五周计划

## 三个关键设计决策

1. **RL 放在编排层，不训练模型。** 模型只做两件事：按题面写首解、按失败信息改代码。RL 智能体不生成代码，只从 7 个动作里选下一个（跑公开样例 / 随机对拍 / 定向修复 / 重写 / 复杂度审查 / 提交）。理由是 7B 模型做 policy gradient 需要上万条 rollout 和显存，五周内跑不出来；而编排层状态空间小，表格 Q-learning 单机几小时收敛，还能画成"状态 × 动作"热力图。
2. **目标函数带预算约束。** `max E[AC] − λ·cost`，扫 λ 得到成本–成功率 Pareto 曲线。这是与 AlphaCodium 式固定流程最本质的区别。
3. **提交动作不暴露失败用例。** `submit` 只回 AC/WA，迫使策略先花调用去收集信息，而不是盲目重试。

## 计划

| 周 | 里程碑 |
|---|---|
| W1 | 数据清洗 + 沙箱 + 逐 assert 判定；单次生成基线 pass@1 与耗时（**首解通过率 spike**） |
| W2 | fuzz / audit 工具 + 固定脚本 workflow；与基线对照 |
| W3 | `ToolChainEnv` + 表格 Q-learning 跑通，学习曲线初版 |
| W4 | 3 seeds 正式实验 + 消融 + Pareto |
| W5 | 一键复现、文档、报告与答辩 |
| W6 | 缓冲；（可选）小规模 GRPO 加分项 |

## 目录

```
.
├── README.md
└── docs/
    └── 课程设计方案.md
```

## 相关工作

[AlphaCodium](https://github.com/Codium-ai/AlphaCodium) · [LDB](https://github.com/FloridSleeves/LLMDebugger) · [AFlow](https://arxiv.org/abs/2410.10762) · [MapCoder](https://aclanthology.org/2024.acl-long.269/) · [LiveCodeBench](https://github.com/janhq/LiveCodeBench) · [LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset)

## 技术栈（计划）

Python 3.10+，本地 Qwen-Coder（Ollama / vLLM，OpenAI 兼容接口），不依赖 LangChain 等框架。

## 许可证

待定。

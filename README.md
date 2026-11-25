# RL_final_work · 执行反馈驱动的代码修复强化学习

强化学习课程设计。用**隐藏测试作为可验证奖励**，对 `Qwen2.5-Coder-7B` 做 LoRA + GRPO/GDPO 训练：模型先"一次生成"，失败后把执行反馈回灌，再"修复一轮"；用 GDPO 解耦多奖励归一化，解决"做对"与"省一轮 / 不超时"之间的尺度冲突。

> 状态：环境层、数据层、评测层已可运行；训练层待 GPU 机器验证。

## 文档

- [课程设计方案 v2](docs/课程设计方案.md) — 问题定义、方法（GRPO/GDPO）、奖励设计、数据、实验与五周计划
- 附录 A 保留 v1 的"编排层 RL"方案：作为降级预案与对照基线

## 快速开始

环境层与数据层**零依赖**（只用标准库），可以在无 GPU 的机器上跑：

```bash
python -m pip install -e ".[dev]"     # 只装 pytest
pytest                                # 41 项：判定 / 沙箱 / 清洗 / 切分 / 指标

python scripts/fetch_data.py --limit 400 --offset 0 --out data/cache/old.jsonl
python scripts/fetch_data.py --limit 200 --offset 2441 --out data/cache/new.jsonl
# 合并成一个文件后：
python scripts/prepare_data.py --raw data/cache/leetcode_raw.jsonl --out data/processed
python scripts/selftest.py --split test --n 8       # 参考解必须 AC、坏解必须失败
python -m rl_algo.train.grpo_single_turn --data data/processed/train.jsonl --dry-run
```

**先测基线（最重要的一步）**——判断小模型到底能不能学：

```bash
# 采样口径：pass@1 / pass@4 / pass@8，同时产出难度校准用的 pass_rates.json
python -m rl_algo.eval.baseline --data data/processed/test.jsonl \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct --samples 8 --out results/base-1.5b-test

# 贪心口径：模型"第一反应"的通过率，与采样口径分开报
python -m rl_algo.eval.baseline --data data/processed/test.jsonl \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct --samples 1 --temperature 0 --out results/base-1.5b-greedy

# 训练集上的 pass_rates.json 直接喂给难度校准
python -m rl_algo.eval.baseline --data data/processed/train.jsonl \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct --samples 8 --out results/base-1.5b-train
python scripts/prepare_data.py --raw data/cache/leetcode_raw.jsonl --out data/processed \
    --pass-rates results/base-1.5b-train/pass_rates.json
```

训练（GPU 机器，需 `pip install -e ".[train]"`）：

```bash
python -m rl_algo.train.grpo_single_turn \
    --data data/processed/train.jsonl \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --output outputs/grpo-7b --limit 200 --use-vllm
# 多奖励解耦（GDPO）：加 --gdpo；TRL 版本不支持时会明确告警并回退 GRPO
```

## 核心设计

1. **工具链是 RL 环境，不是被调度的对象。** `rl_algo/env` 提供受限执行、逐 assert 判定与反馈构造，构成可验证奖励的全部来源。
2. **两轮，而不是 8 步 Agent。** `生成 → 修复` 已能验证"模型学会用执行反馈"，多轮 rollout 的基建成本是两轮的 5 倍以上。
3. **GDPO 而不是直接加权求和。** 正确性是 0/1 稀疏信号，通过率是稠密塑形，超时是惩罚项；加权求和后组内标准差被稠密项主导，模型会学"刷部分分"而不是"改对"。
4. **数据侧两道保险。** 时间切分（数据集最新只到 2024-07，需叠加扰动测试）+ 难度校准（只留 base 通过率 `0.1 < p < 0.7` 的题，保证组内有梯度）。

## 目录

```
.
├── docs/课程设计方案.md
├── rl_algo/
│   ├── env/          # sandbox.py / judge.py / prompts.py
│   ├── data/         # fetch.py / prepare.py
│   ├── eval/         # metrics.py / baseline.py
│   └── train/        # rewards.py / grpo_single_turn.py
├── scripts/          # fetch_data / prepare_data / selftest
└── tests/            # 41 项离线单测
```

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

## 相关方法

[GDPO](https://github.com/NVlabs/GDPO) · [GD²PO](https://aclanthology.org/2026.findings-emnlp.1070/) · [TRL](https://github.com/huggingface/trl) · [AlphaCodium](https://github.com/Codium-ai/AlphaCodium) · [LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset)

## 许可证

待定。

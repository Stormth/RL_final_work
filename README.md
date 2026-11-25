# RL_final_work · 执行反馈驱动的代码修复强化学习

强化学习课程设计。用**隐藏测试作为可验证奖励**，对代码模型做 LoRA + GRPO 训练：先"一次生成"，失败后把执行反馈回灌，再"修复一轮"。核心对照是 **RL 相对 SFT / 采样基线是否真的有用**。

> 状态：环境层、数据层、评测层已可运行；训练层待 GPU 机器验证。

## 文档

- [课程设计方案 v2](docs/课程设计方案.md) — 问题定义、方法、奖励设计、数据、实验与计划
- 附录 A 保留了 v1 的"编排层 RL"方案：作为降级预案与对照基线

## 快速开始（无 GPU 也能跑）

```bash
python -m pip install -e ".[dev]"
pytest                                    # 41 项：判定 / 沙箱 / 清洗 / 切分 / 指标
python scripts/selftest.py --split test --n 8      # 参考解必须 AC、坏解必须失败
python -m rl_algo.train.grpo_single_turn --data data/processed/train.jsonl --dry-run
```

## 在 Linux 集群上跑

### 0. 先想好用哪张卡

| 卡 | 显存 | 用途 | 限制 |
|---|---|---|---|
| RTX 4090 | 24 GB | 1.5B / 3B 的基线与冒烟 | 7B 放不下，别硬试 |
| A6000 | 48 GB | 7B LoRA 正式训练 | vLLM colocate 会吃紧，先用 HF 后端 |

**先在 A6000 上跑流程，再在 4090 上跑快的实验**——别反过来，24GB 上踩的坑在 48GB 上会重复一遍。

### 1. 环境

```bash
git clone git@github.com:Stormth/RL_final_work.git
cd RL_final_work
conda create -n rl-algo python=3.11 -y && conda activate rl-algo
pip install -e ".[dev]"          # 核心零依赖，先装这个
python scripts/env_check.py      # 看 CUDA / 显存 / 沙箱是否正常
```

`env_check.py` 会打印卡型并按显存给出建议；**沙箱自检不过就不要往下走**，否则所有奖励都是 0。

### 2. 数据

```bash
python scripts/fetch_all.py      # 拉取 + 合并 + 清洗 + 切分，一步到位
```

集群 GPU 节点常常没有外网，两种办法：

```bash
# 办法 A：在登录节点跑（登录节点通常能联网）
python scripts/fetch_all.py

# 办法 B：本机下载后传过去，集群上离线合并
scp -r data/cache 用户@集群:~/RL_final_work/data/
python scripts/fetch_all.py --offline
```

预期输出：`600 → 424 可用`，train 277 / test 147。

### 2.5 模型权重（无外网时的第二个拦路虎）

```bash
export HF_ENDPOINT=https://hf-mirror.com      # 先试镜像
# 镜像也不通：本机下载后 scp
# 本机：huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct --local-dir models/qwen1.5b
# 集群：--model ./models/qwen1.5b
```

模型体积：1.5B 约 3GB，7B 约 15GB。先确认 `~/.cache` 或 `models/` 所在分区有空间。

### 3. 基线评测（**先做这个，再决定要不要训练**）

```bash
# 采样口径：pass@1 / pass@4 / pass@8
python -m rl_algo.eval.baseline --data data/processed/test.jsonl \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct --samples 8 --out results/base-1.5b-test

# 贪心口径
python -m rl_algo.eval.baseline --data data/processed/test.jsonl \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct --samples 1 --temperature 0 \
    --out results/base-1.5b-greedy

# 训练集：产出难度校准用的 pass_rates.json
python -m rl_algo.eval.baseline --data data/processed/train.jsonl \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct --samples 8 --out results/base-1.5b-train
```

看 `results/base-1.5b-test/summary.md` 里的 `pass@1` / `pass@8` / `format_rate`，按里面的"怎么解读"表决定下一步。

### 4. 训练（A6000）

```bash
pip install -e ".[train]"
python -m rl_algo.train.grpo_single_turn \
    --data data/processed/train.jsonl \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --output outputs/grpo-7b --limit 150 --group-size 4 \
    --rewards correct efficiency
```

### 5. Slurm

```bash
sbatch scripts/job.sbatch python -m rl_algo.eval.baseline \
    --data data/processed/test.jsonl --out results/base-1.5b-test
sbatch scripts/job.sbatch python -m rl_algo.train.grpo_single_turn \
    --data data/processed/train.jsonl --output outputs/grpo-7b --limit 150
```

模板里只需按集群改 `--partition` / `--account` 和 conda 路径。

## 核心设计

1. **工具链是 RL 环境。** `rl_algo/env` 提供受限执行、逐 assert 判定与反馈构造，构成可验证奖励的全部来源。
2. **两轮，而不是 8 步 Agent。** 多轮 rollout 的基建成本是两轮的 5 倍以上。
3. **奖励多维。** 正确性（0/1）、通过率（稠密塑形）、超时惩罚；归一化方式本身就是一个可对照的变量。
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
├── scripts/          # fetch_all / env_check / selftest / job.sbatch
└── tests/            # 41 项离线单测
```

## 计划与 Gate

| 周 | 目标 | Gate |
|---|---|---|
| W1 | 数据清洗/难度校准、沙箱与判定；1.5B 单轮 GRPO 冒烟 | G1：reward 曲线能涨 |
| W2 | 7B 单轮 GRPO + SFT 基线 | **G2：pass@1 ≥ base +5pt** |
| W3 | 两轮 rollout，GRPO-2turn | G3：修复轮 ≥ +3pt |
| W4 | 多奖励对照与消融 | G4：出主结果表 |
| W5 | 一键复现、报告、答辩 | — |

**G2 不过就降级**：切回 v1 的编排层 Q-learning（环境已就绪），训练线作为探索性实验写入报告。

## 相关方法

[GDPO](https://github.com/NVlabs/GDPO) · [GD²PO](https://aclanthology.org/2026.findings-emnlp.1070/) · [TRL](https://github.com/huggingface/trl) · [AlphaCodium](https://github.com/Codium-ai/AlphaCodium) · [LeetCodeDataset](https://huggingface.co/datasets/newfacade/LeetCodeDataset)

## 许可证

待定。

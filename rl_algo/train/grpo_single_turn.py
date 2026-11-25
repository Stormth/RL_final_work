"""单轮 GRPO 训练入口（Phase 0 / G1 冒烟）。

在 GPU 机器上跑：

    python -m rl_algo.train.grpo_single_turn \
        --data data/processed/train.jsonl \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --output outputs/grpo-7b --limit 200

先在本地验证管线（不需要 GPU、不需要 trl）：

    python -m rl_algo.train.grpo_single_turn --data data/processed/train.jsonl --dry-run

版本兼容
--------
TRL 的 ``GRPOConfig`` 字段与 ``GRPOTrainer`` 参数在不同版本间变化频繁，
这里用 ``inspect.signature`` 过滤后再传，缺字段时打印警告而不是崩掉。
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from rl_algo.data.fetch import load_jsonl
from rl_algo.train.rewards import build_suite


def _load_records(path: str) -> list[dict[str, Any]]:
    records = load_jsonl(path)
    if not records:
        raise SystemExit(f"数据集为空: {path}")
    return records


def _filtered_kwargs(target: Any, desired: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    try:
        allowed = set(inspect.signature(target).parameters)
    except (TypeError, ValueError):  # pragma: no cover
        return desired, []
    kept = {key: value for key, value in desired.items() if key in allowed}
    dropped = sorted(set(desired) - set(kept))
    return kept, dropped


def dry_run(args: argparse.Namespace) -> int:
    records = _load_records(args.data)[: args.limit or 3]
    suite = build_suite(records, workers=2, case_timeout_s=1.0, timeout_s=8.0)

    print(f"题目数: {len(records)}")
    sample = records[0]
    print("\n--- prompt 预览（前 600 字）---")
    print(json.dumps(sample["prompt"], ensure_ascii=False)[:600])

    completions = [f"```python\n{r['completion']}```" for r in records]
    task_ids = [r["task_id"] for r in records]
    kwargs = {"task_id": task_ids}
    prompts = [r["prompt"] for r in records]

    results = suite._judge_batch(task_ids, completions)  # noqa: SLF001 - 自检用
    print("\n--- 参考解判定 ---")
    for record, result in zip(records, results):
        print(f"{record['task_id'][:32]:32s} {result['verdict']:12s} {result['passed']}/{result['total']}")

    rewards = {
        "correct": suite.correct(prompts, completions, **kwargs),
        "pass_ratio": suite.pass_ratio(prompts, completions, **kwargs),
        "efficiency": suite.efficiency(prompts, completions, **kwargs),
    }
    print("\n--- 奖励（参考解应当全部 correct=1）---")
    print(json.dumps({k: v[:5] for k, v in rewards.items()}, ensure_ascii=False))

    if not all(value == 1.0 for value in rewards["correct"]):
        print("\n[警告] 参考解没有全部通过：检查数据清洗或判定逻辑。", file=sys.stderr)
        return 1
    print("\n[dry-run] 管线正常。")
    return 0


def train(args: argparse.Namespace) -> int:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover
        print(f"缺少训练依赖: {exc}\n请先 pip install -e '.[train]'", file=sys.stderr)
        return 2

    records = _load_records(args.data)
    if args.limit:
        records = records[: args.limit]
    suite = build_suite(records, workers=args.workers)

    dataset = Dataset.from_list(
        [{"prompt": record["prompt"], "task_id": record["task_id"]} for record in records]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    desired = {
        "output_dir": args.output,
        "learning_rate": args.lr,
        "num_generations": args.group_size,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_steps": args.max_steps,
        "logging_steps": 1,
        "save_steps": args.save_steps,
        "temperature": args.temperature,
        "beta": args.kl_beta,
        "use_vllm": args.use_vllm,
        "vllm_mode": "colocate" if args.use_vllm else None,
        "gradient_checkpointing": True,
        "bf16": True,
        "report_to": "none",
        "seed": args.seed,
    }
    config_params = set(inspect.signature(GRPOConfig).parameters)
    trainer_params = set(inspect.signature(GRPOTrainer.__init__).parameters)

    cfg_kwargs = {key: value for key, value in desired.items() if value is not None and key in config_params}
    for key in sorted(set(desired) - set(cfg_kwargs)):
        if desired[key] is not None:
            print(f"[warn] 当前 TRL 版本不支持 GRPOConfig.{key}，已忽略", file=sys.stderr)

    reward_funcs = suite.funcs(tuple(args.rewards))
    # TRL 0.14+ 把 reward_funcs 挪到了 Trainer；更早的版本在 Config 上
    if "reward_funcs" in config_params:
        cfg_kwargs["reward_funcs"] = reward_funcs
        trainer_reward_kwargs: dict[str, Any] = {}
    else:
        trainer_reward_kwargs = {"reward_funcs": reward_funcs}
    if args.gdpo:
        if "apply_gdpo" in set(inspect.signature(GRPOConfig).parameters):
            cfg_kwargs["apply_gdpo"] = True
        else:
            print("[warn] 当前 TRL 没有 apply_gdpo，回退为 GRPO", file=sys.stderr)

    config = GRPOConfig(**cfg_kwargs)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        **trainer_reward_kwargs,
        "args": config,
        "train_dataset": dataset,
        "peft_config": lora,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:  # pragma: no cover - 旧版 TRL
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"已保存到 {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="单轮 GRPO/GDPO 训练")
    parser.add_argument("--data", required=True, help="data/processed/train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--output", default="outputs/grpo")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--kl-beta", type=float, default=0.001)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8, help="判题并发数")
    parser.add_argument("--rewards", nargs="+", default=["correct", "pass_ratio", "efficiency"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-vllm", action="store_true", help="用 vLLM 生成（显存充足时更快）")
    parser.add_argument("--gdpo", action="store_true", help="多奖励解耦归一化（需 TRL 支持）")
    parser.add_argument("--dry-run", action="store_true", help="只验证管线，不训练")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data and not Path(args.data).exists() and args.dry_run:
        print(f"找不到数据集: {args.data}", file=sys.stderr)
        return 2
    return dry_run(args) if args.dry_run else train(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

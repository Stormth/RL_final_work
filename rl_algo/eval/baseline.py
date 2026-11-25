"""零样本基线评测：小模型裸跑，不加 SFT、不加 RL、不加工具。

为什么先跑这个
--------------
RL 只能放大"偶尔能做对"的行为。先测出 base 的 ``pass@1`` 与 ``pass@8``，
才知道后面该不该上 RL、该用多大的题：

    python -m rl_algo.eval.baseline --data data/processed/test.jsonl \
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct --samples 8 --out results/base-1.5b

    # 贪心解码（"模型第一反应"的通过率，与采样口径分开报）
    python -m rl_algo.eval.baseline --data ... --samples 1 --temperature 0

``--backend fake`` 不需要 GPU/模型，用来验证整条管线（本地开发用）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rl_algo.data.fetch import load_jsonl
from rl_algo.env.judge import evaluate
from rl_algo.env.prompts import extract_code
from rl_algo.eval.metrics import summarize

DECISION_GUIDE = """
## 怎么解读这些数字

| 观察 | 含义 | 下一步 |
|---|---|---|
| `pass@8 < 0.15` | 模型基本不会，RL 没有梯度可学 | 换更简单的题（Easy 前 60 题）或升到 3B；不要再调学习率 |
| `pass@1 < 0.05` 且 `pass@8 > 0.3` | 有能力但不稳定——**最适合 RL 的区间** | SFT 冷启动 → GRPO |
| `pass@1 > 0.6` | 题太简单，RL 收益有限 | 提高难度，或改用效率/多解作为奖励 |
| `format_rate < 0.9` | 输出经常没有代码块 | 先 SFT 冷启动修格式，否则奖励全是 0 |
| `RE` 占比高 | 模型爱写出跑不起来的代码 | 检查是否为签名/导入问题；可作为 RE 惩罚项 |
"""


def _judge_all(
    records: list[dict[str, Any]],
    generated: list[list[str]],
    *,
    workers: int,
    case_timeout_s: float,
    timeout_s: float,
) -> list[dict[str, Any]]:
    jobs: list[tuple[int, int]] = []
    for problem_idx, samples in enumerate(generated):
        for sample_idx in range(len(samples)):
            jobs.append((problem_idx, sample_idx))

    def work(job: tuple[int, int]) -> dict[str, Any]:
        problem_idx, sample_idx = job
        record = records[problem_idx]
        text = generated[problem_idx][sample_idx]
        code = extract_code(text)
        started = time.perf_counter()
        if not code:
            result = {"verdict": "EMPTY", "passed": 0, "total": 0, "pass_ratio": 0.0}
        else:
            result = evaluate(record, code, case_timeout_s=case_timeout_s, timeout_s=timeout_s)
        return {
            "task_id": record["task_id"],
            "difficulty": record.get("difficulty"),
            "sample_idx": sample_idx,
            "verdict": result["verdict"],
            "passed": result["passed"],
            "total": result["total"],
            "pass_ratio": result["pass_ratio"],
            "code_len": len(code),
            "tokens": len(text.split()),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(work, jobs))


def _generate_fake(records: list[dict[str, Any]], n: int) -> list[list[str]]:
    """管线自检用：第 0 条给参考解，其余给坏解。"""
    broken = "```python\nclass Solution:\n    pass\n```"
    output: list[list[str]] = []
    for record in records:
        samples = [f"```python\n{record.get('completion') or ''}```"]
        samples.extend([broken] * (n - 1))
        output.append(samples[:n])
    return output


def _generate_hf(
    records: list[dict[str, Any]],
    n: int,
    args: argparse.Namespace,
) -> list[list[str]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"缺少依赖: {exc}\n请先 pip install -e '.[train]'")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    texts = [
        tokenizer.apply_chat_template(
            record["prompt"], tokenize=False, add_generation_prompt=True
        )
        for record in records
    ]
    do_sample = args.temperature > 0
    n_eff = n if do_sample else 1
    if not do_sample and n > 1:
        print("[warn] temperature=0 时 num_return_sequences 无意义，已按 1 条采样")

    generated: list[list[str]] = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_tokens,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=max(args.temperature, 1e-5),
                top_p=0.95,
                num_return_sequences=n_eff,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = encoded["input_ids"].shape[1]
        for idx in range(len(batch)):
            rows = output[idx * n_eff : (idx + 1) * n_eff]
            generated.append(
                [tokenizer.decode(row[prompt_len:], skip_special_tokens=True) for row in rows]
            )
        print(f"  已生成 {min(start + args.batch_size, len(texts))}/{len(texts)} 题", flush=True)
    return generated


def _generate_vllm(
    records: list[dict[str, Any]],
    n: int,
    args: argparse.Namespace,
) -> list[list[str]]:
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"缺少 vLLM: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_prompt_tokens + args.max_new_tokens)
    params = SamplingParams(
        n=n if args.temperature > 0 else 1,
        temperature=max(args.temperature, 1e-5),
        top_p=0.95,
        max_tokens=args.max_new_tokens,
    )
    texts = [
        tokenizer.apply_chat_template(record["prompt"], tokenize=False, add_generation_prompt=True)
        for record in records
    ]
    outputs = llm.generate(texts, params)
    return [[completion.text for completion in item.outputs] for item in outputs]


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_path = Path(args.data)
    if not data_path.exists():
        print(
            f"[错误] 找不到数据文件: {data_path}\n"
            "先准备数据（二选一）：\n"
            "  1) 能联网：python scripts/fetch_all.py\n"
            "  2) 无外网：把本机打包的 data/cache/*.jsonl 拷到 data/cache/，再 python scripts/fetch_all.py --offline\n"
            f"完成后应存在: {data_path}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    records = load_jsonl(args.data)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"数据集为空: {args.data}")
    print(f"题目数 {len(records)}，每题采样 {args.samples} 条，temperature={args.temperature}")

    started = time.perf_counter()
    if args.backend == "fake":
        generated = _generate_fake(records, args.samples)
    elif args.backend == "vllm":
        generated = _generate_vllm(records, args.samples, args)
    else:
        generated = _generate_hf(records, args.samples, args)
    elapsed_s = time.perf_counter() - started
    print(f"生成完成，用时 {elapsed_s:.1f}s")

    samples = _judge_all(
        records,
        generated,
        workers=args.workers,
        case_timeout_s=args.case_timeout,
        timeout_s=args.timeout,
    )
    summary = summarize(samples)
    summary.update(
        {
            "model": args.model if args.backend != "fake" else "fake",
            "backend": args.backend,
            "data": str(args.data),
            "temperature": args.temperature,
            "generation_seconds": round(elapsed_s, 1),
        }
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rates = {}
    for row in samples:
        rates.setdefault(row["task_id"], []).append(row["verdict"] == "AC")
    (out / "pass_rates.json").write_text(
        json.dumps({key: sum(values) / len(values) for key, values in rates.items()}, indent=2),
        encoding="utf-8",
    )
    (out / "summary.md").write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def _render_markdown(summary: dict[str, Any]) -> str:
    key_metrics = ["pass@1", "pass@2", "pass@4", "pass@8"]
    rows = "\n".join(
        f"| `{key}` | {summary[key]:.3f} |" for key in key_metrics if key in summary
    )
    difficulties = "\n".join(
        f"| {name} | {value['problems']} | {value['pass@1']:.3f} |"
        for name, value in summary.get("by_difficulty", {}).items()
    )
    verdicts = ", ".join(f"{key}={value}" for key, value in summary["verdicts"].items())
    return f"""# 零样本基线：{summary['model']}

- 数据：`{summary['data']}`
- 题目数：{summary['problems']}；采样数：{summary['samples']}；每题 {summary['samples_per_problem']} 条
- temperature：{summary['temperature']}；生成耗时：{summary['generation_seconds']}s
- 平均生成 token 数：{summary['avg_tokens']}；平均代码长度：{summary['avg_code_len']} 字符

## 主指标

| 指标 | 值 |
|---|---|
{rows}
| 代码提取率 | {summary['format_rate']:.3f} |
| 空输出率 | {summary['empty_rate']:.3f} |

## 判定分布

{verdicts}

## 分难度

| 难度 | 题数 | pass@1 |
|---|---|---|
{difficulties}
{DECISION_GUIDE}"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="零样本代码基线评测")
    parser.add_argument("--data", default="data/processed/test.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--backend", choices=["hf", "vllm", "fake"], default="hf")
    parser.add_argument("--out", default="results/base")
    parser.add_argument("--samples", type=int, default=8, help="每题采样条数（pass@k 用）")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--case-timeout", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=12.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(
        json.dumps(
            {key: summary[key] for key in ("problems", "samples", "pass@1", "pass@8", "format_rate")},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n结果已写入 {args.out}/（summary.md / summary.json / samples.jsonl / pass_rates.json）")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

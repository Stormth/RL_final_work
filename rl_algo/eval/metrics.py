"""评测指标（纯函数，便于单测）。

pass@k 用 Chen et al. 2021 的无偏估计：

    pass@k = 1 − C(n−c, k) / C(n, k)

其中 n 为每题采样数，c 为其中通过的条数。直接数"k 条里至少一条过"会高估，
样本少时尤其明显——课程报告里被追问"你的 pass@8 怎么算的"时，这个公式就是答案。
"""

from __future__ import annotations

from math import comb
from typing import Any, Iterable


def pass_at_k(n: int, c: int, k: int) -> float:
    """无偏 pass@k。n: 采样数；c: 通过数；k: 取的条数。"""
    if k <= 0 or n <= 0:
        raise ValueError("k 与 n 必须为正")
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _group(samples: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(sample["task_id"], []).append(sample)
    return grouped


def summarize(samples: list[dict[str, Any]], ks: tuple[int, ...] = (1, 2, 4, 8)) -> dict[str, Any]:
    """把逐条采样结果汇总成报告需要的数字。"""
    if not samples:
        raise ValueError("没有采样结果")

    grouped = _group(samples)
    n_problems = len(grouped)

    pass_at = {f"pass@{k}": 0.0 for k in ks}
    solve_rate_sum = 0.0
    for rows in grouped.values():
        n = len(rows)
        c = sum(1 for row in rows if row["verdict"] == "AC")
        solve_rate_sum += c / n
        for k in ks:
            pass_at[f"pass@{k}"] += pass_at_k(n, c, k)
    for key in pass_at:
        pass_at[key] /= n_problems

    total = len(samples)
    verdicts: dict[str, int] = {}
    for sample in samples:
        verdicts[sample["verdict"]] = verdicts.get(sample["verdict"], 0) + 1

    by_difficulty: dict[str, dict[str, Any]] = {}
    for difficulty in sorted({str(sample.get("difficulty")) for sample in samples}):
        rows = [s for s in samples if str(s.get("difficulty")) == difficulty]
        if not rows:
            continue
        subset = _group(rows)
        by_difficulty[difficulty] = {
            "problems": len(subset),
            "pass@1": sum(
                sum(1 for r in rs if r["verdict"] == "AC") / len(rs) for rs in subset.values()
            )
            / len(subset),
            "samples": len(rows),
        }

    latencies = [float(s.get("latency_ms") or 0) for s in samples]
    tokens = [int(s.get("tokens") or 0) for s in samples]
    code_lengths = [int(s.get("code_len") or 0) for s in samples]

    return {
        "problems": n_problems,
        "samples": total,
        "samples_per_problem": round(total / n_problems, 2),
        "solve_rate": round(solve_rate_sum / n_problems, 4),
        **{key: round(value, 4) for key, value in pass_at.items()},
        "format_rate": round(sum(1 for length in code_lengths if length > 0) / total, 4),
        "empty_rate": round(sum(1 for length in code_lengths if length == 0) / total, 4),
        "verdicts": dict(sorted(verdicts.items(), key=lambda item: -item[1])),
        "verdict_rate": {key: round(value / total, 4) for key, value in verdicts.items()},
        "by_difficulty": by_difficulty,
        "avg_latency_ms": round(sum(latencies) / total, 1),
        "avg_tokens": round(sum(tokens) / total, 1),
        "avg_code_len": round(sum(code_lengths) / total, 1),
    }

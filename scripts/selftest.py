#!/usr/bin/env python3
"""端到端自检：用真实题库验证「参考解必须 AC、坏解必须失败」。

    python scripts/selftest.py --split test --n 8

开发机上跑这个，能在不碰 GPU 的情况下确认判定链路是对的。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_algo.data.fetch import load_jsonl  # noqa: E402
from rl_algo.env.judge import evaluate  # noqa: E402
from rl_algo.env.prompts import format_feedback  # noqa: E402

BROKEN = "class Solution:\n    pass\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="判定链路端到端自检")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args(argv)

    path = Path(args.data) / f"{args.split}.jsonl"
    if not path.exists():
        print(f"找不到 {path}，先跑 scripts/prepare_data.py", file=sys.stderr)
        return 2

    records = load_jsonl(path)[: args.n]
    print(f"{'task_id':34s} {'参考解':10s} {'通过率':8s} {'坏解':8s} {'耗时(ms)':>8s}")
    print("-" * 78)

    reference_ok = 0
    broken_rejected = 0
    for record in records:
        good = evaluate(record, record["completion"], case_timeout_s=1.0, timeout_s=8.0)
        bad = evaluate(record, BROKEN, case_timeout_s=1.0, timeout_s=8.0)
        reference_ok += good["verdict"] == "AC"
        broken_rejected += bad["verdict"] != "AC"
        ratio = f"{good['passed']}/{good['total']}"
        print(
            f"{record['task_id'][:34]:34s} {good['verdict']:10s} {ratio:8s} "
            f"{bad['verdict']:8s} {good['elapsed_ms']:8.1f}"
        )

    if records:
        sample = evaluate(records[0], BROKEN, case_timeout_s=1.0, timeout_s=8.0)
        print("\n--- 反馈样例（喂给模型的就长这样）---")
        print(format_feedback(sample))

    total = len(records)
    print(f"\n参考解 AC: {reference_ok}/{total}；坏解被拒: {broken_rejected}/{total}")
    if reference_ok != total:
        print("[失败] 有参考解没通过，检查清洗与判定。", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "n": total}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

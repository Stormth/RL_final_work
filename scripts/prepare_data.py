#!/usr/bin/env python3
"""清洗原始题库，产出 train.jsonl / test.jsonl / stats.json。

    python scripts/prepare_data.py --raw data/cache/leetcode_raw.jsonl --out data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_algo.data.prepare import build_datasets  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="清洗并切分 LeetCodeDataset")
    parser.add_argument("--raw", default="data/cache/leetcode_raw.jsonl")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--test-after", default="2025-01-01", help="晚于此日期的题进测试集")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--no-reference-check", action="store_true", help="跳过参考解自检（不建议）")
    parser.add_argument("--pass-rates", default=None, help="难度校准用的 {task_id: p} JSON")
    parser.add_argument("--low", type=float, default=0.1)
    parser.add_argument("--high", type=float, default=0.7)
    args = parser.parse_args(argv)

    pass_rates = None
    if args.pass_rates:
        pass_rates = json.loads(Path(args.pass_rates).read_text(encoding="utf-8"))

    stats = build_datasets(
        args.raw,
        args.out,
        check_reference=not args.no_reference_check,
        test_after=args.test_after,
        max_train=args.max_train,
        pass_rates=pass_rates,
        low=args.low,
        high=args.high,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

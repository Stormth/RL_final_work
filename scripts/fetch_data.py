#!/usr/bin/env python3
"""拉取题库原始数据。

    python scripts/fetch_data.py --limit 400 --offset 0    --out data/cache/old.jsonl
    python scripts/fetch_data.py --limit 200 --offset 2441 --out data/cache/new.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_algo.data.fetch import fetch_rows, save_jsonl, trim_columns  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拉取 LeetCodeDataset")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    rows = [trim_columns(row) for row in fetch_rows(args.limit, split=args.split, start=args.offset)]
    count = save_jsonl(rows, args.out)
    print(f"已保存 {count} 条 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

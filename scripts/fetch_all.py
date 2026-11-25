#!/usr/bin/env python3
"""一条命令完成：拉取 → 合并 → 清洗 → 切分。

    python scripts/fetch_all.py                 # 下载（需能访问 huggingface）
    python scripts/fetch_all.py --offline       # 只用 data/cache/*.jsonl（集群无外网时）

为什么分两段拉
--------------
LeetCodeDataset 按题号排序，前 400 条全是 2015 年的老题；测试集需要 2024 年的题，
只能从尾部（offset≈2441）单独拉。合并后按 ``estimated_date`` 做时间切分。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_algo.data.fetch import fetch_rows, save_jsonl, trim_columns  # noqa: E402
from rl_algo.data.prepare import build_datasets  # noqa: E402

DEFAULT_OLD = (400, 0)
DEFAULT_NEW = (200, 2441)


def _merge(paths: list[Path], out_path: Path) -> int:
    rows: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            print(f"[warn] 缺少 {path}，跳过", file=sys.stderr)
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            if task_id in seen:
                continue
            seen.add(task_id)
            rows.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, out_path)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拉取题库并生成 train/test")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--raw", default=None, help="已有的合并结果，存在则跳过下载")
    parser.add_argument("--offline", action="store_true", help="不联网，只用已有缓存")
    parser.add_argument("--old-limit", type=int, default=DEFAULT_OLD[0])
    parser.add_argument("--new-limit", type=int, default=DEFAULT_NEW[0])
    parser.add_argument("--new-offset", type=int, default=DEFAULT_NEW[1])
    parser.add_argument("--test-after", default="2024-01-01")
    parser.add_argument("--no-reference-check", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--pass-rates", default=None)
    args = parser.parse_args(argv)

    cache = Path(args.cache)
    raw_path = Path(args.raw) if args.raw else cache / "leetcode_raw.jsonl"
    old_path, new_path = cache / "old.jsonl", cache / "new.jsonl"

    if not raw_path.exists():
        if args.offline:
            if not (old_path.exists() or new_path.exists()):
                print(f"[错误] --offline 但 {cache} 下没有 old.jsonl / new.jsonl", file=sys.stderr)
                return 2
        else:
            print(f"拉取老题：limit={args.old_limit}")
            save_jsonl(
                [trim_columns(row) for row in fetch_rows(args.old_limit, start=0)], old_path
            )
            print(f"拉取新题：limit={args.new_limit} offset={args.new_offset}")
            save_jsonl(
                [
                    trim_columns(row)
                    for row in fetch_rows(args.new_limit, start=args.new_offset)
                ],
                new_path,
            )
        merged = _merge([old_path, new_path], raw_path)
        print(f"合并 {merged} 条 → {raw_path}")
    else:
        print(f"复用已有 {raw_path}")

    pass_rates = None
    if args.pass_rates:
        pass_rates = json.loads(Path(args.pass_rates).read_text(encoding="utf-8"))

    stats = build_datasets(
        raw_path,
        args.out,
        check_reference=not args.no_reference_check,
        test_after=args.test_after,
        max_train=args.max_train,
        pass_rates=pass_rates,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""从 HuggingFace datasets-server 分页拉取 LeetCodeDataset。

为什么不直接 ``datasets.load_dataset``
--------------------------------------
本地环境里 ``datasets`` 是 2.12（旧版），且训练机与开发机版本不一致；
datasets-server 的 HTTP 接口更稳定、零依赖，也便于把原始数据缓存成 JSONL 复现。
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

DATASET = "newfacade/LeetCodeDataset"
ENDPOINT = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
USER_AGENT = "rl-algo/0.1 (+course-project)"

# 只保留需要的列，丢掉 query/response（长且冗余）
KEEP_COLUMNS = (
    "task_id",
    "question_id",
    "difficulty",
    "tags",
    "problem_description",
    "starter_code",
    "estimated_date",
    "prompt",
    "completion",
    "entry_point",
    "test",
    "input_output",
)


def _get_json(url: str, *, retries: int = 3, timeout_s: float = 30.0) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - 网络错误类型太多
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"拉取失败: {url} ({last_error})")


def list_splits(dataset: str = DATASET, config: str = "default") -> list[dict[str, Any]]:
    url = (
        "https://datasets-server.huggingface.co/splits?"
        + urllib.parse.urlencode({"dataset": dataset})
    )
    payload = _get_json(url)
    return [s for s in payload.get("splits", []) if s.get("config") == config]


def fetch_rows(
    limit: int,
    *,
    split: str = "train",
    start: int = 0,
    dataset: str = DATASET,
    config: str = "default",
    page_size: int = PAGE_SIZE,
) -> list[dict[str, Any]]:
    """按页拉取，最多 ``limit`` 条；``start`` 用于跳过前 N 条（取尾部的近期题目）。"""
    rows: list[dict[str, Any]] = []
    offset = max(0, start)
    while len(rows) < limit:
        want = min(page_size, limit - len(rows))
        query = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": want}
        )
        payload = _get_json(f"{ENDPOINT}?{query}")
        if payload.get("error"):
            raise RuntimeError(f"datasets-server 返回错误: {payload['error']}")
        batch = payload.get("rows") or []
        if not batch:
            break
        rows.extend(item["row"] for item in batch)
        offset += len(batch)
        print(f"  已拉取 {len(rows)}/{limit}", flush=True)
    return rows[:limit]


def save_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def trim_columns(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in KEEP_COLUMNS if key in row}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="拉取 LeetCodeDataset 并缓存为 JSONL")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--split", default="train")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", default="data/cache/leetcode_raw.jsonl")
    args = parser.parse_args(argv)

    rows = [trim_columns(row) for row in fetch_rows(args.limit, split=args.split, start=args.offset)]
    count = save_jsonl(rows, args.out)
    print(f"已保存 {count} 条 → {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

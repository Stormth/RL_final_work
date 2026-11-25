"""清洗、时间切分、难度校准。

三道过滤缺一不可
----------------
1. **参考解自检**：数据集的测例是自动生成的，存在"参考解自己都过不了"的题。
2. **多解题过滤**：``any order`` / ``any valid`` 类题目用 ``==`` 断言本身就是错的。
3. **时间切分**：训练用老题，评测用较新的题，否则"背题"会被误读成"学会修复"。
   注意 LeetCodeDataset 最新只到 2024-07，所以"晚于截止日期"这条只能部分满足，
   测试集需配合扰动测试（见方案 §4）。
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rl_algo.data.fetch import load_jsonl, save_jsonl
from rl_algo.env.judge import evaluate_reference
from rl_algo.env.prompts import build_turn1

MULTI_ANSWER_MARKERS = (
    "any order",
    "in any order",
    "any valid",
    "multiple valid",
    "return any",
    "any of the valid",
)
REQUIRED_FIELDS = ("task_id", "entry_point", "test", "problem_description", "starter_code", "completion")
DEFAULT_TEST_AFTER = "2024-01-01"

# 下游训练/评测会用到，但原始数据里叫 prompt（代码前置声明），改名避免与聊天提示冲突
_RENAME = {"prompt": "preamble"}
_DROP = ("query", "response")


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """原始行 → 内部结构。"""
    problem: dict[str, Any] = {key: value for key, value in raw.items() if key not in _DROP}
    for old, new in _RENAME.items():
        if old in problem:
            problem[new] = problem.pop(old)
    pairs = problem.get("input_output") or []
    problem["input_output"] = [pair for pair in pairs if isinstance(pair, dict)][:3]
    return problem


def _parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value)
    for candidate in (text, text[:19]):
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def clean_rows(
    rows: list[dict[str, Any]],
    *,
    check_reference: bool = True,
    difficulties: tuple[str, ...] = ("Easy", "Medium"),
    case_timeout_s: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """逐条过滤，返回 (保留的题, 统计报告)。"""
    dropped: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in rows:
        problem = normalize(row)
        task_id = problem.get("task_id")
        if not task_id:
            dropped["missing_task_id"] += 1
            continue
        if task_id in seen:
            dropped["duplicate"] += 1
            continue
        if any(not problem.get(field) for field in REQUIRED_FIELDS):
            dropped["missing_field"] += 1
            continue
        if problem.get("difficulty") not in difficulties:
            dropped["difficulty"] += 1
            continue
        if _parse_date(problem.get("estimated_date")) is None:
            dropped["no_date"] += 1
            continue
        description = (problem.get("problem_description") or "").lower()
        if any(marker in description for marker in MULTI_ANSWER_MARKERS):
            dropped["multi_answer"] += 1
            continue
        if not problem["input_output"]:
            dropped["no_examples"] += 1
            continue
        if check_reference:
            result = evaluate_reference(problem, case_timeout_s=case_timeout_s, timeout_s=8.0)
            if result["verdict"] != "AC":
                dropped[f"reference_{result['verdict'].lower()}"] += 1
                continue
            if result["total"] < 3:
                dropped["too_few_tests"] += 1
                continue

        seen.add(task_id)
        kept.append(problem)

    report = {
        "input": len(rows),
        "kept": len(kept),
        "dropped": dict(dropped.most_common()),
    }
    return kept, report


def temporal_split(
    rows: list[dict[str, Any]], test_after: str = DEFAULT_TEST_AFTER
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按发布时间切分：晚于阈值的进测试集。"""
    cutoff = _parse_date(test_after)
    if cutoff is None:
        raise ValueError(f"无法解析时间阈值: {test_after}")
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for row in rows:
        moment = _parse_date(row.get("estimated_date"))
        (test if moment and moment >= cutoff else train).append(row)
    return train, test


def apply_pass_rates(
    rows: list[dict[str, Any]],
    pass_rates: dict[str, float],
    *,
    low: float = 0.1,
    high: float = 0.7,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """难度校准：只保留 base 模型"有时对、有时错"的题。

    组内全对或全错时 GRPO 的优势恒为 0，梯度为 0——这是训练不动最常见的原因。
    """
    kept: list[dict[str, Any]] = []
    stats = {"too_easy": 0, "too_hard": 0, "no_rate": 0, "kept": 0}
    for row in rows:
        rate = pass_rates.get(row["task_id"])
        if rate is None:
            stats["no_rate"] += 1
            continue
        if rate > high:
            stats["too_easy"] += 1
        elif rate < low:
            stats["too_hard"] += 1
        else:
            kept.append(row)
    stats["kept"] = len(kept)
    return kept, stats


def to_record(problem: dict[str, Any]) -> dict[str, Any]:
    """内部题目 → 训练/评测用的 JSONL 记录。

    注意：``test``（隐藏测试）会被写进记录，但**只允许奖励函数使用**，
    绝不能进入 ``prompt``。这是整个实验有效性的底线。
    """
    return {
        "task_id": problem["task_id"],
        "difficulty": problem.get("difficulty"),
        "tags": problem.get("tags") or [],
        "estimated_date": problem.get("estimated_date"),
        "problem_description": problem.get("problem_description"),
        "starter_code": problem.get("starter_code"),
        "preamble": problem.get("preamble"),
        "completion": problem.get("completion"),
        "entry_point": problem["entry_point"],
        "test": problem["test"],
        "input_output": problem["input_output"],
        "prompt": build_turn1(problem),
    }


def build_datasets(
    raw_path: str | Path,
    out_dir: str | Path,
    *,
    check_reference: bool = True,
    test_after: str = DEFAULT_TEST_AFTER,
    max_train: int | None = None,
    pass_rates: dict[str, float] | None = None,
    low: float = 0.1,
    high: float = 0.7,
) -> dict[str, Any]:
    rows = [normalize(row) for row in load_jsonl(raw_path)]
    kept, clean_report = clean_rows(rows, check_reference=check_reference)
    train, test = temporal_split(kept, test_after=test_after)

    calibration: dict[str, Any] = {}
    if pass_rates:
        train, calibration = apply_pass_rates(train, pass_rates, low=low, high=high)
    if max_train is not None:
        train = train[:max_train]

    out = Path(out_dir)
    save_jsonl([to_record(row) for row in train], out / "train.jsonl")
    save_jsonl([to_record(row) for row in test], out / "test.jsonl")

    stats = {
        "clean": clean_report,
        "calibration": calibration,
        "train": len(train),
        "test": len(test),
        "test_after": test_after,
        "by_difficulty": {
            "train": dict(Counter(row.get("difficulty") for row in train)),
            "test": dict(Counter(row.get("difficulty") for row in test)),
        },
        "date_range": {
            "train": _range(train),
            "test": _range(test),
        },
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def _range(rows: list[dict[str, Any]]) -> list[str]:
    dates = sorted(str(row.get("estimated_date") or "")[:10] for row in rows if row.get("estimated_date"))
    return [dates[0], dates[-1]] if dates else []

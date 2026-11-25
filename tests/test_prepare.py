"""数据清洗与切分的单元测试（全部离线，不依赖网络与模型）。"""

from __future__ import annotations

from rl_algo.data.prepare import (
    apply_pass_rates,
    clean_rows,
    normalize,
    temporal_split,
    to_record,
)


def _row(task_id: str = "t1", date: str = "2023-01-01T00:00:00", **overrides):
    row = {
        "task_id": task_id,
        "entry_point": "Solution().solve",
        "test": "def check(candidate):\n    assert candidate(1) == 1\n",
        "problem_description": "Return the input.",
        "starter_code": "class Solution:\n    def solve(self, x):\n        pass",
        "completion": "class Solution:\n    def solve(self, x):\n        return x\n",
        "prompt": "from typing import *\n",
        "input_output": [{"input": "1", "output": "1"}],
        "difficulty": "Easy",
        "estimated_date": date,
        "query": "q",
        "response": "r",
    }
    row.update(overrides)
    return row


def test_normalize_renames_and_trims():
    problem = normalize(_row())
    assert "preamble" in problem and "prompt" not in problem
    assert "query" not in problem and "response" not in problem
    assert len(problem["input_output"]) == 1


def test_normalize_caps_public_examples():
    pairs = [{"input": str(i), "output": str(i)} for i in range(10)]
    problem = normalize(_row(input_output=pairs))
    assert len(problem["input_output"]) == 3


def test_clean_rows_drops_multi_answer():
    rows = [
        _row("ok"),
        _row("multi", problem_description="Return the answer in any order."),
    ]
    kept, report = clean_rows(rows, check_reference=False)
    assert [row["task_id"] for row in kept] == ["ok"]
    assert report["dropped"].get("multi_answer") == 1


def test_clean_rows_drops_missing_fields():
    rows = [_row("ok"), _row("broken", test="")]
    kept, report = clean_rows(rows, check_reference=False)
    assert [row["task_id"] for row in kept] == ["ok"]
    assert report["dropped"].get("missing_field") == 1


def test_clean_rows_deduplicates():
    kept, report = clean_rows([_row("a"), _row("a")], check_reference=False)
    assert len(kept) == 1
    assert report["dropped"].get("duplicate") == 1


def test_clean_rows_drops_hard_difficulty():
    kept, _ = clean_rows([_row("a", difficulty="Hard")], check_reference=False)
    assert kept == []


def test_temporal_split_boundary():
    rows = [_row("old", date="2024-12-31T00:00:00"), _row("new", date="2025-01-01T00:00:00")]
    train, test = temporal_split(rows, test_after="2025-01-01")
    assert [row["task_id"] for row in train] == ["old"]
    assert [row["task_id"] for row in test] == ["new"]


def test_apply_pass_rates_filters_both_ends():
    rows = [_row("a"), _row("b"), _row("c"), _row("d")]
    rates = {"a": 0.0, "b": 0.3, "c": 0.95, "d": 0.5}
    kept, stats = apply_pass_rates(rows, rates, low=0.1, high=0.7)
    assert [row["task_id"] for row in kept] == ["b", "d"]
    assert stats["too_easy"] == 1 and stats["too_hard"] == 1


def test_to_record_keeps_hidden_tests_out_of_prompt():
    record = to_record(normalize(_row()))
    prompt_text = str(record["prompt"])
    assert "candidate(1) == 1" not in prompt_text
    assert record["test"].startswith("def check")


def test_to_record_prompt_is_chat_messages():
    record = to_record(normalize(_row()))
    assert isinstance(record["prompt"], list)
    assert record["prompt"][0]["role"] == "system"
    assert record["prompt"][1]["role"] == "user"

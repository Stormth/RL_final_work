"""指标计算的单元测试（纯函数，离线）。"""

from __future__ import annotations

import pytest

from rl_algo.eval.metrics import pass_at_k, summarize


def _sample(task_id: str, verdict: str, difficulty: str = "Easy", **extra):
    row = {
        "task_id": task_id,
        "verdict": verdict,
        "difficulty": difficulty,
        "latency_ms": 10.0,
        "tokens": 100,
        "code_len": 50 if verdict != "EMPTY" else 0,
    }
    row.update(extra)
    return row


def test_pass_at_k_matches_unbiased_estimator():
    # n=8, c=1: pass@1 = 1/8；pass@8 = 1
    assert pass_at_k(8, 1, 1) == pytest.approx(0.125)
    assert pass_at_k(8, 1, 8) == 1.0
    # n=8, c=4: pass@1 = 0.5
    assert pass_at_k(8, 4, 1) == pytest.approx(0.5)
    # n=4, c=2, k=2: 1 - C(2,2)/C(4,2) = 1 - 1/6
    assert pass_at_k(4, 2, 2) == pytest.approx(5 / 6)


def test_pass_at_k_clamps_k_to_n():
    assert pass_at_k(4, 0, 8) == 0.0


def test_pass_at_k_rejects_bad_input():
    with pytest.raises(ValueError):
        pass_at_k(0, 0, 1)


def test_summarize_counts_verdicts_and_format():
    samples = [
        _sample("a", "AC"),
        _sample("a", "WA"),
        _sample("b", "RE"),
        _sample("b", "EMPTY"),
    ]
    summary = summarize(samples, ks=(1, 2))
    assert summary["problems"] == 2
    assert summary["samples"] == 4
    assert summary["solve_rate"] == pytest.approx(0.25)  # a: 1/2, b: 0/2 → 0.25
    assert summary["pass@1"] == pytest.approx(0.25)
    assert summary["format_rate"] == pytest.approx(0.75)
    assert summary["empty_rate"] == pytest.approx(0.25)
    assert summary["verdicts"]["AC"] == 1


def test_summarize_groups_by_difficulty():
    samples = [
        _sample("a", "AC", difficulty="Easy"),
        _sample("b", "WA", difficulty="Medium"),
        _sample("c", "WA", difficulty="Medium"),
    ]
    summary = summarize(samples, ks=(1,))
    assert set(summary["by_difficulty"]) == {"Easy", "Medium"}
    assert summary["by_difficulty"]["Easy"]["pass@1"] == pytest.approx(1.0)
    assert summary["by_difficulty"]["Medium"]["pass@1"] == pytest.approx(0.0)

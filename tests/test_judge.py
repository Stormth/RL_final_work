"""判定与沙箱的单元测试（合成题目，不依赖网络与模型）。"""

from __future__ import annotations

import pytest

from rl_algo.env.judge import evaluate, transform_check
from rl_algo.env.sandbox import run_python

PREAMBLE = "from typing import *\n"

TWO_SUM = {
    "task_id": "two-sum",
    "entry_point": "Solution().twoSum",
    "problem_description": "Return indices of the two numbers that add up to target.",
    "starter_code": "class Solution:\n    def twoSum(self, nums, target):\n        pass",
    "preamble": PREAMBLE,
    "test": (
        "def check(candidate):\n"
        "    assert candidate(nums=[2, 7, 11, 15], target=9) == [0, 1]\n"
        "    assert candidate(nums=[3, 2, 4], target=6) == [1, 2]\n"
        "    assert candidate(nums=[3, 3], target=6) == [0, 1]\n"
    ),
    "completion": (
        "class Solution:\n"
        "    def twoSum(self, nums, target):\n"
        "        seen = {}\n"
        "        for i, x in enumerate(nums):\n"
        "            if target - x in seen:\n"
        "                return [seen[target - x], i]\n"
        "            seen[x] = i\n"
    ),
}

WRONG = (
    "class Solution:\n"
    "    def twoSum(self, nums, target):\n"
    "        return [1, 0]\n"
)


def test_reference_is_accepted():
    result = evaluate(TWO_SUM, TWO_SUM["completion"])
    assert result["verdict"] == "AC", result
    assert result["passed"] == result["total"] == 3
    assert result["pass_ratio"] == 1.0


def test_wrong_answer_reports_first_failure():
    result = evaluate(TWO_SUM, WRONG)
    assert result["verdict"] == "WA"
    assert result["passed"] < result["total"]
    first = result["first_fail"]
    assert first is not None
    assert first["lhs"] is not None and first["rhs"] is not None
    assert "[1, 0]" in first["lhs"]


def test_partial_credit_is_counted():
    """部分通过要能被真实计入，而不是"一失败就 0 分"。"""
    problem = {
        "task_id": "square",
        "entry_point": "solve",
        "preamble": "",
        "test": (
            "def check(candidate):\n"
            "    assert candidate(1) == 1\n"
            "    assert candidate(2) == 4\n"
            "    assert candidate(3) == 9\n"
            "    assert candidate(4) == 16\n"
        ),
    }
    result = evaluate(problem, "def solve(x):\n    return x * x if x < 3 else x\n")
    assert result["verdict"] == "WA"
    assert result["passed"] == 2
    assert result["total"] == 4
    assert result["pass_ratio"] == 0.5
    assert result["first_fail"]["lhs"] == "3"
    assert result["first_fail"]["rhs"] == "9"


def test_compile_error():
    result = evaluate(TWO_SUM, "class Solution:\n    def twoSum(self, nums, target)\n        return []\n")
    assert result["verdict"] == "CE"


def test_runtime_error():
    code = (
        "class Solution:\n"
        "    def twoSum(self, nums, target):\n"
        "        return 1 / 0\n"
    )
    result = evaluate(TWO_SUM, code)
    assert result["verdict"] == "RE"


def test_time_limit_per_case():
    code = (
        "class Solution:\n"
        "    def twoSum(self, nums, target):\n"
        "        while True:\n"
        "            pass\n"
    )
    result = evaluate(TWO_SUM, code, case_timeout_s=0.3, timeout_s=8)
    assert result["verdict"] == "TLE", result


def test_assert_inside_loop_is_rewritten():
    problem = {
        "task_id": "loop",
        "entry_point": "solve",
        "preamble": "",
        "test": (
            "def check(candidate):\n"
            "    for i in range(4):\n"
            "        assert candidate(i) == i * i\n"
        ),
    }
    result = evaluate(problem, "def solve(x):\n    return x * x\n")
    assert result["verdict"] == "AC"
    assert result["total"] == 4


def test_non_equality_assert_falls_back_to_truthy():
    problem = {
        "task_id": "truthy",
        "entry_point": "solve",
        "preamble": "",
        "test": "def check(candidate):\n    assert candidate(4)\n    assert not candidate(0)\n",
    }
    result = evaluate(problem, "def solve(x):\n    return x > 0\n")
    assert result["verdict"] == "AC"
    assert result["total"] == 2


def test_no_asserts_is_unsupported():
    problem = {
        "task_id": "empty",
        "entry_point": "solve",
        "preamble": "",
        "test": "def check(candidate):\n    candidate(1)\n",
    }
    result = evaluate(problem, "def solve(x):\n    return x\n")
    assert result["verdict"] == "UNSUPPORTED"


def test_untransformable_test_falls_back_to_coarse():
    problem = {
        "task_id": "coarse",
        "entry_point": "solve",
        "preamble": "",
        "test": "def check(candidate):\n    assert candidate(1) == 1\n",
    }
    original = transform_check(problem["test"])
    assert original is not None
    assert "_rec_eq" in original


def test_sandbox_timeout():
    result = run_python("while True:\n    pass\n", timeout_s=0.5)
    assert result.timed_out is True
    assert result.ok is False


def test_sandbox_normal_exit():
    result = run_python("print('hello')\n")
    assert result.ok is True
    assert "hello" in result.stdout


def test_sandbox_output_is_truncated():
    result = run_python("print('x' * 200000)\n", timeout_s=10)
    assert result.ok is True
    assert len(result.stdout) < 20000


@pytest.mark.parametrize("verdict", ["AC", "WA", "CE", "RE", "TLE"])
def test_feedback_never_leaks_hidden_cases(verdict):
    from rl_algo.env.prompts import format_feedback

    feedback = format_feedback(
        {
            "verdict": verdict,
            "passed": 1,
            "total": 9,
            "error": {"msg": "boom"},
            "first_fail": {"lhs": "[1, 0]", "rhs": "[0, 1]"},
        }
    )
    assert "9" not in feedback or "passed" in feedback

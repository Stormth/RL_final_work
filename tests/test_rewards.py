"""奖励函数的单元测试（离线，用合成题目）。"""

from __future__ import annotations

from rl_algo.train.rewards import build_suite, completion_text

PROBLEM = {
    "task_id": "square",
    "entry_point": "solve",
    "preamble": "",
    "test": (
        "def check(candidate):\n"
        "    assert candidate(1) == 1\n"
        "    assert candidate(2) == 4\n"
        "    assert candidate(3) == 9\n"
    ),
}

RECORD = dict(PROBLEM, prompt=[{"role": "user", "content": "square it"}], completion="")

CORRECT = "```python\ndef solve(x):\n    return x * x\n```"
PARTIAL = "```python\ndef solve(x):\n    return x * x if x < 3 else x\n```"


def _suite():
    return build_suite([RECORD], workers=2, case_timeout_s=1.0, timeout_s=8.0)


def _run(suite, completion: str):
    prompts = [RECORD["prompt"]]
    task_ids = {"task_id": [RECORD["task_id"]]}
    return {
        "correct": suite.correct(prompts, [completion], **task_ids)[0],
        "pass_ratio": suite.pass_ratio(prompts, [completion], **task_ids)[0],
        "efficiency": suite.efficiency(prompts, [completion], **task_ids)[0],
    }


def test_correct_solution_scores_full():
    rewards = _run(_suite(), CORRECT)
    assert rewards == {"correct": 1.0, "pass_ratio": 1.0, "efficiency": 0.0}


def test_partial_solution_gets_dense_credit_only():
    rewards = _run(_suite(), PARTIAL)
    assert rewards["correct"] == 0.0
    assert 0.0 < rewards["pass_ratio"] < 1.0


def test_empty_completion_is_zero():
    rewards = _run(_suite(), "I cannot solve this.")
    assert rewards == {"correct": 0.0, "pass_ratio": 0.0, "efficiency": 0.0}


def test_timeout_is_penalised_by_efficiency_only():
    suite = build_suite([RECORD], workers=1, case_timeout_s=0.3, timeout_s=8.0)
    rewards = _run(suite, "```python\ndef solve(x):\n    while True:\n        pass\n```")
    assert rewards["correct"] == 0.0
    assert rewards["efficiency"] == -1.0


def test_cache_is_reused_across_reward_dims():
    suite = _suite()
    _run(suite, CORRECT)
    _run(suite, CORRECT)
    # 三个维度共享一次判定：两条不同代码 → 至多 2 个缓存项
    assert len(suite.cache._data) <= 2  # noqa: SLF001


def test_unknown_task_id_fails_loudly():
    suite = _suite()
    try:
        suite.correct([RECORD["prompt"]], [CORRECT], task_id=["nope"])
    except KeyError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当抛出 KeyError")


def test_completion_text_handles_chat_messages():
    messages = [{"role": "assistant", "content": CORRECT}]
    assert completion_text(messages) == CORRECT


def test_funcs_order_matches_request():
    suite = _suite()
    names = [func.__name__ for func in suite.funcs(("correct", "efficiency"))]
    assert names == ["correct", "efficiency"]

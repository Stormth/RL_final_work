"""GRPO/GDPO 的奖励函数。

TRL 的约定：``reward_fn(prompts, completions, **kwargs) -> list[float]``，
其中 ``kwargs`` 会带上数据集的其它列（我们靠 ``task_id`` 找回题目与隐藏测试）。

三个维度对应方案 §3.2：

============  ========================================  ==========
维度          含义                                      取值范围
============  ========================================  ==========
correct       隐藏测试全过                               {0, 1}
pass_ratio    通过率（稠密塑形）                          [0, 1]
efficiency    超时惩罚                                  {-1, 0}
============  ========================================  ==========

``pass_ratio`` 是**塑形奖励**，会诱导"刷部分分"（比如故意只过 9/12 就收手），
必须做消融；``efficiency`` 在单轮阶段只惩罚 TLE，多轮阶段的"轮数代价"在 Phase 1 再加。
"""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from rl_algo.env.judge import evaluate
from rl_algo.env.prompts import extract_code

DEFAULT_WORKERS = 8


class JudgeCache:
    """同一批次里三个奖励会各判一次，缓存把沙箱调用降到 1/3。"""

    def __init__(self, max_items: int = 8192) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._max = max_items

    def key(self, task_id: str, code: str) -> str:
        digest = hashlib.sha1(code.encode("utf-8")).hexdigest()[:16]
        return f"{task_id}:{digest}"

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            if len(self._data) >= self._max:
                self._data.clear()
            self._data[key] = value


def completion_text(completion: Any) -> str:
    """兼容两种 completion 形态：纯文本，或 chat messages。"""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion or "")


class RewardSuite:
    """把题目表、判题、缓存包在一起，对外暴露 TRL 可用的奖励函数。"""

    def __init__(
        self,
        problems: dict[str, dict[str, Any]],
        *,
        workers: int = DEFAULT_WORKERS,
        case_timeout_s: float = 2.0,
        timeout_s: float = 12.0,
    ) -> None:
        self.problems = problems
        self.case_timeout_s = case_timeout_s
        self.timeout_s = timeout_s
        self.cache = JudgeCache()
        self._pool = ThreadPoolExecutor(max_workers=workers)

    # ---- 内部 ----
    def _judge_one(self, task_id: str, code: str) -> dict[str, Any]:
        key = self.cache.key(task_id, code)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        problem = self.problems[task_id]
        result = evaluate(
            problem, code, case_timeout_s=self.case_timeout_s, timeout_s=self.timeout_s
        )
        self.cache.put(key, result)
        return result

    def _judge_batch(self, task_ids: list[str], completions: list[Any]) -> list[dict[str, Any]]:
        if len(task_ids) != len(completions):
            raise ValueError(f"task_id 与 completion 数量不一致: {len(task_ids)} != {len(completions)}")
        unknown = [tid for tid in task_ids if tid not in self.problems]
        if unknown:
            raise KeyError(f"未知 task_id（数据集与题库不一致）: {unknown[:3]}")

        def work(pair: tuple[str, Any]) -> dict[str, Any]:
            task_id, completion = pair
            code = extract_code(completion_text(completion))
            if not code:
                return {
                    "task_id": task_id,
                    "verdict": "EMPTY",
                    "passed": 0,
                    "total": 0,
                    "pass_ratio": 0.0,
                    "elapsed_ms": 0.0,
                }
            return self._judge_one(task_id, code)

        return list(self._pool.map(work, zip(task_ids, completions)))

    @staticmethod
    def _task_ids(kwargs: dict[str, Any], n: int) -> list[str]:
        task_ids = kwargs.get("task_id")
        if task_ids is None:
            raise ValueError(
                "奖励函数拿不到 task_id。请确认训练集里保留了 task_id 列，"
                "且 GRPOTrainer 没有丢弃未使用列。"
            )
        return list(task_ids)[:n]

    # ---- 对外 ----
    def correct(self, prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        task_ids = self._task_ids(kwargs, len(completions))
        results = self._judge_batch(task_ids, completions)
        return [1.0 if res["verdict"] == "AC" else 0.0 for res in results]

    def pass_ratio(self, prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        task_ids = self._task_ids(kwargs, len(completions))
        results = self._judge_batch(task_ids, completions)
        return [float(res.get("pass_ratio", 0.0)) for res in results]

    def efficiency(self, prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        task_ids = self._task_ids(kwargs, len(completions))
        results = self._judge_batch(task_ids, completions)
        return [-1.0 if res["verdict"] == "TLE" else 0.0 for res in results]

    def funcs(self, names: tuple[str, ...] = ("correct", "pass_ratio", "efficiency")) -> list[Callable]:
        return [getattr(self, name) for name in names]


def build_suite(
    records: list[dict[str, Any]],
    *,
    workers: int = DEFAULT_WORKERS,
    case_timeout_s: float = 2.0,
    timeout_s: float = 12.0,
) -> RewardSuite:
    return RewardSuite(
        {record["task_id"]: record for record in records},
        workers=workers,
        case_timeout_s=case_timeout_s,
        timeout_s=timeout_s,
    )

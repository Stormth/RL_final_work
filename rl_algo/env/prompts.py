"""提示词与反馈格式。

原则
----
1. **题面用英文**（数据集是英文的，Qwen-Coder 在英文指令下更稳）。
2. **只给一个失败用例**：信息量刚好够定位，又不至于让模型把它当模板背下来。
3. **反馈里绝不出现隐藏测试全集**，也不出现"还有 N 条用例"这类可推断信息。
"""

from __future__ import annotations

import re
from typing import Any

_CODE_FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are an expert Python programmer solving LeetCode-style problems. "
    "Write a complete, correct, efficient solution. "
    "Output ONLY one Python code block. "
    "Do not include tests, prints, explanations, or usage examples."
)

REPAIR_SYSTEM_PROMPT = (
    "You are an expert Python programmer fixing a failing solution. "
    "Given the execution feedback, change the code so that the failing case passes, "
    "without breaking the cases that already pass. "
    "Prefer the smallest change that fixes the root cause. "
    "Output ONLY one Python code block."
)

MAX_REPR = 300


def extract_code(text: str) -> str:
    """从模型输出里取出代码块；取最后一个（模型常在前面先给一段错解）。"""
    blocks = _CODE_FENCE.findall(text or "")
    if blocks:
        return blocks[-1].strip() + "\n"
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    # 没有围栏时，退化为"看起来像代码就整段收下"
    if "def " in cleaned or "class " in cleaned:
        return cleaned + "\n"
    return ""


def _clip(text: str, limit: int = MAX_REPR) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "…"


def format_public_examples(problem: dict[str, Any], limit: int = 3) -> str:
    """公开样例来自 ``input_output``，与隐藏测试分离。"""
    pairs = problem.get("input_output") or []
    lines: list[str] = []
    for pair in pairs[:limit]:
        if isinstance(pair, dict):
            lines.append(f"Input: {pair.get('input', '')}\nOutput: {pair.get('output', '')}")
    return "\n\n".join(lines)


def build_turn1(problem: dict[str, Any]) -> list[dict[str, str]]:
    """首解提示。"""
    parts = [
        "### Problem",
        _clip(problem.get("problem_description", ""), 4000),
        "### Starter code",
        "```python\n" + (problem.get("starter_code") or "").strip() + "\n```",
    ]
    examples = format_public_examples(problem)
    if examples:
        parts += ["### Examples", examples]
    parts.append(
        "### Requirement\n"
        "Implement the solution inside the given class/function signature. "
        "The code will be executed against hidden tests."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def format_feedback(result: dict[str, Any]) -> str:
    """把判定结果转成模型可读的反馈。"""
    verdict = result.get("verdict", "RE")
    passed = result.get("passed", 0)
    total = result.get("total", 0)

    if verdict == "AC":
        return "[Execution result] Accepted (all hidden tests passed)."

    if verdict == "CE":
        error = result.get("error") or {}
        return f"[Execution result] CompileError\n[Message] {_clip(error.get('msg', ''))}"

    if verdict == "TLE":
        return (
            "[Execution result] TimeLimitExceeded\n"
            "[Message] The solution did not finish on some case. "
            "Reconsider the time complexity against the constraints."
        )

    if verdict == "RE":
        error = result.get("error") or {}
        detail = error.get("msg") or error.get("type", "runtime error")
        return f"[Execution result] RuntimeError\n[Message] {_clip(detail)}"

    if verdict == "WA":
        case = result.get("first_fail") or {}
        lines = [f"[Execution result] WrongAnswer ({passed}/{total} hidden tests passed)"]
        if case.get("lhs") is not None or case.get("rhs") is not None:
            lines.append(f"[Expected] {_clip(case.get('rhs'))}")
            lines.append(f"[Actual]   {_clip(case.get('lhs'))}")
        else:
            lines.append("[Message] One assertion failed; check edge cases and boundary values.")
        lines.append("[Hint] The hidden tests are not shown. Fix the root cause, not this case.")
        return "\n".join(lines)

    return f"[Execution result] {verdict} (unusable submission)."


def build_turn2(problem: dict[str, Any], previous_code: str, feedback: str) -> list[dict[str, str]]:
    """修复提示：题面 + 上一版代码 + 反馈。"""
    parts = [
        "### Problem",
        _clip(problem.get("problem_description", ""), 3000),
        "### Your previous solution",
        "```python\n" + previous_code.strip() + "\n```",
        "### Execution feedback",
        feedback,
        "### Requirement\n"
        "Fix the code. Keep the same entry point signature. "
        "Output the complete corrected solution in one Python code block.",
    ]
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def render_plain(messages: list[dict[str, str]]) -> str:
    """不带 chat template 的纯文本渲染，用于调试与本地自检。"""
    return "\n\n".join(f"<{m['role']}>\n{m['content']}" for m in messages)

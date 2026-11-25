"""逐 assert 判定：把 LeetCode 风格的 ``def check(candidate)`` 拆成独立用例。

为什么不用"整段跑、一失败即停"
------------------------------
训练信号需要**部分通过率**（12 条断言过了 9 条 vs 过了 1 条，信息量完全不同），
也需要**首个失败用例**（喂给模型做修复提示）。所以这里用 AST 改写：

``assert candidate(x) == y``  →  ``_rec_eq(lambda: candidate(x), lambda: y, lineno)``

每个断言独立求值、独立捕获异常、独立计时，全部跑完再统一汇报。
改写失败（不是 ``assert`` 形式的测试）时退化为粗粒度模式，只给 AC/WA。
"""

from __future__ import annotations

import ast
import json
from typing import Any

from rl_algo.env.sandbox import DEFAULT_TIMEOUT_S, run_files

MARKER = "__RLRESULT__"
DEFAULT_CASE_TIMEOUT_S = 2.0
REQUIRED_KEYS = ("task_id", "entry_point", "test")

RUNNER_SOURCE = r'''
import json, os, signal, sys, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = []


def _read(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


META = json.loads(_read("meta.json") or "{}")
CASE_TIMEOUT = float(META.get("case_timeout") or 0)
ENTRY_POINT = META["entry_point"]
MODE = META.get("mode", "per_case")


def _emit(status, error=None, fatal=None):
    """只回传**摘要**：全部用例的 JSON 会被输出截断切断，导致父进程解析失败。"""
    fails = [case for case in CASES if not case.get("ok")]
    payload = {
        "status": status,
        "total": len(CASES),
        "passed": len(CASES) - len(fails),
        "n_timeout": sum(1 for case in fails if case.get("err") == "Timeout"),
        "n_error": sum(1 for case in fails if case.get("err") and case.get("err") != "Timeout"),
        "first_fail": fails[0] if fails else None,
        "fails": fails[:3],
        "error": error,
        "fatal": fatal,
    }
    sys.stdout.write("__RLRESULT__" + json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


class _CaseTimeout(Exception):
    pass


def _fmt(exc):
    if isinstance(exc, _CaseTimeout):
        return "Timeout"
    return type(exc).__name__ + ": " + str(exc)[:200]


def _on_alarm(signum, frame):
    raise _CaseTimeout()


signal.signal(signal.SIGALRM, _on_alarm)


def _deadline(seconds):
    if seconds and seconds > 0:
        signal.setitimer(signal.ITIMER_REAL, seconds)


def _cancel():
    signal.setitimer(signal.ITIMER_REAL, 0)


def _repr(value, limit=200):
    try:
        text = repr(value)
    except BaseException:
        text = "<unrepr>"
    return text if len(text) <= limit else text[:limit] + "..."


def _rec_eq(lhs_fn, rhs_fn, line):
    rec = {"line": line, "kind": "eq", "ok": False, "err": None, "lhs": None, "rhs": None}
    _deadline(CASE_TIMEOUT)
    try:
        try:
            lhs = lhs_fn()
        except BaseException as exc:
            rec["err"] = _fmt(exc)
            return
        try:
            rhs = rhs_fn()
        except BaseException as exc:
            rec["err"] = _fmt(exc)
            return
        rec["lhs"] = _repr(lhs)
        rec["rhs"] = _repr(rhs)
        try:
            rec["ok"] = bool(lhs == rhs)
        except BaseException as exc:
            rec["err"] = _fmt(exc)
    finally:
        _cancel()
        CASES.append(rec)


def _rec_true(fn, line):
    rec = {"line": line, "kind": "true", "ok": False, "err": None, "lhs": None, "rhs": None}
    _deadline(CASE_TIMEOUT)
    try:
        try:
            rec["ok"] = bool(fn())
        except BaseException as exc:
            rec["err"] = _fmt(exc)
    finally:
        _cancel()
        CASES.append(rec)


def main():
    solution = _read("solution.py")
    if solution is None:
        _emit("error", error={"kind": "RE", "type": "MissingSolution", "msg": "solution.py not found"})
        return
    try:
        compile(solution, "solution.py", "exec")
    except SyntaxError as exc:
        _emit("error", error={"kind": "CE", "type": type(exc).__name__, "msg": str(exc)})
        return

    ns = {"__name__": "solution"}
    preamble = _read("preamble.py")
    if preamble:
        try:
            exec(compile(preamble, "preamble.py", "exec"), ns)
        except BaseException as exc:
            _emit("error", error={"kind": "RE", "type": type(exc).__name__, "msg": "preamble: " + str(exc)})
            return

    try:
        exec(compile(solution, "solution.py", "exec"), ns)
    except BaseException as exc:
        _emit(
            "error",
            [],
            error={
                "kind": "RE",
                "type": type(exc).__name__,
                "msg": str(exc)[:300],
                "traceback": traceback.format_exc()[-1200:],
            },
        )
        return

    check_src = _read("check.py")
    if check_src is None:
        _emit("error", error={"kind": "RE", "type": "MissingCheck", "msg": "check.py not found"})
        return

    if MODE == "per_case":
        ns["_rec_eq"] = _rec_eq
        ns["_rec_true"] = _rec_true

    try:
        exec(compile(check_src, "check.py", "exec"), ns)
        check = ns["check"]
    except BaseException as exc:
        _emit("error", error={"kind": "RE", "type": type(exc).__name__, "msg": "check: " + str(exc)})
        return

    try:
        candidate = eval(ENTRY_POINT, ns)
    except BaseException as exc:
        _emit("error", error={"kind": "RE", "type": type(exc).__name__, "msg": "entry point: " + str(exc)})
        return

    if MODE == "coarse":
        try:
            check(candidate)
            CASES.append({"line": 0, "kind": "coarse", "ok": True, "err": None})
        except BaseException as exc:
            CASES.append({"line": 0, "kind": "coarse", "ok": False, "err": _fmt(exc)})
        _emit("ok")
        return

    fatal = None
    try:
        check(candidate)
    except BaseException as exc:
        fatal = _fmt(exc)
    _emit("ok", fatal=fatal)


main()
'''


class _AssertRewriter(ast.NodeTransformer):
    """把 ``assert`` 改写成可计数的记录调用。"""

    def _lambda(self, body: ast.expr) -> ast.Lambda:
        return ast.Lambda(
            args=ast.arguments(
                posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
            ),
            body=body,
        )

    def visit_Assert(self, node: ast.Assert) -> ast.Expr:
        target = node.test
        if (
            isinstance(target, ast.Compare)
            and len(target.ops) == 1
            and isinstance(target.ops[0], ast.Eq)
            and len(target.comparators) == 1
        ):
            call = ast.Call(
                func=ast.Name(id="_rec_eq", ctx=ast.Load()),
                args=[
                    self._lambda(target.left),
                    self._lambda(target.comparators[0]),
                    ast.Constant(value=node.lineno),
                ],
                keywords=[],
            )
        else:
            call = ast.Call(
                func=ast.Name(id="_rec_true", ctx=ast.Load()),
                args=[self._lambda(target), ast.Constant(value=node.lineno)],
                keywords=[],
            )
        return ast.copy_location(ast.Expr(value=call), node)


def transform_check(test_source: str) -> str | None:
    """改写 ``check`` 函数体；无法改写时返回 ``None``（调用方退化到 coarse 模式）。"""
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return None

    if not any(isinstance(node, ast.FunctionDef) and node.name == "check" for node in tree.body):
        return None

    rewritten = _AssertRewriter().visit(tree)
    ast.fix_missing_locations(rewritten)
    try:
        return ast.unparse(rewritten)
    except Exception:  # pragma: no cover - unparse 极少失败
        return None


def _parse_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(MARKER):
            try:
                return json.loads(line[len(MARKER) :])
            except json.JSONDecodeError:
                return None
    return None


def _classify(result, payload: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if result.timed_out:
        return "TLE", {"type": "GlobalTimeout", "msg": f"exceeded {DEFAULT_TIMEOUT_S}s"}
    if payload is None:
        if "SyntaxError" in result.stderr or "IndentationError" in result.stderr:
            return "CE", {"type": "SyntaxError", "msg": result.stderr[-800:]}
        return "RE", {"type": "NoResult", "msg": (result.stderr or result.stdout)[-800:]}
    if payload.get("status") == "error":
        error = payload.get("error") or {"kind": "RE", "type": "Unknown", "msg": ""}
        kind = error.get("kind", "RE")
        return ("CE" if kind == "CE" else "RE"), error

    cases = payload.get("fails") or []
    total = int(payload.get("total") or 0)
    passed = int(payload.get("passed") or 0)
    fatal = payload.get("fatal")
    if total == 0:
        return "UNSUPPORTED", {"type": "NoCases", "msg": "no assert executed"}
    if fatal:
        return "RE", {"type": "Fatal", "msg": fatal}
    if payload.get("n_timeout"):
        return "TLE", {"type": "CaseTimeout", "msg": "a test case timed out"}
    if payload.get("n_error"):
        first = next((case for case in cases if case.get("err")), {})
        return "RE", {"type": "CaseError", "msg": first.get("err"), "line": first.get("line")}
    if passed < total:
        return "WA", {"type": "WrongAnswer", "msg": "assertion failed"}
    return "AC", {"type": "Accepted", "msg": ""}


def evaluate(
    problem: dict[str, Any],
    code: str,
    *,
    case_timeout_s: float = DEFAULT_CASE_TIMEOUT_S,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """判定一段代码在题目隐藏测试上的表现。

    返回字典字段：``verdict`` / ``passed`` / ``total`` / ``pass_ratio`` /
    ``first_fail`` / ``error`` / ``elapsed_ms`` / ``cases``。
    """
    missing = [key for key in REQUIRED_KEYS if not problem.get(key)]
    if missing:
        raise ValueError(f"problem 缺少字段: {missing}")

    transformed = transform_check(problem["test"])
    mode = "per_case" if transformed else "coarse"
    files = {
        "meta.json": json.dumps(
            {
                "entry_point": problem["entry_point"],
                "case_timeout": case_timeout_s,
                "mode": mode,
            }
        ),
        "solution.py": code,
        "check.py": transformed or problem["test"],
        "runner.py": RUNNER_SOURCE,
    }
    if problem.get("preamble"):
        files["preamble.py"] = problem["preamble"]

    result = run_files(files, "runner.py", timeout_s=timeout_s)
    payload = _parse_payload(result.stdout) or {}
    verdict, error = _classify(result, payload if payload else None)

    passed = int(payload.get("passed") or 0)
    total = int(payload.get("total") or 0)
    first_fail = payload.get("first_fail")
    fails = payload.get("fails") or []

    return {
        "task_id": problem["task_id"],
        "verdict": verdict,
        "passed": passed,
        "total": total,
        "pass_ratio": (passed / total) if total else 0.0,
        "first_fail": first_fail,
        "error": error if verdict not in {"AC", "WA"} else None,
        "elapsed_ms": round(result.elapsed_ms, 1),
        "cases": fails,
        "mode": mode,
        "stderr": "" if verdict == "AC" else result.stderr[-600:],
    }


def evaluate_reference(problem: dict[str, Any], **kwargs) -> dict[str, Any]:
    """用题目自带的参考解自检。清洗阶段用它剔除坏题。"""
    reference = problem.get("completion") or ""
    return evaluate(problem, reference, **kwargs)

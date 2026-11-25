"""受限执行：子进程 + 超时 + 资源上限 + 输出截断。

设计取舍
--------
**这不是安全沙箱。** macOS 上 ``RLIMIT_AS`` 实际不生效，也没有可靠的方式禁用网络；
本模块的目标是"挡住学生代码的常见事故"——死循环、内存爆炸、刷屏输出、误删文件，
而不是对抗恶意代码。

生产级评测请换 Docker 或 Linux 上的 ``isolate``（见方案 §7 风险表）。
本模块的接口在两种后端下保持一致，替换时只需改 ``run_files`` 的实现。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

try:  # POSIX only
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_CPU_S = 5
DEFAULT_MEM_MB = 1024
DEFAULT_FSIZE_MB = 8
MAX_OUTPUT_CHARS = 6000

# 子进程只继承最小环境，避免 PYTHONPATH / 代理变量泄漏进来
_BASE_ENV = {
    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONDONTWRITEBYTECODE": "1",
}


@dataclass
class ExecResult:
    """一次子进程执行的原始结果。"""

    ok: bool
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    elapsed_ms: float
    workdir: str = field(default="", repr=False)

    @property
    def combined(self) -> str:
        return (self.stdout + ("\n" + self.stderr if self.stderr.strip() else "")).strip()


def _limit_setter(cpu_s: int, mem_mb: int, fsize_mb: int):
    """返回 preexec_fn：在 fork 之后、exec 之前收紧资源。"""

    def _set() -> None:  # pragma: no cover - 子进程内执行
        if resource is None:
            return
        pairs = [
            (resource.RLIMIT_CPU, (cpu_s, cpu_s)),
            (resource.RLIMIT_FSIZE, (fsize_mb << 20, fsize_mb << 20)),
            # macOS 上设置成功但不真正生效；Linux 上有效
            (resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20)),
            (resource.RLIMIT_CORE, (0, 0)),
        ]
        for key, value in pairs:
            try:
                resource.setrlimit(key, value)
            except (ValueError, OSError):
                pass

    return _set


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n…<truncated {len(text) - limit} chars>…\n{tail}"


def run_files(
    files: dict[str, str],
    main: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    cpu_s: int = DEFAULT_CPU_S,
    mem_mb: int = DEFAULT_MEM_MB,
    fsize_mb: int = DEFAULT_FSIZE_MB,
) -> ExecResult:
    """把 ``files`` 写进临时目录，执行 ``main``，返回结果。

    子进程以 ``-I`` 启动：忽略环境变量与 user site-packages，避免宿主环境影响结果。
    """
    start = time.perf_counter()
    workdir = tempfile.mkdtemp(prefix="rl_algo_")
    try:
        for name, content in files.items():
            path = os.path.join(workdir, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)

        env = dict(_BASE_ENV, HOME=workdir, TMPDIR=workdir)
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", main],
                cwd=workdir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                preexec_fn=_limit_setter(cpu_s, mem_mb, fsize_mb),
            )
        except OSError as exc:  # pragma: no cover - 环境异常
            return ExecResult(False, "", f"failed to spawn: {exc}", None, False, 0.0, workdir)

        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - 极端情况
                stdout, stderr = "", ""

        elapsed_ms = (time.perf_counter() - start) * 1000
        code = proc.returncode
        return ExecResult(
            ok=(code == 0 and not timed_out),
            stdout=_truncate(stdout or ""),
            stderr=_truncate(stderr or ""),
            exit_code=code,
            timed_out=timed_out,
            elapsed_ms=elapsed_ms,
            workdir=workdir,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_python(source: str, *, filename: str = "main.py", **kwargs) -> ExecResult:
    """便捷入口：单文件执行。"""
    return run_files({filename: source}, filename, **kwargs)


def _kill_group(proc: subprocess.Popen) -> None:
    """杀掉整个进程组，避免子进程再开子进程时漏杀。"""
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:  # pragma: no cover
            pass

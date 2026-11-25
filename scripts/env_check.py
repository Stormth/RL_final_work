#!/usr/bin/env python3
"""环境自检：CUDA / torch / trl / 显存 / 沙箱可用性。

集群上出问题，八成是环境而不是代码。先跑这个，把输出贴进群里，能省掉很多来回：

    python scripts/env_check.py
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _line(key: str, value: object) -> None:
    print(f"{key:<22} {value}")


def _nvidia_smi() -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return out.stdout.strip() or out.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        return f"failed: {exc}"


def _version(module: str) -> str:
    try:
        imported = __import__(module)
        return getattr(imported, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({type(exc).__name__})"


def main() -> int:
    print("=" * 60)
    print("环境自检")
    print("=" * 60)
    _line("python", sys.version.split()[0])
    _line("platform", platform.platform())
    _line("executable", sys.executable)

    print("\n--- GPU ---")
    smi = _nvidia_smi()
    _line("nvidia-smi", smi.replace("\n", " | ") if smi else "未找到（CPU 环境？）")

    print("\n--- 包版本 ---")
    for module in ("torch", "transformers", "trl", "peft", "datasets", "vllm", "accelerate"):
        _line(module, _version(module))

    torch_info = "未安装"
    devices: list[tuple[str, float]] = []
    try:
        import torch

        torch_info = f"cuda={torch.cuda.is_available()} version={torch.__version__}"
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(idx)
                devices.append((prop.name, prop.total_memory / 1024**3))
    except Exception as exc:  # noqa: BLE001
        torch_info = f"import 失败: {exc}"
    _line("torch.cuda", torch_info)
    for name, gb in devices:
        _line("  device", f"{name}  {gb:.1f} GB")

    print("\n--- 沙箱（子进程 + rlimit）---")
    try:
        from rl_algo.env.judge import evaluate

        problem = {
            "task_id": "smoke",
            "entry_point": "solve",
            "preamble": "",
            "test": "def check(candidate):\n    assert candidate(2) == 4\n",
        }
        result = evaluate(problem, "def solve(x):\n    return x * x\n")
        _line("judge", f"{result['verdict']} ({result['elapsed_ms']}ms)")
        sandbox_ok = result["verdict"] == "AC"
    except Exception as exc:  # noqa: BLE001
        _line("judge", f"失败: {exc}")
        sandbox_ok = False

    print("\n--- 建议 ---")
    if not devices:
        print("· 当前没有可用 GPU：只能跑数据/评测脚本，训练要放到 GPU 节点")
    else:
        biggest = max(gb for _, gb in devices)
        if biggest < 30:
            print(f"· 最大显存 {biggest:.0f}GB：建议 1.5B/3B + LoRA，group_size ≤ 4，别碰 7B 全参")
        elif biggest < 60:
            print(f"· 最大显存 {biggest:.0f}GB：可以跑 7B LoRA（group_size 4，HF 后端更稳）")
        else:
            print(f"· 最大显存 {biggest:.0f}GB：7B LoRA + vLLM colocate 均可")
    if _version("trl").startswith("MISSING"):
        print("· 未装 trl：pip install -e '.[train]' 后才能训练（基线评测不需要）")
    if not sandbox_ok:
        print("· 沙箱自检未通过：先修这个，否则所有奖励都是 0")

    return 0 if sandbox_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

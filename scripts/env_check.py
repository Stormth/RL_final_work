#!/usr/bin/env python3
"""环境自检：GPU / 驱动 / torch / trl / 网络 / 沙箱。

集群上出问题，八成是环境而不是代码。先跑这个，把输出贴进群里能省很多来回：

    python scripts/env_check.py
    python scripts/env_check.py --no-network    # 跳过联网探测（会更慢）

设计要点：**不依赖 torch 也能判断有没有 GPU**——直接解析 ``nvidia-smi``。
很多节点的 torch 还没装，但这不代表没有卡。
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# 某些机器的 ~/.cache 不可写，HF 导入时会打印一大段警告，干扰排错。
os.environ.setdefault("HF_HOME", os.path.join(tempfile.gettempdir(), "rl-algo-hf-home"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROBES = (
    ("pypi", "https://pypi.org/simple/"),
    ("huggingface", "https://huggingface.co"),
    ("hf-mirror", "https://hf-mirror.com"),
)
PROXY_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "HF_ENDPOINT")


def _line(key: str, value: object) -> None:
    print(f"{key:<22} {value}")


def _run(cmd: list[str], timeout: int = 15, *, keep_stderr: bool = False) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        text = out.stdout if not keep_stderr else (out.stdout or out.stderr)
        return (text or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"failed: {exc}"


def _gpus_from_smi() -> list[tuple[str, float]]:
    if not shutil.which("nvidia-smi"):
        return []
    text = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[tuple[str, float]] = []
    for row in text.splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) < 2:
            continue
        try:
            gpus.append((parts[0], float(parts[1]) / 1024))
        except ValueError:
            continue
    return gpus


def _version(module: str) -> str:
    try:
        imported = __import__(module)
        return getattr(imported, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({type(exc).__name__})"


def _probe(url: str, timeout: float = 4.0) -> str:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "rl-algo/env-check"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return f"OK ({response.status})"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL ({type(exc).__name__})"


def main(argv: list[str] | None = None) -> int:
    skip_network = "--no-network" in (argv or sys.argv[1:])
    print("=" * 62)
    print("环境自检")
    print("=" * 62)
    _line("python", sys.version.split()[0])
    _line("platform", platform.platform())
    _line("executable", sys.executable)

    print("\n--- GPU（nvidia-smi，不依赖 torch）---")
    gpus = _gpus_from_smi()
    if gpus:
        _line("可见 GPU", f"{len(gpus)} 张")
        for idx, (name, gb) in enumerate(gpus):
            _line(f"  [{idx}]", f"{name}  {gb:.0f} GB")
        _line("driver", _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]))
    else:
        _line("nvidia-smi", "未找到（CPU 环境？）")

    print("\n--- 包版本 ---")
    for module in ("torch", "transformers", "trl", "peft", "datasets", "vllm", "accelerate"):
        _line(module, _version(module))

    torch_info = "未安装"
    try:
        import torch

        torch_info = f"cuda_available={torch.cuda.is_available()} version={torch.__version__}"
        if torch.cuda.is_available():
            torch_info += f" device_count={torch.cuda.device_count()}"
    except Exception as exc:  # noqa: BLE001
        torch_info = f"import 失败: {exc}"
    _line("torch.cuda", torch_info)

    print("\n--- 代理与镜像 ---")
    found = False
    for key in PROXY_KEYS:
        if os.environ.get(key):
            _line(key, os.environ[key])
            found = True
    if not found:
        _line("(none)", "未设置代理 / HF_ENDPOINT")
    _line("pip 配置", _run(["pip", "config", "list"]).replace("\n", " | ") or "空")

    if not skip_network:
        print("\n--- 联网（各 4s 超时）---")
        for name, url in PROBES:
            _line(name, _probe(url))

    print("\n--- 沙箱（子进程 + rlimit）---")
    sandbox_ok = False
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

    print("\n--- 建议 ---")
    if gpus:
        biggest = max(gb for _, gb in gpus)
        if biggest < 30:
            print(f"· 单卡 {biggest:.0f}GB：1.5B/3B + LoRA，group_size ≤ 4；7B 放不下")
        elif biggest < 60:
            print(f"· 单卡 {biggest:.0f}GB：7B LoRA 可行（group_size 4，先别上 vLLM colocate）")
        else:
            print(f"· 单卡 {biggest:.0f}GB：7B LoRA + vLLM 均可，甚至可试 14B LoRA")
        if len(gpus) > 1:
            print(f"· 检测到 {len(gpus)} 张卡：先用 1 张跑通，再用 accelerate 做多卡数据并行")
    else:
        print("· 没有可见 GPU：只能跑数据/评测脚本，训练要放到 GPU 节点")

    if _version("torch").startswith("MISSING"):
        print("· 未装 torch：pip install -e '.[train]'（基线评测也需要它来跑模型）")
    if _version("trl").startswith("MISSING"):
        print("· 未装 trl：训练需要；只做基线评测可以暂时不装")
    if not sandbox_ok:
        print("· 沙箱自检未通过：先修这个，否则所有奖励都是 0")

    return 0 if sandbox_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

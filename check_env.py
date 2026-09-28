"""环境自检：一次性输出设备、驱动、依赖版本、CUDA 可用性与磁盘状况。

用法：
    python check_env.py
"""

import os
import platform
import shutil
import subprocess
import sys


def run(cmd: str) -> str:
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=20
        )
        return proc.stdout.strip() or proc.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<执行失败: {exc}>"


def section(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


section("1. NVIDIA 驱动与 GPU 全貌")
print(run("nvidia-smi"))

section("2. 关键字段速读")
print(
    run(
        "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,"
        "utilization.gpu,temperature.gpu,power.draw,power.limit "
        "--format=csv"
    )
)

section("3. Python 与依赖版本")
print(f"Python  {sys.version.split()[0]}    {platform.platform()}")
for pkg in ("torch", "transformers", "gradio", "accelerate", "modelscope"):
    try:
        module = __import__(pkg)
        print(f"  {pkg:14s} {getattr(module, '__version__', 'unknown')}")
    except ImportError:
        print(f"  {pkg:14s} 未安装")

section("4. CUDA 可用性（决定性判据）")
try:
    import torch

    driver_cuda = run(
        "nvidia-smi | head -n 3 | grep -o 'CUDA Version: [0-9.]*'"
    ).replace("CUDA Version: ", "")
    print(f"驱动支持的 CUDA 上限 : {driver_cuda or '未获取到'}")
    print(f"torch 编译的 CUDA    : {torch.version.cuda}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"可见设备数           : {torch.cuda.device_count()}")
        print(f"当前设备             : {torch.cuda.get_device_name(0)}")
        cap = "".join(map(str, torch.cuda.get_device_capability(0)))
        print(f"计算能力             : sm_{cap}")
        free, total = torch.cuda.mem_get_info()
        print(
            f"显存                 : 已用 {(total - free) / 1024 ** 3:.2f} GB "
            f"/ 共 {total / 1024 ** 3:.2f} GB"
        )
    else:
        print("\n[警告] torch 无法调用 GPU，请检查 torch 编译版本是否高于驱动上限。")
except ImportError:
    print("未安装 torch，跳过。")

section("5. 缓存与磁盘空间")
targets = [
    "/mnt/workspace",
    os.environ.get("MODELSCOPE_CACHE", ""),
    os.environ.get("HF_HOME", ""),
    os.path.expanduser("~"),
]
seen = set()
for path in targets:
    if not path or path in seen or not os.path.exists(path):
        continue
    seen.add(path)
    usage = shutil.disk_usage(path)
    print(
        f"  {path:38s} 总 {usage.total / 1024 ** 3:7.1f} GB  "
        f"可用 {usage.free / 1024 ** 3:7.1f} GB"
    )

print("\n自检完成。")

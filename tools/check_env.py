"""GPU / 环境体检：多节点排队时先跑这个，一眼看清「这台能不能跑 vLLM，能跑什么」。

背景（踩过的坑）：集群上每次 ssh 落到不同容器，torch/vLLM/CUDA 版本各不相同。
典型错配：torch 是 cu130，装的 vLLM wheel 却是按 CUDA 12 编的 → 报
`ImportError: libcudart.so.12`；或者 wheel 里没编本机 SM 架构 → 报
`no kernel image is available for execution on the device`（RTX 50 系最常见）。

用法:
  python tools/check_env.py            # 体检当前环境
  python tools/check_env.py --envs     # 附带列出所有 conda 环境各自的 torch/vLLM
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def sh(cmd, timeout=30) -> str:
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                           text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:                                   # noqa: BLE001
        return f"<{type(e).__name__}: {e}>"


def hdr(t):
    print(f"\n=== {t} ===")


def _clean_arch(a: str) -> str:
    return a.rstrip("af")                                # 12.0f / 12.0a → 12.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", action="store_true", help="顺带列出所有 conda 环境")
    args = ap.parse_args()
    problems: list[str] = []

    hdr("0. 位置/解释器")
    print("host     :", sh("hostname").strip())
    print("cwd      :", os.getcwd())
    print("python   :", sys.executable, sys.version.split()[0])
    print("conda    :", os.environ.get("CONDA_PREFIX", "-"))
    print("cuda工具 :", ",".join(glob.glob("/usr/local/cuda*")) or "无")
    no_gpu = False

    hdr("1. GPU（本容器能看到的）")
    print("CUDA_VISIBLE_DEVICES  :", os.environ.get("CUDA_VISIBLE_DEVICES", "-"))
    print("NVIDIA_VISIBLE_DEVICES:", os.environ.get("NVIDIA_VISIBLE_DEVICES", "-"))
    # 先判「有没有分到卡」：容器隔离下这是最容易搞错的一件事——
    # 没分到卡时 /dev/nvidia* 不存在，libcuda.so.1 指向的是 0 字节占位空壳，
    # 于是 vllm._C 报 `libcuda.so.1: file too short`、torch.cuda.is_available()=False，
    # 这时怎么改 torch/vllm 都没用，只能换到有卡的会话。
    devs = sorted(glob.glob("/dev/nvidia*")) if sys.platform.startswith("linux") else []
    print("/dev/nvidia*          :", ",".join(devs) or ("无" if sys.platform.startswith("linux")
                                                       else "（非 Linux，跳过该判定）"))
    stub = None
    for p in ("/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib/x86_64-linux-gnu/libcuda.so.1"):
        if os.path.lexists(p):
            try:
                stub = (p, os.path.getsize(p))
            except OSError:
                pass
            break
    if stub:
        tail = "  ← 0 字节占位空壳，真驱动没注入" if stub[1] == 0 else ""
        print(f"{stub[0]:38s}: {stub[1]} 字节{tail}")
    if sys.platform.startswith("linux") and (not devs or (stub and stub[1] == 0)):
        no_gpu = True
        problems.append("本容器没有 GPU（无 /dev/nvidia*、libcuda 是空壳占位）"
                        " → 环境怎么修都没用，换到有卡的会话/节点")
        print("  ⚠️ 无卡会话：下面 torch.cuda / vllm._C 的失败都属于预期现象，"
              "不代表环境错配；但第 4/5 节（wheel 里的 SM 架构、CUDA 运行时）"
              "**不需要卡也能查**，照样有效")
    caps = []
    if shutil.which("nvidia-smi"):
        print(sh("nvidia-smi -L").strip())
        csv = sh("nvidia-smi --query-gpu=index,name,memory.total,compute_cap,"
                 "driver_version --format=csv,noheader").strip()
        for line in csv.splitlines():
            print("  ", line)
            m = re.search(r",\s*(\d+)\.(\d+)\s*,", line)
            if m:
                caps.append(f"{m.group(1)}.{m.group(2)}")
        if caps and any(float(c) < 8.0 for c in caps):
            print("  ⚠️ compute capability < 8.0（Turing/更老）：FlashAttention/FlashInfer 不可用，"
                  "启动前需 export VLLM_ATTENTION_BACKEND=TRITON_ATTN")
    else:
        print("nvidia-smi 不存在")
        problems.append("没有 nvidia-smi（这台不是 GPU 节点？）")

    hdr("2. torch")
    torch_ok = False
    torch_cuda = None
    archs: list[str] = []
    try:
        import torch
        torch_cuda = torch.version.cuda
        torch_ok = bool(torch.cuda.is_available())
        archs = list(torch.cuda.get_arch_list() or [])
        print(f"version={torch.__version__}  built_for_cuda={torch_cuda}  "
              f"available={torch_ok}  gpu_count={torch.cuda.device_count()}")
        print("arch_list:", ",".join(archs) or "-")
        if not torch_ok and not no_gpu:
            problems.append("torch 看不到 GPU（有 /dev/nvidia* 但 is_available=False → 驱动/torch 版本问题）")
        for c in caps:
            if f"sm_{c}" not in {_clean_arch(a) for a in archs}:
                problems.append(f"torch 的 arch_list 里没有 sm_{c}")
    except Exception as e:                                   # noqa: BLE001
        print(f"import torch 失败：{type(e).__name__}: {e}")
        problems.append("torch 不可用")

    hdr("3. vLLM")
    vllm_dir = None
    try:
        import vllm
        print("version:", vllm.__version__, "\npath   :", vllm.__file__)
        vllm_dir = Path(vllm.__file__).parent
    except Exception as e:                                   # noqa: BLE001
        print(f"import vllm 失败：{type(e).__name__}: {e}")
        problems.append("vLLM 不可用")
    try:
        import vllm._C  # noqa: F401
        print("vllm._C  : import OK")
    except Exception as e:                                   # noqa: BLE001
        print(f"vllm._C  : import 失败 → {type(e).__name__}: {e}")
        if no_gpu and "libcuda" in str(e):
            print("           （无卡会话：libcuda.so.1 是空壳占位导致，属预期，"
                  "不是环境错配；换到有卡节点再验这一项）")
        else:
            problems.append("vllm._C 加载失败（多为 CUDA 运行时不匹配，见第 5 节）")

    hdr("4. vLLM wheel 里编了哪些 SM 架构（决定本机卡能不能跑）")
    if vllm_dir:
        sos = sorted(vllm_dir.glob("_C*.so"))
        exe = shutil.which("cuobjdump")
        if not exe:
            for c in ("/usr/local/cuda-13.0/bin/cuobjdump", "/usr/local/cuda/bin/cuobjdump"):
                if Path(c).exists():
                    exe = c
                    break
        if not sos:
            print("找不到 _C*.so")
        elif not exe:
            print("没有 cuobjdump（装 CUDA toolkit 或用 /usr/local/cuda-*/bin/cuobjdump）")
        else:
            out = sh([exe, "--list-elf", str(sos[0])], timeout=180)
            found = sorted({f"sm_{m}" for m in re.findall(r"sm_(\d+[af]?)", out)})
            print(f"{sos[0].name}: {','.join(found) or '未解析到'}")
            for c in caps:
                if f"sm_{c}" not in found:
                    problems.append(
                        f"vLLM wheel 里没有 sm_{c} 的 kernel → 在 {c} 卡上会 "
                        f"『no kernel image is available』（换 wheels.vllm.ai 对应 cu 版本的 wheel，"
                        f"或源码编译 TORCH_CUDA_ARCH_LIST={c}）")
    else:
        print("跳过（vLLM 未安装）")

    hdr("5. CUDA 运行时库（cu12/cu13 混装判定）")
    nv = [ln for ln in sh([sys.executable, "-m", "pip", "list"], timeout=120).splitlines()
          if re.match(r"^(nvidia-|torch|vllm)", ln)]
    # cutlass-dsl 本来就同时发 cu12/cu13 两版，属于正常，别当成混装
    both_ok = re.compile(r"cutlass-dsl")
    fam = {"12": [], "13": []}
    for ln in nv:
        name = ln.split()[0]
        if both_ok.search(name):
            continue
        if "cu13" in ln or re.search(r"\s1[3-9]\.", ln):
            fam["13"].append(name)
        elif "cu12" in ln or re.search(r"\s12\.", ln):
            fam["12"].append(name)
    print("cu13 系:", ", ".join(fam["13"]) or "-")
    print("cu12 系:", ", ".join(fam["12"]) or "-")
    print("libcudart:", sh("find / -name 'libcudart.so*' -not -path '/proc/*' 2>/dev/null "
                           "| head -5").strip() or "无")
    if fam["12"] and fam["13"]:
        problems.append("同时存在 cu12 和 cu13 的包 → 混装，vLLM/torch 很容易起不来")
    if torch_cuda:
        major = str(torch_cuda).split(".")[0]
        if fam["12"] and major == "13":
            problems.append(f"torch 是 cu{torch_cuda}，但装了 cu12 的 nvidia 包（vLLM wheel 大概也是 cu12 编的）")

    if args.envs:
        hdr("6. 所有 conda 环境各自的 torch/vLLM")
        envs = []
        try:
            envs = json.loads(sh(["conda", "env", "list", "--json"])).get("envs", [])
        except Exception:                                    # noqa: BLE001
            print("读不到 conda 环境列表")
        for e in envs:
            py = Path(e) / "bin" / "python"
            if not py.exists():
                continue
            info = sh([str(py), "-c",
                       "import sys;"
                       "t=__import__('torch');"
                       "print('py'+sys.version.split()[0], 'torch'+t.__version__, "
                       "'cu'+str(t.version.cuda), 'avail', t.cuda.is_available())"],
                      timeout=120).strip().splitlines()
            v = sh([str(Path(e) / "bin" / "pip"), "show", "vllm"], timeout=60)
            ver = next((ln.split(":", 1)[1].strip() for ln in v.splitlines()
                        if ln.startswith("Version:")), "-")
            print(f"{e:45s} vllm={ver:16s} {(info[-1] if info else '?')}")

    hdr("结论")
    if problems:
        for p in problems:
            print("  ✗", p)
    else:
        print("  ✓ 没发现明显问题")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

"""同步核对：生成本地文件清单（相对路径 + sha256），或按清单核对另一份拷贝。

为什么需要：本项目在 Windows 上改、在 Linux 服务器上跑，靠 scp/rsync/Jupyter 上传。
上传漏文件很常见，而漏一个 `.py` 的表现就是一行 `ModuleNotFoundError`——
一次只暴露一个，来回试很多轮。本工具一次列出**所有**缺失/不一致的文件。

    # 开发机：生成清单（写 PROJECT_FILES.txt）
    python tools/verify_sync.py --manifest

    # 服务器：按清单核对当前目录（缺什么、哪个文件旧了什么，一次列全）
    python tools/verify_sync.py --check

退出码：--check 模式下有缺失/不一致 → 1，全一致 → 0（可直接接进部署脚本）。
"""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "PROJECT_FILES.txt"

# 这些目录**不进清单**：baselines 各自 clone、data 是语料产物、results/logs 是跑批输出。
# ★ `traces` 与 prompts/ 下的去重 prompt 也必须在列：逐题 trace 是**每次 run 的产物**
#   （默认开启，单数据集轻松上千个文件），混进清单会让"同步核对"变成核对一堆会变的文件，
#   真正的代码/prompt 差异反而被淹掉（实测：一次被中断的 2 题测试就贡献了 336 个文件）。
EXCLUDE_DIRS = {
    "baselines", "data", "results", "logs", "traces", "lr_test_wd",
    "__pycache__", ".git", ".idea", ".vscode", ".ipynb_checkpoints",
    ".cache", ".pytest_cache", ".mypy_cache",
}
# 只收这些后缀（.pyc/.npy/.jsonl 这些产物不收）
INCLUDE_SUFFIXES = {".py", ".json", ".md", ".sh", ".txt", ".cfg", ".toml", ".ini"}


def _walk():
    """产出 (相对路径 posix 风格, Path)，跳过排除目录与产物后缀。"""
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in EXCLUDE_DIRS for part in rel.parts[:-1]):
            continue
        if rel.name.startswith("."):
            continue
        if p.suffix.lower() not in INCLUDE_SUFFIXES:
            continue
        if rel.name == MANIFEST.name:
            continue
        yield rel.as_posix(), p


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build_manifest() -> int:
    rows = [(rel, p.stat().st_size, _sha256(p)) for rel, p in _walk()]
    MANIFEST.write_text(
        "# 项目文件清单（相对仓库根）。由 tools/verify_sync.py --manifest 生成。\n"
        "# 格式：sha256  size  path    —— 服务器上跑 --check 核对\n"
        + "".join(f"{h}  {n:>9}  {rel}\n" for rel, n, h in rows),
        encoding="utf-8", newline="\n")
    print(f"已写 {MANIFEST.name}：{len(rows)} 个文件")
    print("  拿到服务器上后：python tools/verify_sync.py --check")
    return 0


def check_manifest() -> int:
    if not MANIFEST.exists():
        print(f"✗ 找不到 {MANIFEST.name} —— 先在开发机跑 --manifest 并把它一起上传")
        return 1
    want = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        h, _size, rel = line.split(None, 2)
        want[rel.strip()] = h

    missing, differ, extra = [], [], []
    for rel, h in sorted(want.items()):
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
        elif _sha256(p) != h:
            differ.append(rel)
    for rel, _p in _walk():
        if rel not in want:
            extra.append(rel)

    total = len(want)
    if not missing and not differ:
        print(f"✓ 全部一致：{total} 个文件（无缺失、无内容差异）")
        if extra:
            print(f"  （多出 {len(extra)} 个本地新增/清单外文件，通常无害）")
        return 0

    print(f"✗ 与清单不一致（清单共 {total} 个文件）\n")
    if missing:
        print(f"【缺失 {len(missing)} 个】—— 这些会直接导致 ModuleNotFoundError：")
        for rel in missing:
            print(f"    ✗ {rel}")
        print()
    if differ:
        print(f"【内容不同 {len(differ)} 个】—— 服务器上是旧版本：")
        for rel in differ:
            print(f"    ≠ {rel}")
        print()
    if extra:
        print(f"（清单外新增 {len(extra)} 个，可能是有意加的，也可能该更新清单）")
    return 1


def main():
    ap = argparse.ArgumentParser(description="项目文件同步核对")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", action="store_true", help="生成本地清单")
    g.add_argument("--check", action="store_true", help="按清单核对当前目录")
    a = ap.parse_args()
    sys.exit(build_manifest() if a.manifest else check_manifest())


if __name__ == "__main__":
    main()
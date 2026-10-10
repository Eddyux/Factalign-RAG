"""清理某次 run 的产物，让下一轮实验不受干扰（SIGMOD ARI 要求的「三阶段」之 clean-up）。

ARI 对每个实验要求写清 setup / running / **clean-up** 三段，并说明「怎么确认各阶段
按预期完成」。本脚本是 clean-up 那一段，`tools/verify_run.py` 是「确认」那一段。

⚠️ 默认是 **dry-run**（只打印要删什么，不真删）。要真删必须加 `--yes`。
   删除范围被硬限制在仓库内（`config.ROOT` 之下），任何越界路径直接拒绝。

三种典型用法：
  1) 只清某一格（换参数重跑前）：
       python tools/clean_run.py --method proprag --dataset hotpotqa --v 1 --r 1
  2) 清某个方法全部的临时 trace（保留结果表）：
       python tools/clean_run.py --method proprag --what traces
  3) 「冷缓存」准备 + 全清后重跑（每个方案/每个数据集之间做一次）：
       python tools/clean_run.py --all-but-results --yes
       python tools/clean_run.py --check-cold                       # 确认环境是干净的

可选「冷缓存」动作（默认只检查，不执行；要动必须显式给 flag）：
  --check-cold        检查 vLLM 是否在线、是否重启过（prefix cache 是进程内的，重启即清空）
  --flush-cache       调 vLLM 的缓存清理（若该版本暴露；失败只警告）
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.paths import TRACES_DIR, rel_to_root
from common.result import graph_dir, result_path
from config import CORPUS_DIR

TRACES_ROOT = TRACES_DIR
LOGS_ROOT = ROOT / "logs"
RESULTS_ROOT = ROOT / "results"
DATA_ROOT = ROOT / "data"

# 可以清的目标（--what 可选值）
TARGETS = ("graph", "traces", "results", "logs", "figures", "server", "all")


def _within_root(p: Path) -> bool:
    """只允许删仓库内的路径（防止 --dataset ../.. 这类越界）。"""
    try:
        Path(p).resolve().relative_to(ROOT.resolve())
        return True
    except Exception:                               # noqa: BLE001
        return False


class Cleaner:
    def __init__(self, yes: bool = False):
        self.yes = yes
        self.planned: list[Path] = []
        self.deleted: list[Path] = []
        self.skipped: list[str] = []
        self.freed = 0

    def plan(self, p: Path, note: str = ""):
        p = Path(p)
        # 先查越界、再查存在性：顺序反了的话，仓库外但恰好不存在的路径会被报成
        # 「不存在」而不是「越界」，守卫看起来像没生效（实测踩过）。
        if not _within_root(p):
            self.skipped.append(f"⛔ 越界（不在仓库内），拒绝：{p}")
            return
        if not p.exists():
            self.skipped.append(f"不存在：{rel_to_root(p)}")
            return
        self.planned.append(p)
        try:
            size = (sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    if p.is_dir() else p.stat().st_size)
        except Exception:                           # noqa: BLE001
            size = 0
        print(f"  {'删' if self.yes else '将删'} {rel_to_root(p):60s} "
              f"{size / 1e6:8.2f} MB{('  # ' + note) if note else ''}")

    def run(self):
        if not self.planned:
            print("（没有要清的东西）")
            return 0
        if not self.yes:
            total = sum((sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                         if p.is_dir() else p.stat().st_size)
                        for p in self.planned if p.exists())
            print(f"\n[dry-run] 共 {len(self.planned)} 项、约 {total / 1e6:.1f} MB —— "
                  f"加 --yes 才真删")
            return 0
        for p in self.planned:
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                self.deleted.append(p)
            except Exception as e:                  # noqa: BLE001
                print(f"  ⚠ 删除失败 {rel_to_root(p)}: {e}")
        print(f"\n[clean] 已删 {len(self.deleted)} 项")
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# 各类目标的清理
# ══════════════════════════════════════════════════════════════════════════════
def clean_graph(c: Cleaner, method: str, dataset: str, v: int, r: int | None):
    """图/索引目录 + 结果侧的建图凭证登记。

    ⚠️ 图与结果**不匹配**是常见坑：删了图但留着结果，下次读到的结果其实来自旧图。
    所以这里连 `results/{m}/{ds}/v{v}r*.build.json` 一起清，并在提示里说明。
    """
    d = graph_dir(method, dataset, v)
    c.plan(d, "图/索引 + build/ 凭证")
    if method == "factalign":
        c.plan(CORPUS_DIR / dataset / f"v{v}", "本方法语料图")
    for p in (RESULTS_ROOT / method / dataset).glob(f"v{v}r*.build.json"):
        c.plan(p, "建图凭证登记（与图一起清，避免下次读到旧图的口径）")


def clean_traces(c: Cleaner, method: str, dataset: str, v: int, r: int | None, sample: int = 0):
    """清逐题 trace。

    ⚠️ trace 目录名不带题数是 `v{n}r{m}`、带题数是 `v{n}r{m}_n{sample}`。
    只按精确名字找会漏掉另一批（实测踩过：不带 --sample 时静默漏清），
    所以这里按 `v{n}r{m}*` 扫；给了 sample 才收窄。
    """
    root = TRACES_ROOT / method / dataset
    if r is None:
        c.plan(root, "该方法该数据集全部逐题 trace")
        return
    pat = f"v{v}r{r}_n{sample}" if sample else f"v{v}r{r}*"
    for d in sorted(root.glob(pat)):
        c.plan(d, "逐题 trace 目录" if d.is_dir() else "trace 索引 jsonl")


def clean_results(c: Cleaner, method: str, dataset: str, v: int, r: int | None):
    root = RESULTS_ROOT / method / dataset
    if r is None:
        for p in root.glob(f"v{v}r*.jsonl"):
            c.plan(p, "逐题结果")
    else:
        c.plan(result_path(method, dataset, v, r), "逐题结果")
    for p in root.glob(f"v{v}r*.build.json"):
        c.plan(p, "建图凭证登记")


def clean_logs(c: Cleaner, method: str, dataset: str, v: int, r: int | None):
    d = LOGS_ROOT / method
    pat = f"{dataset}_*v{v}r{r if r is not None else '*'}_*.log"
    for p in d.glob(pat):
        c.plan(p, "stdout 日志分段")


def clean_figures(c: Cleaner):
    for p in RESULTS_ROOT.glob("figures*"):
        c.plan(p, "生成的图/表")


def clean_server(c: Cleaner):
    c.plan(RESULTS_ROOT / "server_config.json", "服务端配置快照（下次 run 会重探）")
    c.plan(RESULTS_ROOT / "summary.jsonl", "汇总（⚠ 论文数据来源，通常**不该**删）")
    c.plan(RESULTS_ROOT / "run_manifest.jsonl", "manifest（⚠ 同上）")
    c.plan(RESULTS_ROOT / "failures.jsonl", "失败统计")
    c.plan(RESULTS_ROOT / "build_tokens.jsonl", "建图 token 差值")


# ══════════════════════════════════════════════════════════════════════════════
# 阶段检查 / 冷缓存
# ══════════════════════════════════════════════════════════════════════════════
def check_cold(flush: bool = False) -> int:
    """确认「环境是冷的」：vLLM 可达、prefix cache 计数器为 0（= 本次进程还没命中过）。

    为什么要检查：prefix caching 会把上一次 run 的前缀留在引擎里，导致本轮 prefill
    几乎不耗时 → GPU 时间与 token 都不可比。ARI 要求说明"冷缓存是怎么强制的"。
    """
    from common.server_probe import probe_server
    from config import VLLM_BASE_URL, VLLM_EMBED_BASE_URL
    ok = True
    for name, base in (("chat", VLLM_BASE_URL), ("embed", VLLM_EMBED_BASE_URL)):
        info = probe_server(base)
        m = info.get("engine_metrics") or {}
        q, h = m.get("prefix_cache_queries"), m.get("prefix_cache_hits")
        print(f"[cold] {name}: available={info.get('available')} "
              f"prefix_cache_queries={q} hits={h} "
              f"running={m.get('num_requests_running')}")
        if not info.get("available"):
            ok = False
            print(f"[cold] ⚠ {name} 探测不到 → 无法确认冷缓存（成本口径需注明）")
            continue
        if q:
            print(f"[cold] ⚠ {name} 的 prefix cache 已累计 {q:.0f} 次查询"
                  f"（hits={h}）→ 若要与干净状态对比，**重启 vLLM 进程**最彻底"
                  f"（prefix cache 是进程内的）")
            if flush:
                import requests
                root = base.rsplit("/v1", 1)[0] if "/v1" in base else base
                for ep in ("/reset_prefix_cache", "/flush_cache"):
                    try:
                        r = requests.post(f"{root}{ep}", timeout=5)
                        print(f"[cold] {ep} → HTTP {r.status_code}")
                        if r.status_code < 400:
                            break
                    except Exception as e:          # noqa: BLE001
                        print(f"[cold] {ep} 失败：{type(e).__name__}: {e}")
    print("[cold] 结论：" + ("可确认冷启动 ✓" if ok else "无法确认（见上面警告）"))
    return 0 if ok else 1


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="清理某次 run 的产物（默认 dry-run；ARI 的 clean-up 阶段）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--method", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--v", type=int, default=1, help="graph_version")
    ap.add_argument("--r", type=int, default=None, help="retrieval 号；不给=该 v 的全部 r")
    ap.add_argument("--what", default="all",
                    help=f"逗号分隔：{','.join(TARGETS)}（默认 all=除 results 表以外的全部）")
    ap.add_argument("--all-but-results", action="store_true",
                    help="清掉图/trace/logs/缓存，但**保留** results/ 里的逐题结果与汇总")
    ap.add_argument("--yes", action="store_true", help="真的要删（不加则 dry-run）")
    ap.add_argument("--check-cold", action="store_true", help="只检查环境是否冷（不删东西）")
    ap.add_argument("--flush-cache", action="store_true", help="尝试调 vLLM 清缓存")
    args = ap.parse_args()

    if args.check_cold or args.flush_cache:
        return check_cold(flush=args.flush_cache)

    if not args.method or not args.dataset:
        raise SystemExit("要给 --method 和 --dataset（或用 --check-cold）")
    if args.r is None:
        print(f"[clean] 未指定 --r → 清理 v{args.v} 下**所有** r")

    wants = {s.strip() for s in args.what.split(",") if s.strip()}
    if args.all_but_results:
        wants = {"graph", "traces", "logs", "figures"}
    if "all" in wants:
        wants = {"graph", "traces", "logs", "server"} - {"results"}   # 结果表默认保留

    c = Cleaner(yes=args.yes)
    print(f"[clean] 目标：{sorted(wants)} | method={args.method} dataset={args.dataset} "
          f"v={args.v} r={args.r if args.r is not None else '*'}"
          f" | {'真删' if args.yes else 'dry-run'}")
    print("⚠️ 图与结果必须一起清（否则下次读到的结果可能来自旧图）；"
          "论文数据在 results/ 的逐题 jsonl + summary.jsonl，清之前先备份。\n")

    if "graph" in wants:
        clean_graph(c, args.method, args.dataset, args.v, args.r)
    if "traces" in wants:
        clean_traces(c, args.method, args.dataset, args.v, args.r, args.sample)
    if "results" in wants:
        clean_results(c, args.method, args.dataset, args.v, args.r)
    if "logs" in wants:
        clean_logs(c, args.method, args.dataset, args.v, args.r)
    if "figures" in wants:
        clean_figures(c)
    if "server" in wants:
        clean_server(c)

    for s in c.skipped:
        print(f"  （跳过）{s}")
    return c.run()


if __name__ == "__main__":
    sys.exit(main())

"""校验一次 run 是否「按预期完成」——SIGMOD ARI 要求说明「怎么检查各阶段」。

ARI 原文要求作者记录：(i) 怎么做 setup/running/clean-up，以及 (ii) **怎么检查**
这些阶段按预期完成。本脚本就是 (ii)：把「这次 run 到底是不是干净、完整、可比」
变成一串 PASS/FAIL 断言，而不是靠人肉翻日志。

检查项（每条都给出「为什么这条重要」）：
  构建阶段
    1. build/{config,env,stats,checksums}.json 四件套是否齐全
    2. 语料指纹：data/merged/{ds}_{n}/checksums.json 存在且无 problems
    3. 图规模：stats 里实体/关系/命题/超边至少有一类 > 0（建图类方法）
    4. chunk 数一致：图侧 chunk 数 == merged 的 chunk 数（口径漂移会在这里暴露）
    5. 服务端口径：results/server_config.json 存在、prefix_caching 不是 unknown
    6. tokenizer 可用（不可用会退化到字符级切块 → chunk 数与别的 run 不可比）
  检索阶段
    7. 逐题结果行数 == 样本题数（或明确记录为不完整）
    8. 每题都有 em/f1 与成本字段（prompt/completion/embed token）
    9. 失败/重试在阈值内（默认：失败率 ≤ 5%、重试题 ≤ 20%）
   10. 逐题 trace 覆盖率（若开了 trace）
   11. 成本自洽：summary 的三种 token 之和 == 逐题累加；GPU 采样可用时 busy 秒 > 0

用法:
  python tools/verify_run.py --method proprag --dataset hotpotqa --sample 500
  python tools/verify_run.py --method proprag --dataset hotpotqa --sample 500 --v 1 --r 1
  python tools/verify_run.py --all --sample 500          # 遍历 results/ 里所有 run
  python tools/verify_run.py --method standardrag --dataset hotpotqa --sample 10 --strict
退出码：0 = 全通过；1 = 有 FAIL（`--warn-only` 可让它永远返回 0）
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.paths import build_dir, merged_paths, rel_to_root, trace_run_dir
from common.result import graph_dir, result_path
from config import CORPUS_DIR, RESULTS_DIR, SAMPLES_DIR

# 建图类方法（对它们才要求图规模 > 0）
GRAPH_METHODS = {"standardrag", "factalign", "lightrag", "hipporag",
                 "hypergraphrag", "ircot", "s2grag", "selfask", "proprag"}


class Report:
    def __init__(self, warn_only: bool = False):
        self.rows: list[tuple[str, str, str]] = []      # (状态, 检查项, 说明)
        self.warn_only = warn_only

    def add(self, ok, name: str, detail: str = "", warn: bool = False):
        status = "PASS" if ok else ("WARN" if warn else "FAIL")
        self.rows.append((status, name, detail))

    def note(self, name: str, detail: str = ""):
        """信息行：**永远打印，但不计入 PASS/WARN/FAIL**。

        用途：那些"每次都会触发、且不可行动"的提示（典型是排队放大 —— 它是并发配置的
        必然结果，降 workers 只会拉长墙钟、不改变 GPU busy 秒）。把它们混进 WARN 计数
        会让 `WARN N` 永远挂着一个数字 → **告警疲劳**，真正的 WARN 反而被忽略。
        信息本身仍要留在输出里，所以降级而不是删除。
        """
        self.rows.append(("INFO", name, detail))

    def dump(self, title: str) -> int:
        n_fail = sum(1 for s, _, _ in self.rows if s == "FAIL")
        n_warn = sum(1 for s, _, _ in self.rows if s == "WARN")
        print(f"\n═══ {title} ═══")
        for s, name, detail in self.rows:
            mark = {"PASS": "✓", "WARN": "!", "FAIL": "✗", "INFO": "·"}[s]
            print(f"  [{mark}] {name:38s} {detail}")
        print(f"  → PASS {sum(1 for s,_,_ in self.rows if s=='PASS')} / "
              f"WARN {n_warn} / FAIL {n_fail}"
              f"（另有 {sum(1 for s,_,_ in self.rows if s=='INFO')} 条信息行，不计入）")
        if n_fail and not self.warn_only:
            return 1
        return 0


def _load_jsonl(p: Path) -> list[dict]:
    out = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except Exception:                   # noqa: BLE001
                    pass
    return out


def verify_one(method: str, dataset: str, sample: int, v: int, r: int,
               rep: Report, max_fail_rate: float = 0.05, max_retry_rate: float = 0.20,
               require_trace: bool = False, require_gpu: bool = False):
    tag = f"{method}/{dataset} v{v}r{r} n{sample}"
    print(f"\n── {tag} ──")

    # ── 1. 建图凭证四件套 ──
    d = graph_dir(method, dataset, v)
    if method == "factalign":
        d = CORPUS_DIR / dataset / f"v{v}"
    if method in GRAPH_METHODS and method not in ("standardrag", "zeroshot"):
        bd = build_dir(d)
        need = ("config.json", "env.json", "stats.json", "checksums.json")
        have = [n for n in need if (bd / n).exists()]
        rep.add(len(have) == len(need), "build 四件套齐全",
                f"{rel_to_root(bd)}（{len(have)}/{len(need)}）",
                warn=(len(have) > 0))
        # ── 3/4. 图规模 + chunk 数一致 ──
        stats = {}
        if (bd / "stats.json").exists():
            stats = json.loads((bd / "stats.json").read_text(encoding="utf-8"))
        summ = stats.get("summary") or {}
        if stats:
            keys = ("entities", "relations", "propositions", "hyperedges", "chunks")
            got = {k: summ.get(k) for k in keys if summ.get(k)}
            rep.add(bool(got), "图规模统计非空", json.dumps(got, ensure_ascii=False))
            # ★ factalign **不参与**「chunk 数与 merged 一致」这条 —— 单位不同（句 vs 段）：
            #   · factalign 走 `data/corpus/<ds>/v{n}/`，那里 `chunks.json` 是**句子**索引
            #     （`extract.py:138` 一句一条：2wiki 实测 3,356 段 → 12,236 句）；
            #   · 而 merged 的 `n_chunks` 是**段落/块**数（`build_merged_corpus.py:106`
            #     对每个 context 传 `sentences=[full]`，不切块 → 1 段 = 1 块）。
            #   拿"句数"比"段数"必然不相等，差值随数据集的句/段比变化 → **必然假警**，
            #   而且提示会把"两份产物都没问题"误报成"口径漂移"。
            #   factalign 真正的检索单元是**命题**（`propositions`，上面那行已报），
            #   不是 chunks.json；所以这条对它的意义本来就是零。
            if method != "factalign":
                gchunks = summ.get("chunks")
                mp = merged_paths(dataset, sample)
                mchunks = None
                if mp["checksums"].exists():
                    mchunks = json.loads(mp["checksums"].read_text(encoding="utf-8")).get("n_chunks")
                if gchunks and mchunks:
                    same = int(gchunks) == int(mchunks)
                    rep.add(same, "chunk 数与 merged 一致",
                            f"图 {gchunks} vs merged {mchunks}"
                            + ("" if same else " → 口径漂移！两份产物的语料单元不同"),
                            warn=not same)
                else:
                    rep.add(False, "chunk 数与 merged 一致",
                            f"缺一方（图={gchunks} merged={mchunks}）→ 图侧没取到 chunk 数"
                            f"（不是 merged 的问题，别去跑 build_merged_corpus）",
                            warn=True)

    # ── 2. 语料指纹 ──
    mp = merged_paths(dataset, sample)
    if mp["checksums"].exists():
        ck = json.loads(mp["checksums"].read_text(encoding="utf-8"))
        probs = ck.get("problems") or []
        rep.add(not probs, "语料指纹无 problems",
                f"{ck.get('n_contexts')} contexts / {ck.get('n_chunks')} chunks"
                + (f" ⚠ {probs}" if probs else ""), warn=bool(probs))
    else:
        rep.add(False, "语料指纹无 problems",
                f"缺 {rel_to_root(mp['checksums'])}（可选，但建议补）", warn=True)

    # ── 5/6. 服务端口径 ──
    scp = RESULTS_DIR / "server_config.json"
    if scp.exists():
        sc = json.loads(scp.read_text(encoding="utf-8"))
        pc = sc.get("prefix_caching")
        rep.add(pc in ("on", "off"), "prefix caching 已判定",
                f"{pc}（{sc.get('prefix_caching_evidence', '')[:70]}…）", warn=(pc == "unknown"))
        tk = (sc.get("tokenizer") or {}).get("available")
        rep.add(bool(tk), "切块 tokenizer 可用",
                "可用" if tk else (sc.get("tokenizer") or {}).get("impact", "不可用"),
                warn=not tk)
        for w in (sc.get("warnings") or []):
            rep.add(False, "服务端口径告警", w[:110], warn=True)
    else:
        rep.add(False, "prefix caching 已判定",
                "缺 results/server_config.json（老结果或未跑探测）→ 成本口径无法自证", warn=True)

    # ── 7/8. 逐题结果 ──
    rows = _load_jsonl(result_path(method, dataset, v, r))
    sp = SAMPLES_DIR / f"{dataset}_{sample}.json"
    n_items = None
    if sp.exists():
        n_items = len(json.loads(sp.read_text(encoding="utf-8")).get("items", []))
    if not rows:
        rep.add(False, "逐题结果存在", f"缺/空 {rel_to_root(result_path(method, dataset, v, r))}")
        return
    rep.add(True, "逐题结果存在", f"{len(rows)} 行")
    if n_items:
        rep.add(len(rows) == n_items, "题数完整",
                f"{len(rows)}/{n_items}" + ("" if len(rows) == n_items else " → 未跑完"),
                warn=(len(rows) < n_items))
    need_keys = {"em", "f1", "prompt_tokens", "completion_tokens", "embed_tokens",
                 "chat_seconds", "embed_seconds"}
    miss = sorted(need_keys - set(rows[0].keys()))
    rep.add(not miss, "成本字段齐全",
            "OK" if not miss else f"缺 {miss} → 老格式，重跑一次即可",
            warn=bool(miss))

    # ── 9. 失败/重试 ──
    fails = sum(1 for x in rows if str(x.get("pred", "")).startswith("<ERROR"))
    fr = fails / len(rows)
    rep.add(fr <= max_fail_rate, "失败率达标",
            f"{fails}/{len(rows)} = {fr:.2%}（阈值 {max_fail_rate:.0%}）", warn=fr > 0)
    # failures.jsonl 里有重试分布
    rs = None
    for row in _load_jsonl(RESULTS_DIR / "failures.jsonl"):
        if (row.get("method") == method and row.get("dataset") == dataset
                and row.get("graph_version") == v and row.get("retrieval") == r):
            rs = row
    if rs:
        hist = {int(k): int(vv) for k, vv in (rs.get("retry_hist") or {}).items()}
        n_retried = sum(vv for k, vv in hist.items() if k >= 1)
        done = max(1, rs.get("n_done") or len(rows))
        rr = n_retried / done
        rep.add(rr <= max_retry_rate, "重试率达标",
                f"{n_retried}/{done} = {rr:.1%}（阈值 {max_retry_rate:.0%}）", warn=rr > 0)

    # ── 10. trace 覆盖 ──
    tdir = trace_run_dir(method, dataset, v, r, sample)
    if tdir.exists():
        n_trace = len(list(tdir.glob("*.json"))) - (1 if (tdir / "_run.json").exists() else 0)
        cov = n_trace / max(1, len(rows))
        rep.add(cov >= 0.99, "trace 覆盖完整", f"{n_trace}/{len(rows)} = {cov:.1%}",
                warn=cov < 1.0)
    elif require_trace:
        rep.add(False, "trace 覆盖完整", f"缺 {rel_to_root(tdir)}（要求 trace）")
    else:
        rep.add(False, "trace 覆盖完整", "未开 trace（--no-trace 或老结果）", warn=True)

    # ── 11. 成本自洽：summary vs 逐题 ──
    sm = None
    for row in _load_jsonl(RESULTS_DIR / "summary.jsonl"):
        if (row.get("method") == method and row.get("dataset") == dataset
                and row.get("graph_version") == v and row.get("retrieval") == r):
            sm = row
    if not sm:
        rep.add(False, "summary 行存在", "缺 summary.jsonl 对应行", warn=True)
        return
    f2 = lambda x: int(x or 0)
    per_p = sum(f2(x.get("prompt_tokens")) for x in rows)
    per_c = sum(f2(x.get("completion_tokens")) for x in rows)
    per_e = sum(f2(x.get("embed_tokens")) for x in rows)
    sm_p, sm_c, sm_e = f2(sm.get("prompt_tokens")), f2(sm.get("completion_tokens")), f2(sm.get("embed_tokens"))
    # 逐题行与 query 段是同一批调用，量级应当一致（允许 5% 差）
    def close(a, b):
        base = max(1, max(a, b))
        return abs(a - b) / base <= 0.05
    rep.add(close(per_p, sm_p), "prompt token 自洽", f"逐题 {per_p:,} vs summary {sm_p:,}")
    rep.add(close(per_c, sm_c), "completion token 自洽", f"逐题 {per_c:,} vs summary {sm_c:,}")
    # ★ 逐题行只累加 **query 段**：build 段发生在逐题循环**之前**，是语料级的、
    #   不落在任何一题上（`run.py:643-644` 的 prompt/completion 同样只累加逐题行）。
    #   ⚠️ 所以这里必须拿 `query_embed_tokens` 比，**绝不能**拿 summary 的 `embed_tokens`
    #   —— 后者是 `build + query` 的总量（`run.py:702`，专供成本核算），
    #   而任何在 build 阶段嵌过语料的方法（factalign / hipporag / hypergraphrag / …）
    #   拿它去比都**必然 FAIL**。build 段单独报出来，供人核对量级。
    sm_eq = (f2(sm.get("query_embed_tokens"))
             if sm.get("query_embed_tokens") is not None else sm_e)
    sm_be = f2(sm.get("build_embed_tokens"))
    rep.add(close(per_e, sm_eq), "embed token 自洽",
            f"逐题 {per_e:,} vs query 段 {sm_eq:,}"
            + (f"（另有 build 段 {sm_be:,}，语料级、不落在单题上）" if sm_be else ""))
    # ── 12. 并发口径（动态并发下必须报"实际并发"，不能只报 --workers）──
    conc = sm.get("concurrency") or {}
    if conc.get("available"):
        amp = conc.get("client_over_gpu_amplification")
        rep.add(True, "实际并发已记录",
                f"avg={conc.get('avg')}（峰值 {conc.get('peak')}，配置上限 "
                f"{conc.get('workers_configured')}）"
                + (f"｜客户端/GPU 放大 {amp}×" if amp else ""))
        if amp and amp > 3:
            # ★ 信息行，不是 WARN：这个放大是**并发配置的必然结果**（N 个客户端挤 1 张卡），
            #   不是这次 run 的毛病，也**不可行动** —— 降 workers 只会拉长墙钟，
            #   `gpu_busy_seconds` 一点都不变。它每次都会触发，混进 WARN 计数
            #   只会让「WARN N」永远挂着、把真 WARN（缺 merged 语料）淹掉。
            #   信息本身必须留着：它决定了论文里报哪个时间字段。
            rep.note("时间口径（信息，非告警）",
                     f"客户端/GPU 放大 {amp}× —— 排队所致，属正常。"
                     f"论文/表格里**只能**用 `gpu_busy_seconds`（GPU 真实忙时），"
                     f"**不能**用客户端墙钟，否则会把成本夸大 {amp}×")
    else:
        rep.add(False, "实际并发已记录", "summary 无 concurrency（老结果或未走自适应闸）", warn=True)

    gb = sm.get("gpu_busy_seconds")
    gpu_info = sm.get("gpu") or {}
    if gb is None:
        rep.add(False, "GPU 时间可用", "summary 无 gpu_busy_seconds（无 nvidia-smi？）",
                warn=True if not require_gpu else False)
    else:
        # ⚠️ 不要拿 busy>0 当判据：GPU 可用但整轮几乎空闲时 busy 可以是 0.0（合法值）。
        # 只看「采样确实发生过 + 有显存读数」，否则会误报（第一版就踩了这个坑）。
        sampled = bool(gpu_info.get("available")) and int(gpu_info.get("samples") or 0) > 0
        mem = sm.get("gpu_peak_mem_mb")
        rep.add(sampled and mem is not None, "GPU 时间可用",
                f"busy={gb:.1f}s（采样 {gpu_info.get('samples')} 次，平均利用率 "
                f"{gpu_info.get('gpu_utilization_avg')}）峰值显存 {mem} MB"
                + ("" if sampled else " → 未采到样，成本表不能报 GPU 时间"),
                warn=not sampled)


def discover_runs() -> list[tuple[str, str, int, int, int]]:
    """遍历 results/ 找出所有 (method, dataset, sample, v, r)。"""
    out = []
    for row in _load_jsonl(RESULTS_DIR / "summary.jsonl"):
        try:
            out.append((row["method"], row["dataset"], int(row.get("sample") or 0),
                        int(row.get("graph_version") or 1), int(row.get("retrieval") or 1)))
        except Exception:                           # noqa: BLE001
            continue
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description="校验一次 run 是否按预期完成（ARI 检查项）")
    ap.add_argument("--method")
    ap.add_argument("--dataset")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--v", type=int, default=1)
    ap.add_argument("--r", type=int, default=1)
    ap.add_argument("--all", action="store_true", help="校验 results/summary.jsonl 里的全部 run")
    ap.add_argument("--max-fail-rate", type=float, default=0.05)
    ap.add_argument("--max-retry-rate", type=float, default=0.20)
    ap.add_argument("--strict", action="store_true", help="trace/GPU 也要求必须存在")
    ap.add_argument("--warn-only", action="store_true", help="即使有 FAIL 也返回 0（只报告）")
    args = ap.parse_args()

    rep = Report(warn_only=args.warn_only)
    if args.all:
        runs = discover_runs()
        if not runs:
            raise SystemExit("results/summary.jsonl 里没有 run（先跑一次 run.py）")
        print(f"[verify] 共 {len(runs)} 个 run")
        for m, ds, n, v, r in runs:
            verify_one(m, ds, n, v, r, rep, args.max_fail_rate, args.max_retry_rate,
                       require_trace=args.strict, require_gpu=args.strict)
    else:
        if not args.method or not args.dataset:
            raise SystemExit("给 --method/--dataset，或用 --all")
        verify_one(args.method, args.dataset, args.sample, args.v, args.r, rep,
                   args.max_fail_rate, args.max_retry_rate,
                   require_trace=args.strict, require_gpu=args.strict)
    return rep.dump(f"校验结果（{len(rep.rows)} 项）")


if __name__ == "__main__":
    sys.exit(main())

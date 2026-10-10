"""FactAlign-RAG 评测（可分开跑 + 断点续，支持所有方法）。

流程：建语料图（一次，缓存）→ 逐题检索+打分 → 每题结果立刻 append 到 jsonl。
中途停了，下次同命令自动跳过已完成的题继续。

方法:
  factalign                     —— 本方法：命题（自包含事实句）索引 + 开放槽命题查询
                                   （语料级 build 一次 + 逐题 query）
  standardrag                   —— 自带，**纯稠密**分块检索 top-5（无 BM25/RRF）
  lightrag                      —— 需 pip install lightrag（语料级建索引+逐题 query）
  hipporag                      —— 需 git clone 到 baselines/ + 装其 requirements（OpenIE 三元组
                                   知识图 + PPR 检索，语料级建索引+逐题 query，LLM/embedding 走本地 vLLM）
  hypergraphrag                 —— 需 git clone 到 baselines/ + pip install -r 其 requirements.txt
                                   （语料级建索引+逐题 query，LLM/embedding 走本地 vLLM）

  ircot                         —— 迭代检索基线（官方 prompt，max_steps 按数据集 2/2/4）

  新注册基线（源码未接入时打「不可用」后退出，不写结果行，见 methods/scaffold.py）:
  s2grag                        —— L2/L3 补位，语料级 build + 逐题 query；
                                   「官方源码放置区」填好即生效（见 EXPERIMENT_TABLE.md）
   另: selfask / proprag 已接入源码（见各自 adapter）。

  已退役: full / no_head —— 旧 HyperfactRAG 自带语料图路线（方法主线已改为 FactAlign-RAG）。
          消融请走 test_layer.py（带全部开关），run.py 只保留主方法的可比口径。
  已移除: graphrag / unirag / adaptive_rag / raptor / adarague / proh / structrag
          （不再进表、adapter 与入口一并删除）。

结果行字段: id / question / gold / pred / em / f1 /
           prompt_tokens / completion_tokens / llm_seconds / llm_calls（每题自己的 LLM 开销）

用法:
  python run.py --method factalign --dataset 2wiki --sample 300 --workers 8
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ══════════════════════════════════════════════════════════════════════════════
# ★ 必须最先执行：`--config` 的 env 注入要抢在 `import config` 之前
#   （config.py 顶层就把模型名/切块/并发读成常量；adapter 顶层也读 env）。
#   这里只用 stdlib 手写扫描，不 import 项目内任何模块。
# ══════════════════════════════════════════════════════════════════════════════
def _prescan_config(argv=None):
    """只扫 `--config` / `--method`（argparse 之前，不 import 项目内任何模块）。

    ★ 必须把 `--method` 也扫出来：`config_file.config_to_env` 只在
    `method` 非空时才注入 `methods.<m>.*`，而 `methods.*` 正是各方法的私有旋钮
    （`HI PORAG_OPENIE_TRIES` / `PROPRAG_TOP_DOCS` / `IRCOT_RETRIEVAL_COUNT` …）。
    曾经这里只扫 `--config` → `method=""` → **整个 `methods` 块被静默跳过**，
    配置文件里写了一堆方法参数却一个都没生效，而日志看起来一切正常
    （只是少了 `（method=xxx）` 那个后缀）。
    """
    a = list(sys.argv[1:] if argv is None else argv)
    cfg, method, dataset = None, "", ""
    for i, tok in enumerate(a):
        if tok == "--config" and i + 1 < len(a):
            cfg = a[i + 1]
        elif tok.startswith("--config="):
            cfg = tok.split("=", 1)[1]
        elif tok == "--method" and i + 1 < len(a):
            method = a[i + 1]
        elif tok.startswith("--method="):
            method = tok.split("=", 1)[1]
        elif tok == "--dataset" and i + 1 < len(a):
            dataset = a[i + 1]
        elif tok.startswith("--dataset="):
            dataset = tok.split("=", 1)[1]
    return cfg, method, dataset


_CONFIG_PATH, _CONFIG_METHOD, _PRESCAN_DATASET = _prescan_config()
_CONFIG_APPLIED = None
_CONFIG_PARAMS = None
if _CONFIG_PATH:
    from common.config_file import apply_config_file
    # ★ 不再逐行打印"注入成功的全部键"：顶层键对所有方法注入，而 `configs/*.json` 的顶层
    #   混了几十个方法私有键（IRCOT_* / FACTALIGN_* / HGR_* …）→ 跑 zeroshot 也会看到
    #   `topk=20`，被误读成"这次 run 用了它"。改由 `run_params.effective_params()`
    #   只打印**当前方法真正读的键**（并标注值来自配置文件/环境变量）。
    #   要看全量：`--dump-all-config` 或 `DSH_DUMP_ALL_CONFIG=1`。
    _dump_all = ("--dump-all-config" in sys.argv) or (
        os.environ.get("DSH_DUMP_ALL_CONFIG", "") not in ("", "0", "false", "False"))
    _CONFIG_APPLIED = apply_config_file(_CONFIG_PATH, method=_CONFIG_METHOD, verbose=_dump_all)
    if not _dump_all:
        # 只扫源码、不 import config/adapter，所以放在 env 注入之后、`import config` 之前是安全的。
        from run_params import effective_params, format_effective_lines
        _CONFIG_PARAMS = effective_params(
            _CONFIG_METHOD, cfg_raw=_CONFIG_APPLIED.get("raw"),
            kept_env=_CONFIG_APPLIED.get("kept_env"),
            dataset=_PRESCAN_DATASET or "hotpotqa")
        for _ln in format_effective_lines(_CONFIG_PARAMS, _CONFIG_METHOD, _CONFIG_PATH):
            print(_ln, flush=True)

from common.data import iter_questions                                     # noqa: E402
from common.metrics import exact_match_aliases, f1_score_aliases           # noqa: E402
from common.llm import (get_stats, reset_stats, phase, embed_batch,        # noqa: E402
                        snapshot, diff_snapshot, vllm_token_count, embed_len_stats,
                        set_default_phase)
from common.run_record import write_manifest, write_failures, error_kind   # noqa: E402
from common.result import graph_dir, result_path                          # noqa: E402
from common.paths import rel_to_root, trace_run_dir                       # noqa: E402
from common.trace import (start_trace, finish_trace, clear_trace,          # noqa: E402
                          trace_run_header, rounds_from_result,
                          normalize_result, set_trace_dir, prompt_dedup_stats)
from methods.standard_rag import build_chunks, query_chunks, STANDARDRAG_TOP_K  # noqa: E402
from methods import get_method, make_method                               # noqa: E402
from corpus import merged_contexts                                        # noqa: E402
from search import build_faiss_index                                      # noqa: E402
from config import (RESULTS_DIR, SAMPLES_DIR, CORPUS_DIR,                  # noqa: E402
                    EMBED_MAX_TOKENS)


# 语料级方法（建一次图/索引，逐题 query）
CORPUS_METHODS = {"standardrag", "factalign", "lightrag",
                  "hipporag", "hypergraphrag", "ircot",
                  # 新注册基线（同样语料级：build 一次 + 逐题 query）
                  "s2grag", "selfask", "proprag"}
# 语料级走「adapter.build()/query()」契约的方法（外部 baseline + 新注册基线）
# 早退分支 standardrag（不建图）与 factalign（自带语料图）在前面已拦掉
ADAPTER_METHODS = ("lightrag", "hipporag", "hypergraphrag", "ircot",
                   "s2grag", "selfask", "proprag")
# 其余外部 baseline 仍走 per-question run()
ALL_METHODS = ["standardrag", "factalign", "lightrag",
               "hypergraphrag", "hipporag", "ircot",
               "zeroshot",  # No-Retrieval 下界基线（per-question，无检索）
               # ↓ 新注册基线（源码接入前调用会被 scaffold.SourceNotReady 拦下，
               #   只打印「不可用」，不写结果行）
               "s2grag", "selfask", "proprag"]


def _result_complete(args, results_path) -> bool:
    """该 (method, dataset, v, r) 的结果是否已完整跑过（summary 有 complete=True 行）。

    完整 → 提示已有并跳过（不覆盖不重跑）；不完整（中断）→ 断点续。
    """
    if not results_path.exists():
        return False
    sp = RESULTS_DIR / "summary.jsonl"
    if not sp.exists():
        return False
    for line in sp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if (r.get("method") == args.method and r.get("dataset") == args.dataset
                and r.get("graph_version", 1) == args.graph_version
                and r.get("retrieval", 1) == args.retrieval
                and r.get("complete")):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=ALL_METHODS)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--native-generator", action="store_true",
                    help="外部 baseline（lightrag 等）用各自原生态生成器；"
                         "默认统一走本项目 generate_answer（prompt 与自带方法同口径）")
    ap.add_argument("--graph-version", type=int, default=1,
                    help="第几次建图（v 号）：目录 data/{method}/{dataset}/v{n}/；"
                         "换图参数想保留旧图时开新 v")
    ap.add_argument("--retrieval", type=int, default=1,
                    help="同一张图上的第几次验证检索（r 号）：结果文件 "
                         "results/{method}/{dataset}/v{n}r{m}.jsonl；"
                         "换查询参数（prompt/native-generator 等）时开新 r，互不覆盖")
    # ── factalign（本方法）旋钮：默认 = 主方法口径（EXPERIMENT_PLAN §三 第 5 格 q_stmt_he）──
    # ★ **全部消融开关已从 run.py 移除**（消融放 test_layer.py，那边有独立的
    #   `--recall`（默认 rrf）/`--decompose_recall`/`--rerank`/`--gen_top` 等）：
    #     `--query-form desc|inst` / `--no-decompose` / `--decompose-recall chunk` /
    #     `--single-recall rrf|bm25|dense`（含 **rrf 融合检索**）/
    #     `--factalign-rerank none|dense`（池内精排消融）/ `--hope`（一跳扩展）/
    #     `--ppr-w`（池内 PPR）。
    #   代码里的实现仍在（rerank 默认 hybrid、hope/ppr_w 默认 0 = 关闭），只是不再从
    #   命令行调 —— run.py 只跑主方法。
    #   保留的 `--decomp-top-k` / `--decomp-steps` / `--gen-top` 不是消融：
    #   前两个是主方法的拆解深度（`decomp_top_k`×`decomp_steps`=20 就是证据预算的来源），
    #   第三个是生成端条数覆盖（0 = 用 FACTALIGN_EVIDENCE_PROPS=20）。
    ap.add_argument("--decomp-top-k", type=int, default=5, help="factalign 每步检索条数")
    ap.add_argument("--decomp-steps", type=int, default=4, help="factalign 最多拆解步数")
    ap.add_argument("--gen-top", type=int, default=0,
                    help="factalign 喂生成的命题条数（0=用 FACTALIGN_EVIDENCE_PROPS=20）")
    # ── 目录/留档（见 common/paths.py）──
    ap.add_argument("--config", default=_CONFIG_PATH,
                    help="参数配置文件（configs/*.json）；键→环境变量，须最先注入。"
                         "优先级：CLI/环境变量 > 配置文件 > 代码默认值。"
                         "注意：本参数已在启动时生效（见文件顶部预扫描）")
    ap.add_argument("--trace", dest="trace", action="store_true", default=True,
                    help="逐题 trace（默认开）：traces/{method}/{dataset}/v{n}r{m}/{qid}.json "
                         "留每次 LLM 调用的完整 prompt + 原始输出 + 每轮检索的 chunks/实体")
    ap.add_argument("--no-trace", dest="trace", action="store_false",
                    help="关掉逐题 trace（只留 logs/ 文本日志与 results/ 指标）")
    ap.add_argument("--no-trace-prompt", dest="trace_prompt", action="store_false",
                    default=True,
                    help="trace 里 system/assistant 消息只存 sha256（默认存原文），"
                         "体积约降到 1/4，但无法从 trace 单独复现完整请求")
    args = ap.parse_args()
    if _CONFIG_APPLIED is None and args.config:
        # 走到这里说明 cfg 的键都已被真实环境变量覆盖（或 --config 是命令行后加的）
        print(f"[config] 提示：--config {args.config} 的环境变量在 import config 之前已注入；"
              f"若本次没看到注入日志，说明同名环境变量已存在（环境变量优先）")

    # 数据先查（放在建日志之前，免得白建两个空日志文件）：
    #   data/samples/{数据集名}_{样本数}.json   ← 数据集名里**不带**样本数
    if not list(SAMPLES_DIR.glob(f"{args.dataset}_*.json")):
        print(f"没找到数据 {SAMPLES_DIR}/{args.dataset}_*.json")
        have = sorted(p.name for p in SAMPLES_DIR.glob("*.json"))
        if have:
            print(f"  data/samples/ 里现有：{'、'.join(have)}")
        print("  提示：--dataset 只写数据集名（hotpotqa / 2wiki / musique / popqa / triviaqa），"
              "样本数由 --sample 决定。\n"
              f"        10 题 smoke： python run.py --method {args.method} "
              f"--dataset hotpotqa --sample 10 --graph-version 99\n"
              "        500 题全量： python run.py --method "
              f"{args.method} --dataset hotpotqa --sample 500\n"
              "  题集没生成就跑： python tools/build_eval_samples.py（500 题）"
              " / python tools/make_smoke_samples.py（每数据集前 10 题）")
        return

    # 日志按阶段分文件（终端同步双写）：
    #   logs/{method}/{ds}_{n}_v{n}r{m}_build_{时间戳}.log  ← **只有真建/重建图时才生成**
    #   logs/{method}/{ds}_{n}_v{n}r{m}_query_{时间戳}.log  ← 检索段，开头标明"用的哪张图"
    #
    # ★ P0：先探 vLLM 服务端配置（prefix caching / max_model_len / 版本 / 容量）。
    #   必须在开日志之前，这样日志头里 `max_model_len` 不再是 None、并写明
    #   prefix caching 到底开没开（它直接决定 prefill 耗时与 GPU 时间口径）。
    _server_snap = None
    try:
        from common.server_probe import server_snapshot, write_server_config, summary_lines
        _server_snap = server_snapshot()
        _sp = write_server_config(_server_snap)
        _ok = (f"{_server_snap.get('chat', {}).get('served_model') or '?'}"
               f"(max_len={_server_snap.get('chat', {}).get('max_model_len')})")
        print(f"[server] 服务端配置已探测：{_ok} | prefix caching="
              f"{_server_snap.get('prefix_caching')}"
              + (f" → {rel_to_root(_sp)}" if _sp else "（落盘失败）"), flush=True)
        for line in summary_lines(_server_snap):
            print(line, flush=True)
    except Exception as e:                       # noqa: BLE001
        print(f"[server] ⚠ 服务端配置探测失败（不影响实验）："
              f"{type(e).__name__}: {e}", flush=True)

    from common.log_setup import start_log, switch_log
    tee = start_log(args.method, args.dataset, args.sample,
                    extra=f"v{args.graph_version}r{args.retrieval}", segments=True,
                    server_snapshot=_server_snap)

    # 逐条 LLM 调用审计（**默认开**；`LLM_CALL_LOG=0` 关）→ 落到本次 run 的 logs/ 目录。
    # ★ 两段**分开存**：`..._build.llm_calls.jsonl` / `..._query.llm_calls.jsonl`。
    #   为什么必须拆：建图跑完、作答阶段挂掉是常态（500 题里任意一题炸、进程被杀、显存打满）。
    #   混在一个文件里时，事后无法回答"这 12 万条里哪些是建图花的、建图到底建成了没有"。
    #   文件名与两段文本日志（`_build_*.log` / `_query_*.log`）同一时间戳，便于互相对照。
    _llm_call_log = {}

    def _use_call_log(stage: str) -> None:
        """把逐条调用审计切到该阶段的文件（best-effort；失败不影响实验）。"""
        try:
            from common.llm import LLM_CALL_LOG_ON, set_call_log_path
            if not LLM_CALL_LOG_ON:
                return
            _lm = getattr(tee, "meta", None) or {}
            _parts = [f"{args.dataset}_{args.sample}"]
            if _lm.get("extra"):
                _parts.append(str(_lm["extra"]))
            _parts += [stage, str(_lm.get("stamp", ""))]
            p = Path(_lm.get("logdir") or (ROOT / "logs")) / ("_".join(_parts) + ".llm_calls.jsonl")
            set_call_log_path(p)
            _llm_call_log[stage] = p
        except Exception as _e:                      # noqa: BLE001
            print(f"[llm] 逐条调用日志路径设置失败（不影响实验）：{type(_e).__name__}: {_e}",
                  flush=True)

    # 建图段（本方法里 WS / factalign / adapter 三条分支的 build 都在下面）→ build 文件
    _use_call_log("build")

    items = list(iter_questions(args.dataset, args.sample))
    results_path = result_path(args.method, args.dataset, args.graph_version, args.retrieval)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # 结果已存在且完整（summary 里有该 v+r 的 complete=True 行）→ 提示已有，不覆盖不重跑
    if _result_complete(args, results_path):
        print(f"[{args.method}] v{args.graph_version}r{args.retrieval} 结果已存在且完整："
              f"{results_path}\n（不覆盖；要重跑请换 --retrieval 开新 r 或删掉该文件）")
        return

    # 断点续：读已完成题 id
    done_ids = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    pending = [it for it in items if it["id"] not in done_ids]
    print(f"{args.method} @ {args.dataset}: 已完成 {len(done_ids)}，待跑 {len(pending)}，{args.workers} 并发")

    # ★ 逐题 trace 的 run 头 + 共享 prompt 目录（重复的长 prompt 只存一份）
    if args.trace:
        try:
            # ★ 样本文件路径**由解析函数给出**，不再按 `{dataset}_{sample}.json` 拼：
            #   --sample 0 或文件名不严格等于样本数时，拼出来的是一次 run 里根本不存在的
            #   文件（trace 头 = 假凭证）。这里与真正读数据走同一条解析逻辑。
            try:
                from common.data import sample_path
                _sample_file = rel_to_root(sample_path(args.dataset, args.sample))
            except Exception:                       # noqa: BLE001
                _sample_file = ""
            trace_run_header(args.method, args.dataset, args.graph_version, args.retrieval,
                             sample=args.sample,
                             extra={"config_file": args.config,
                                    "keep_prompt": args.trace_prompt,
                                    "sample_file": _sample_file})
            set_trace_dir(trace_run_dir(args.method, args.dataset, args.graph_version,
                                        args.retrieval, args.sample))
            print(f"[trace] 逐题留档 → traces/{args.method}/{args.dataset}/"
                  f"v{args.graph_version}r{args.retrieval}_n{args.sample}/"
                  f"（重复 prompt 去重；原文件在 prompts/ 子目录）", flush=True)
        except Exception as e:                       # noqa: BLE001
            print(f"[trace] ⚠ 初始化 trace 失败（不影响实验）：{e}", flush=True)
            args.trace = False

    def record_build_tokens(before: dict, after: dict):
        """建图结束立即把 vLLM 差值写盘（不等进程结束——崩溃也不丢）。

        results/build_tokens.jsonl 每行：{ts, method, dataset, sample,
        graph_version, chat_prompt, chat_completion, embed_prompt}。
        """
        row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
               "method": args.method, "dataset": args.dataset,
               "sample": args.sample, "graph_version": args.graph_version}
        ok = False
        for name, key_p, key_c in (("chat", "chat_prompt", "chat_completion"),
                                   ("embed", "embed_prompt", None)):
            b, a = (before or {}).get(name), (after or {}).get(name)
            if not (b and a and b["prompt"] is not None and a["prompt"] is not None):
                row[key_p] = None
                row[key_c] = None
                continue
            dp = a["prompt"] - b["prompt"]
            dc = (a["completion"] or 0) - (b["completion"] or 0)
            row[key_p] = dp
            row[key_c] = dc if key_c else None
            ok = ok or (dp or dc)
        if ok:
            with open(RESULTS_DIR / "build_tokens.jsonl", "a",
                      encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[build-tokens] 本次索引构建 vLLM 消耗（chat 与 embed 分开）: "
                  f"chat: prompt={row['chat_prompt']:,} completion="
                  f"{row['chat_completion']:,} | "
                  f"embed: prompt={row['embed_prompt']:,} -> "
                  f"results/build_tokens.jsonl", flush=True)

    # ── 方法准备 ──
    graph = chunks = chunk_vecs = chunk_faiss = None
    method_obj = None
    graph_used = "（无图：本方法不使用语料图）"   # 检索段日志里标明"这次检索用的哪张图"
    graph_reused = False                        # 图已存在 → 本次只是加载，不生成建图 log
    _graph_dir_obj = None                       # 本次用的图目录（Path），进 manifest/_build.json
    reset_stats()
    try:
        from common.llm import reset_concurrency
        reset_concurrency()          # 并发统计清零（建图/检索整轮都算在这一次 run 里）
    except Exception:                # noqa: BLE001
        pass
    vllm_before = vllm_token_count()  # 建图前的 vLLM 累计 token
    # ★ GPU 时间采样（SIGMOD 成本表要 GPU busy 秒，不是客户端等待时间）
    #   采样按「整次 run」算，最后按 build/query 的墙钟占比拆分（见下方 summary_row）
    from common.gpu import GPUProbe
    _gpu = GPUProbe(interval=float(os.environ.get("GPU_PROBE_SEC", "1.0"))).start()
    _run_t0 = time.time()
    if _gpu.available:
        print(f"[gpu] 采样已启动：{[g['name'] for g in _gpu.gpus.values()]}"
              f"（每 {_gpu.interval}s；峰值显存/利用率会写进 summary.jsonl）", flush=True)
    import atexit
    atexit.register(_gpu.stop)          # 早退/异常路径也把采样线程收掉（daemon，不影响退出）
    if args.method == "standardrag":
        # 不建图：分块 + **纯稠密** top-STANDARDRAG_TOP_K（关 BM25、关 RRF）
        # （`chunk_dense` 已删除 —— 它与 standardrag 只差 k，见 PARAMS_ALIGNED.md §3.8）
        # ★ 语料 = **语料级合并语料**（`merged_contexts`，按 (title,text) 去重），与其余
        #   10 个方法**同一份**。此前是「逐题那 10 段」拼起来（500 题 × 10 = 5000 段，不去重），
        #   检索池比别的方法大/小一个数量级且混入大量重复段（2wiki 实测 5000 段里 1644 段重复）
        #   → 与别的 baseline **检索难度不可比**。改后与建图类方法同池同分块。
        chunks = build_chunks(merged_contexts(args.dataset, args.sample), dataset=args.dataset)
        with phase("build"):
            chunk_vecs = embed_batch([c[2] for c in chunks])  # 分块只 embed 一次，逐题复用
            chunk_faiss = build_faiss_index(chunk_vecs) if chunks else None
    elif args.method == "factalign":
        # 本方法：语料图（命题）+ 窗口向量 + 实体描述向量，建一次逐题复用
        _cdir = CORPUS_DIR / args.dataset / f"v{args.graph_version}"
        graph_used = f"语料图 {_cdir}"
        _graph_dir_obj = _cdir
        if _cdir.exists():
            print(f"[factalign] v{args.graph_version} 语料图已建：{_cdir}（直接加载，不重建）",
                  flush=True)
            graph_reused = True
        # ★ 只传**主方法**旋钮；`rerank` / `hope` / `ppr_w` 等消融开关不再从命令行传，
        #   取 FactAlign 的主口径默认值（rerank="hybrid"，hope=0，ppr_w=0.0）。
        method_obj = make_method(
            "factalign",
            decomp_top_k=args.decomp_top_k,
            decomp_steps=args.decomp_steps,
            gen_top=args.gen_top,
        )
        method_obj.prepare_for(args.dataset, args.sample, args.graph_version)
        # ★ set_default_phase：`phase("build")` 是**线程局部**的，对 adapter 内部
        #   `ThreadPoolExecutor` 里的 worker 无效 → 那些 worker 的 LLM 调用会落进
        #   **query 桶**（`_default_phase or "query"`），建图段就显示 chat prompt=0。
        #   见 common/llm.py:set_default_phase 的说明（proprag @2wiki 实测命中）。
        _prev_phase = set_default_phase("build")
        try:
            with phase("build"):
                method_obj.build(merged_contexts(args.dataset, args.sample))
        finally:
            set_default_phase(_prev_phase)
    elif args.method in ADAPTER_METHODS:
        # 语料级：LightRAG / HippoRAG / HyperGraphRAG / IRCoT / PropRAG / S2G-RAG 等适配器
        # 建一次索引（与 factalign 同语料），逐题 query；目录 data/{method}/{dataset}/v{n}/
        # （v = 第几次建图），已存在则直接加载（断点续）；同一 v 内的参数变化由各 adapter
        # 的 marker 判断重建
        method_obj = get_method(args.method)
        if args.native_generator:
            method_obj.native_generator = True  # 端到端原生态对比用
        # 需要知道数据集名的方法（IRCoT 的 max_steps、s2grag 的单跳口径等按数据集官方值）
        if hasattr(method_obj, "dataset"):
            method_obj.dataset = args.dataset
        # ★ 同 factalign 分支：adapter 内部线程池（proprag/hipporag/factalign 的抽取池、
        #   lightrag 的 LLM 池等）里的 worker **没有 phase 上下文** → `_cur_phase()`
        #   落到 `_default_phase or "query"` → 建图期的 LLM 调用全记进 **query 桶**，
        #   建图段就显示 `chat prompt=0 completion=0`（proprag @2wiki 实测命中）。
        #   hipporag/hypergraphrag 各自 set/reset 过，但那容易漏 → 这里统一兜住。
        _prev_phase = set_default_phase("build")
        try:
            with phase("build"):
                gr_dir = str(graph_dir(args.method, args.dataset, args.graph_version))
                graph_used = f"{args.method} 索引 {gr_dir}"
                _graph_dir_obj = Path(gr_dir)
                if Path(gr_dir).exists():
                    print(f"[{args.method}] v{args.graph_version} 图已建：{gr_dir}"
                          f"（完整性由 adapter 校验，完整则直接加载不重建）", flush=True)
                    graph_reused = True
                method_obj.build(merged_contexts(args.dataset, args.sample), working_dir=gr_dir)
                # ★ 有些 adapter 的 build() 是**纯内存、从不落盘**的（如 s2grag 的
                #   CorpusIndex：只有切块 + BM25 打分表），而写"建图凭证"的
                #   `write_build_meta.resolve_graph_dirs()` 只认**已存在**的目录 →
                #   目录不存在就会被静默跳过，验收里表现为「build 四件套齐全 0/4」。
                #   这里在真正建完之后确保目录存在，让 config/env/stats/checksums 有地方落。
                #   （对本来就落盘的 adapter —— hipporag / hypergraphrag 等 —— 是 no-op。）
                #   注意：必须在上面 `Path(gr_dir).exists()` 那次判定**之后**，
                #   否则会把"本次是否复用已有图"判成复用。
                Path(gr_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[{args.method}] 不可用：{e}")
            return
        finally:
            set_default_phase(_prev_phase)
    else:
        method_obj = get_method(args.method)  # 外部 baseline 实例
    # 建图结束：vLLM 差值立即落盘（崩溃也不丢 build token）
    record_build_tokens(vllm_before, vllm_token_count())
    _bs = get_stats()["build"]
    # 判定"这次到底建没建图"：图目录不存在 → 一定建了；存在但统计显示消耗/耗时可观
    # （adapter 自己判断要重建）→ 也算建了。只在真建图时才留 build log。
    _built = (not graph_reused) or _bs["embed_tokens"] > 0 \
        or _bs["completion_tokens"] > 0 or _bs["total_seconds"] >= 30
    # ★ 建图凭证：把 config/env/stats(实体·关系计数)/checksums 写进图目录旁边
    #   （建图类方法要求；标准 RAG 这类没有图的会自己跳过）
    try:
        from tools.write_build_meta import write_build_meta_for_run
        _bm = write_build_meta_for_run(args.method, args.dataset, args.sample,
                                       args.graph_version, args.retrieval,
                                       method_obj=method_obj,
                                       config_file=args.config, built=_built)
        if _bm:
            print(f"[build-meta] 复现凭证已写：{rel_to_root(_bm)} 等 "
                  f"（config/env/stats/checksums）", flush=True)
    except Exception as e:                       # noqa: BLE001
        print(f"[build-meta] ⚠ 写索引构建凭证失败（不影响实验）：{e}", flush=True)
    # ★ 建图口径落进 trace（`_build.json`）：图目录 + **新建/复用** + 产物哈希 + 建图段 token。
    #   逐题 trace 只覆盖 query 期；复用旧图时本次根本没有建图调用，若不记这一份，
    #   事后无法回答"论文成本表里的 build token / 这张图，是哪一次 run 造出来的"。
    _graph_rec = None
    try:
        from common.paths import trace_build_path
        from common.run_record import graph_record, write_build_record
        _graph_rec = graph_record(args.method, args.dataset, _graph_dir_obj,
                                  reused=graph_reused, built_now=_built,
                                  tokens={k: _bs.get(k) for k in
                                          ("prompt_tokens", "completion_tokens",
                                           "embed_tokens", "chat_calls",
                                           "total_seconds")})
        if args.trace:
            _bp = trace_build_path(args.method, args.dataset, args.graph_version,
                                   args.retrieval, args.sample)
            if write_build_record(_bp, _graph_rec):
                print(f"[trace] 建图口径 → {rel_to_root(_bp)}"
                      f"（{'本次新建' if _built else '复用已有图'}）", flush=True)
    except Exception as e:                       # noqa: BLE001
        print(f"[trace] ⚠ 建图口径留档失败（不影响实验）：{e}", flush=True)
    print(f"[build] 索引构建段结束：{_bs['total_seconds']:.1f}s"
          f"（chat {_bs['chat_seconds']:.1f}s / embed {_bs['embed_seconds']:.1f}s）"
          f" | chat prompt={_bs['prompt_tokens']:,} completion={_bs['completion_tokens']:,}"
          f" | embed={_bs['embed_tokens']:,} token"
          f"{'' if _built else '（本次未构建，直接复用已有索引）'}")
    try:
        from common.log_setup import skip_build_log, note_to_log
        _graph_note = f"=== [{args.method}] 本次检索使用的图: {graph_used} ==="
        if not _built:
            skip_build_log()          # 图已存在 → 不生成 build log，整份写到 query log
            note_to_log(_graph_note)  # 已切段（switch_log 会 early-return）→ 直接写本段
        switch_log("query", notes=[_graph_note])
    except Exception as e:
        print(f"[log] 切检索段日志失败（不影响实验）：{e}")
    # ★ 建图段到此结束（成败都算）：逐条调用审计**切到 query 文件**，从下一行起
    #   所有 LLM 调用（逐题检索 + 作答）都进 `..._query.llm_calls.jsonl`，与建图段彻底分开。
    _use_call_log("query")

    # ── 可追溯性记录：重试/失败统计（论文 Limitations 与复现性要用）──
    _rec_lock = threading.Lock()
    _retry_hist = {0: 0, 1: 0, 2: 0}      # 该题实际重试次数 → 题数（由统计桶的 chat_calls 推出）
    _errors = []                          # [{id, retries, kind, msg}]
    _error_kinds = {}

    def _record(rec):
        with _rec_lock:
            _retry_hist[rec["retries"]] = _retry_hist.get(rec["retries"], 0) + 1
            if rec.get("msg"):
                _errors.append({"id": rec["id"], "retries": rec["retries"],
                                "kind": rec["kind"], "msg": rec["msg"][:300]})
                _error_kinds[rec["kind"]] = _error_kinds.get(rec["kind"], 0) + 1

    def answer_one(item):
        """每题最多 3 次重试；用线程内统计快照差值算单题 token/LLM 时间。"""
        fails = 0                      # 这题实际失败次数（= 重试次数，直接数，别用 chat_calls 反推
                                       # —— 一题内部本来就有多次 LLM 调用，反推会全错）
        tr = None
        if args.trace:
            # ★ 逐题 trace：本线程内所有 LLM 调用自动挂到这道题上（llm.py 出口钩子）
            tr = start_trace(method=args.method, dataset=args.dataset, qid=item["id"],
                             sample=args.sample, graph_version=args.graph_version,
                             retrieval=args.retrieval, question=item["question"],
                             gold=item.get("answer", ""),
                             aliases=item.get("answer_aliases", []),
                             keep_prompt=args.trace_prompt)
        for attempt in range(3):
            before = snapshot()
            try:
                if args.method == "standardrag":
                    # 纯稠密：关 BM25、关 RRF，取 top-STANDARDRAG_TOP_K
                    # ★ dataset 必须传：standardrag 是**函数式**入口（不是 adapter 类），
                    #   拿不到 self.dataset；不传的话 triviaqa 会静默退回统一作答 prompt。
                    r = query_chunks(item["question"], chunks, STANDARDRAG_TOP_K,
                                     vecs=chunk_vecs, faiss_index=chunk_faiss,
                                     dense_only=True, dataset=args.dataset)
                elif args.method == "factalign":
                    r = method_obj.query(item["question"])
                elif args.method in ADAPTER_METHODS:
                    r = method_obj.query(item["question"])
                else:
                    r = method_obj.run(item["question"], item["contexts"])
            except Exception as e:
                # 前两次失败：这次尝试的花费**不计**（下次尝试会重取快照）；
                # 第 3 次仍失败：把**最后一次尝试的花费**记进该题（用户口径），
                # 答案写成 <ERROR: ...>，不再抛给上层用 0 token 的错误行。
                fails += 1
                if tr is not None:
                    tr.note_error(f"{type(e).__name__}: {e}", retries=fails)
                if attempt == 2:
                    usage = diff_snapshot(before, snapshot())
                    print(f"[{args.method}] 题目 {item['id']} 重试 3 次仍失败，"
                          f"记最后一次花费：{e}", flush=True)
                    r = {"answer": f"<ERROR: {e}>", "entities": [],
                         "evidence": [], "evidence_locs": []}
                    _record({"id": item["id"], "retries": 3,
                             "kind": error_kind(str(e)), "msg": str(e)})
                    _emit_trace(tr, item, r)
                    return item, r, usage
                print(f"  重试 {attempt + 1}/3: {e}", flush=True)
                time.sleep(1)
                continue
            usage = diff_snapshot(before, snapshot())
            _record({"id": item["id"], "retries": fails, "kind": "", "msg": ""})
            _emit_trace(tr, item, r)
            return item, r, usage

    def _emit_trace(tr, item, r):
        """把这道题的 trace 落盘（含每轮检索）并写索引行。trace 出错不影响实验。"""
        if tr is None:
            return
        try:
            nr = normalize_result(r)
            for rd in rounds_from_result(r, question=item["question"]):
                tr.add_round(round_idx=rd.get("round"), query_used=rd.get("query_used", ""),
                             chunks=rd.get("chunks"), entities=rd.get("entities"),
                             paths=rd.get("paths"), sub_questions=rd.get("sub_questions"),
                             extra={k: v for k, v in rd.items()
                                    if k not in ("round", "query_used", "chunks", "entities",
                                                 "paths", "sub_questions")})
            aliases = item.get("answer_aliases", [])
            em = exact_match_aliases(nr["answer"], item.get("answer", ""), aliases)
            f1 = f1_score_aliases(nr["answer"], item.get("answer", ""), aliases)
            tr.set_final(answer=nr["answer"], evidence=nr["evidence"],
                         evidence_locs=nr["evidence_locs"],
                         extra={"raw_keys": nr["_raw_keys"]})
            finish_trace(tr, args.method, args.dataset, args.graph_version,
                         args.retrieval, sample=args.sample, em=em, f1=f1, answer=nr["answer"])
        except Exception as e:                       # noqa: BLE001
            print(f"[trace] ⚠ 写 {item.get('id')} 的 trace 失败（不影响实验）：{e}", flush=True)
        finally:
            clear_trace()

    lock = threading.Lock()
    fout = open(results_path, "a", encoding="utf-8")

    def write_row(item, r, usage):
        aliases = item.get("answer_aliases", [])
        ub, uq = usage.get("build", {}), usage.get("query", {})
        # ★ 成本要能分开算（SIGMOD）：chat 的 prefill/decode token、embed token、
        #   chat 挂钟 / embed 挂钟、建图段 / 检索段，各自独立列，不再只给一个"总计"。
        #   注意 llm_seconds 旧含义只含 chat → 这里改成**chat+embed 全部挂钟**，
        #   并额外给 *_seconds 明细，旧字段名保留（下游脚本不用改）。
        row = {"id": item["id"], "question": item["question"], "gold": item["answer"],
               "pred": r["answer"],
               "em": exact_match_aliases(r["answer"], item["answer"], aliases),
               "f1": f1_score_aliases(r["answer"], item["answer"], aliases),
               "entities": r.get("entities", []),          # 题目中提取的实体
               "evidence": r.get("evidence", []),          # 检索回来的 top-k 证据
               "evidence_locs": r.get("evidence_locs", []),
               # ── token（prefill / decode / embed 三类分开）──
               "prompt_tokens": ub.get("prompt_tokens", 0) + uq.get("prompt_tokens", 0),
               "completion_tokens": ub.get("completion_tokens", 0) + uq.get("completion_tokens", 0),
               "embed_tokens": ub.get("embed_tokens", 0) + uq.get("embed_tokens", 0),
               # ── 时间（chat / embed / 建图 / 检索 分开）──
               "chat_seconds": round(ub.get("chat_seconds", 0) + uq.get("chat_seconds", 0), 3),
               "embed_seconds": round(ub.get("embed_seconds", 0) + uq.get("embed_seconds", 0), 3),
               "build_seconds": round(ub.get("chat_seconds", 0) + ub.get("embed_seconds", 0), 3),
               "query_seconds": round(uq.get("chat_seconds", 0) + uq.get("embed_seconds", 0), 3),
               "llm_seconds": round(ub.get("chat_seconds", 0) + uq.get("chat_seconds", 0)
                                    + ub.get("embed_seconds", 0) + uq.get("embed_seconds", 0), 2),
               # ── 调用次数 ──
               "llm_calls": ub.get("chat_calls", 0) + uq.get("chat_calls", 0),
               "embed_calls": ub.get("embed_calls", 0) + uq.get("embed_calls", 0)}
        with lock:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

    def run_pending():
        done = len(done_ids)
        total = len(items)
        zero_usage = {"build": {"prompt_tokens": 0, "completion_tokens": 0, "embed_tokens": 0,
                                "chat_seconds": 0, "embed_seconds": 0,
                                "chat_calls": 0, "embed_calls": 0},
                      "query": {"prompt_tokens": 0, "completion_tokens": 0, "embed_tokens": 0,
                                "chat_seconds": 0, "embed_seconds": 0,
                                "chat_calls": 0, "embed_calls": 0}}

        def error_row(e):
            """3 次重试仍失败：记 0 分行，别让一道题杀死整批。"""
            r = {"answer": f"<ERROR: {e}>", "entities": [],
                 "evidence": [], "evidence_locs": []}
            return r, dict(build=dict(zero_usage["build"]),
                           query=dict(zero_usage["query"]))

        def report(_item, r):
            nonlocal done
            done += 1
            em = exact_match_aliases(r["answer"], _item["answer"], _item.get("answer_aliases", []))
            print(f"[{args.method}][{done}/{total}] EM={em} | gold={_item['answer']!r} pred={r['answer'][:40]!r}",
                  flush=True)

        if args.workers <= 1:
            for item in pending:
                try:
                    _item, r, u = answer_one(item)
                except Exception as e:
                    _item, r, u = item, *error_row(e)
                    _record({"id": _item["id"], "retries": 3,
                             "kind": error_kind(str(e)), "msg": str(e)})
                    print(f"[{args.method}] 题目 {_item['id']} 3 次重试失败：{e}",
                          flush=True)
                write_row(_item, r, u)
                report(_item, r)
        else:
            ex = ThreadPoolExecutor(max_workers=args.workers)
            futures = {ex.submit(answer_one, it): it for it in pending}
            try:
                for f in as_completed(futures):
                    _item = futures[f]
                    try:
                        _item, r, u = f.result()
                    except Exception as e:
                        r, u = error_row(e)
                        _record({"id": _item["id"], "retries": 3,
                                 "kind": error_kind(str(e)), "msg": str(e)})
                        print(f"[{args.method}] 题目 {_item['id']} 3 次重试失败：{e}",
                              flush=True)
                    write_row(_item, r, u)
                    report(_item, r)
            except KeyboardInterrupt:
                # 中断：取消未开始的题，只等在途的（≤workers 道），不再等全部跑完
                ex.shutdown(wait=False, cancel_futures=True)
                raise
            ex.shutdown(wait=True)

    # 外部 baseline 若未装/未实现会抛错，先试跑一题验证，再批量
    if args.method not in CORPUS_METHODS and pending:
        try:
            _item, r, u = answer_one(pending[0])
            write_row(_item, r, u)
            pending = pending[1:]
        except Exception as e:
            print(f"[{args.method}] 不可用：{e}")
            fout.close()
            return

    try:
        run_pending()
    except KeyboardInterrupt:
        print("\n已中断，写当前汇总（重跑同命令断点续）")
    finally:
        fout.close()

    # 汇总（正常跑完或中断都会执行；容忍中断残留的半行）
    rows = []
    for l in results_path.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    n = len(rows)
    if n == 0:
        print("没有结果")
        # 一条结果都没有也要留痕（否则"跑了但没出数"在记录里查不到）
        write_manifest(args.method, args.dataset, args.sample,
                       args.graph_version, args.retrieval, 0, False,
                       {"status": "no_results", "graph": _graph_rec})
        write_failures(args.method, args.dataset, args.sample, args.graph_version,
                       args.retrieval, len(items), 0, _retry_hist, _errors, _error_kinds)
        return
    em = sum(r["em"] for r in rows) / n
    f1 = sum(r["f1"] for r in rows) / n
    pt = sum(r.get("prompt_tokens", 0) for r in rows)
    ct = sum(r.get("completion_tokens", 0) for r in rows)
    llm_s = sum(r.get("llm_seconds", 0) for r in rows)
    calls = sum(r.get("llm_calls", 0) for r in rows)
    s = get_stats()
    b, q = s["build"], s["query"]
    print(f"\n{args.method} @ {args.dataset}: n={n} EM={em:.4f} F1={f1:.4f}")
    print(f"逐题累加(结果行): prompt={pt:,} completion={ct:,}, {llm_s:.1f}s, calls={calls}")
    # chat(LLM) 与 embed 分开报 —— 两条链路的 token/耗时完全不同量级，混在一行看不出钱花在哪
    # 阶段名用"索引构建"而不是"建图"：这一桶同时覆盖真建图的（hipporag / hypergraphrag /
    # lightrag）和只建检索索引的（s2grag 纯 BM25、standardrag 等），叫"建图"会误导。
    print(f"索引构建 · chat : prompt={b['prompt_tokens']:,} completion={b['completion_tokens']:,} "
          f"| {b['chat_calls']} 次 | {b['chat_seconds']:.1f}s")
    print(f"索引构建 · embed: prompt={b['embed_tokens']:,} | {b['embed_calls']} 次 | "
          f"{b['embed_seconds']:.1f}s")
    print(f"检索 · chat : prompt={q['prompt_tokens']:,} completion={q['completion_tokens']:,} "
          f"| {q['chat_calls']} 次 | {q['chat_seconds']:.1f}s")
    print(f"检索 · embed: prompt={q['embed_tokens']:,} | {q['embed_calls']} 次 | "
          f"{q['embed_seconds']:.1f}s")
    print(f"合计: chat {b['prompt_tokens'] + b['completion_tokens'] + q['prompt_tokens'] + q['completion_tokens']:,} token"
          f" | embed {b['embed_tokens'] + q['embed_tokens']:,} token"
          f" | 耗时 {s['total']['total_seconds']:.1f}s")
    # ★ embedder 单条长度**实测**：直接给出"服务端 --max-model-len 至少要多大"的答案，
    #   不用再靠推导（切块 400 / 摘要 500 都是上界，不是实测）。>2048 会特别标出。
    try:
        _el = embed_len_stats()
        if _el["n"]:
            _over = ""
            if _el["over_4096"]:
                _over = f"  ❌ 有 {_el['over_4096']} 条超 4096 → 服务端 --max-model-len 必须 ≥4096"
            elif _el["over_2048"]:
                _over = f"  ⚠ 有 {_el['over_2048']} 条超 2048 → 服务端 --max-model-len 必须 ≥4096"
            print(f"[embed] 实测最长单条 = {_el['max']} token（共 {_el['n']:,} 条；"
                  f"客户端截断上限 EMBED_MAX_TOKENS={EMBED_MAX_TOKENS}）"
                  f"{_over}", flush=True)
            print(f"        最长那条开头: {_el['head']!r}", flush=True)
    except Exception:                            # noqa: BLE001
        pass
    # 失败/重试汇总（写进日志，也落到 results/failures.jsonl）
    _retried = sum(v for k, v in _retry_hist.items() if k >= 1)
    print(f"稳定性: 失败题 {len(_errors)}/{n}"
          f"（{100.0 * len(_errors) / n:.2f}%）| 重试过的题 {_retried} | 重试次数分布 {dict(sorted(_retry_hist.items()))}"
          + (f" | 失败原因 {_error_kinds}" if _error_kinds else ""))
    try:
        _logs = getattr(tee, "logs", {}) or {}
        if _logs:
            print("本次 run 日志: "
                  + " | ".join(f"{k}={v}" for k, v in _logs.items()))
    except Exception:
        pass

    # 运行级汇总落盘：每次运行追加一行，所有「方法×数据集×时间」的消耗可追溯
    summary_row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": args.method, "dataset": args.dataset, "sample": args.sample,
        "graph_version": args.graph_version, "retrieval": args.retrieval,
        "n": n, "complete": n >= len(items),  # 是否跑完全部题目
        "em": round(em, 4), "f1": round(f1, 4),
        # ★ 这张图是本次新建还是复用：复用时本次 build 成本为 0，图内容来自更早那次 run
        "graph": _graph_rec,
        "prompt_tokens": pt, "completion_tokens": ct,
        "embed_tokens": s["build"]["embed_tokens"] + s["query"]["embed_tokens"],
        "llm_seconds": round(llm_s, 1),
        # 建图 / 推理分桶明细
        "build_seconds": round(s["build"]["total_seconds"], 1),
        "build_prompt_tokens": s["build"]["prompt_tokens"],
        "build_completion_tokens": s["build"]["completion_tokens"],
        "build_embed_tokens": s["build"]["embed_tokens"],
        "query_seconds": round(s["query"]["total_seconds"], 1),
        "query_embed_tokens": s["query"]["embed_tokens"],
        "chat_calls": calls,
    }

    # ★ 费用/时间账（SIGMOD 要把「GPU 时间」与「prefill / decode / embed token」分开）
    #   gpu_busy_seconds 是整次 run 的采样积分；按 build/query 的**墙钟占比**拆开。
    _gpu_stats = _gpu.stop()
    _wall = max(1e-9, time.time() - _run_t0)
    _bfrac = (s["build"]["total_seconds"] or 0.0) / max(1e-9, s["build"]["total_seconds"]
                                                        + s["query"]["total_seconds"])
    summary_row["wall_seconds"] = round(_wall, 1)
    summary_row["gpu"] = _gpu_stats
    # ★ P0：把服务端口径并进这一行（审稿人看 summary 就知道 prefix caching / 真实模型）
    try:
        from common.server_probe import summary_block
        summary_row["server"] = summary_block(_server_snap)
        if _server_snap and _server_snap.get("warnings"):
            for _w in _server_snap["warnings"]:
                print(f"[server] {_w}", flush=True)
    except Exception:                            # noqa: BLE001
        summary_row["server"] = {"available": False}
    if _gpu_stats.get("available"):
        _gb = float(_gpu_stats.get("gpu_busy_seconds") or 0.0)
        summary_row["gpu_busy_seconds"] = round(_gb, 1)
        summary_row["gpu_busy_seconds_build"] = round(_gb * _bfrac, 1)   # 估算：按墙钟占比
        summary_row["gpu_busy_seconds_query"] = round(_gb * (1 - _bfrac), 1)
        summary_row["gpu_peak_mem_mb"] = _gpu_stats.get("peak_mem_used_mb")
        print(f"[gpu] busy={_gb:.1f}s（≈{_gb / _wall:.0%} of wall {_wall:.0f}s）"
              f" | 峰值显存 {_gpu_stats.get('peak_mem_used_mb')} MB"
              f" | 拆分 build≈{_gb * _bfrac:.1f}s / query≈{_gb * (1 - _bfrac):.1f}s"
              f"（按墙钟占比）", flush=True)
    summary_row["cost"] = {
        # 三类 token：prefill（prompt）/ decode（completion）/ embed —— 单价完全不同
        "prefill_tokens": pt, "decode_tokens": ct,
        "embed_tokens": s["build"]["embed_tokens"] + s["query"]["embed_tokens"],
        "prefill_tokens_build": s["build"]["prompt_tokens"],
        "decode_tokens_build": s["build"]["completion_tokens"],
        "embed_tokens_build": s["build"]["embed_tokens"],
        "prefill_tokens_query": s["query"]["prompt_tokens"],
        "decode_tokens_query": s["query"]["completion_tokens"],
        "embed_tokens_query": s["query"]["embed_tokens"],
        # 时间：客户端挂钟（含排队） vs GPU busy（真算力占用）
        "wall_seconds_total": round(_wall, 1),
        "client_chat_seconds": round(s["build"]["chat_seconds"] + s["query"]["chat_seconds"], 1),
        "client_embed_seconds": round(s["build"]["embed_seconds"] + s["query"]["embed_seconds"], 1),
        "gpu_busy_seconds": summary_row.get("gpu_busy_seconds"),
        "gpu_busy_seconds_build": summary_row.get("gpu_busy_seconds_build"),
        "gpu_busy_seconds_query": summary_row.get("gpu_busy_seconds_query"),
        "gpu_peak_mem_mb": summary_row.get("gpu_peak_mem_mb"),
        # 调用次数与每问平均（论文成本表直接用这几行）
        "chat_calls": s["build"]["chat_calls"] + s["query"]["chat_calls"],
        "embed_calls": s["build"]["embed_calls"] + s["query"]["embed_calls"],
        "per_question": {
            "prefill_tokens": round(pt / n, 1) if n else None,
            "decode_tokens": round(ct / n, 1) if n else None,
            "embed_tokens": round((s["build"]["embed_tokens"] + s["query"]["embed_tokens"]) / n, 1) if n else None,
            "wall_seconds": round(_wall / n, 3) if n else None,
            "gpu_busy_seconds": round((summary_row.get("gpu_busy_seconds") or 0) / n, 3) if n else None,
            "chat_calls": round((s["build"]["chat_calls"] + s["query"]["chat_calls"]) / n, 2) if n else None,
        },
        # 口径与可复现条件（审稿人必看）
        "measurement": {
            "gpu_time_method": "nvidia-smi 采样积分（Σ util%×Δt，逐卡；非客户端等待时间）"
                               if _gpu_stats.get("available") else "unavailable（无 nvidia-smi）",
            "gpu_probe_interval_sec": _gpu_stats.get("interval_sec"),
            "client_latency_includes_queueing": True,
            "temperature": 0.0, "thinking": False,
            "workers": args.workers,          # 这是**上限**；实际并发见 concurrency.avg
            "split_method": "gpu_busy 按 build/query 客户端墙钟占比拆分（估算，非逐请求埋点）",
        },
    }
    # ★ 动态并发：实际并发是**分布**，论文报 avg_while_active 而不是 --workers。
    #   ⚠️ 自适应闸（`gate_for()`）**只有 hypergraphrag 接了**（全仓唯一调用点
    #   hypergraphrag_adapter.py:137）；其余 10 个方法走 `chat()`/`chat_messages()`，
    #   只计数不限流（common/llm.py:475）→ 它们的 `limit_now`/`max_limit` 是 None，
    #   下面 :820 会打印「直连 chat，未走自适应闸」。计数本身是 LLM 调用级的、
    #   覆盖全部方法（_InflightTracker），所以 avg/peak/hist 每个方法都有。
    try:
        from common.llm import concurrency_stats as _conc
        conc = _conc()
        summary_row["concurrency"] = {**conc, "workers_configured": args.workers}
        if conc.get("available"):
            # 客户端时长 / GPU busy ≈ 排队放大倍数：≈1 说明几乎没排队
            _amp = ((s["build"]["chat_seconds"] + s["query"]["chat_seconds"])
                    / max(1e-9, (summary_row.get("gpu_busy_seconds") or 0.0)))
            summary_row["concurrency"]["client_over_gpu_amplification"] = round(_amp, 2)
            _gate = (f"，闸上限 {conc['max_limit']}→当前 {conc['limit_now']}"
                     if conc.get("gate_used") else "（直连 chat，未走自适应闸）")
            print(f"[conc] 实际并发：avg_while_active={conc.get('avg_while_active')}"
                  f"（峰值 {conc.get('peak')}，配置 workers={args.workers}）{_gate}"
                  f" | 等槽 {conc.get('wait_for_slot_seconds')}s"
                  f" | 客户端/GPU 放大 {_amp:.2f}×", flush=True)
    except Exception as e:                            # noqa: BLE001
        summary_row["concurrency"] = {"available": False, "error": str(e)}
    # trace 去重统计（先进 summary，再打印）
    if args.trace:
        try:
            _d = prompt_dedup_stats()
            if _d.get("chars_saved"):
                summary_row["trace_dedup"] = _d
                print(f"[trace] prompt 去重：共享文件 {_d['files']} 个，"
                      f"省下 {_d['chars_saved'] / 1e6:.1f}M 字符"
                      f"（≈{_d['chars_saved'] / 1e6 * 1.1:.0f} MB）", flush=True)
        except Exception:                            # noqa: BLE001
            pass

    with open(RESULTS_DIR / "summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary_row, ensure_ascii=False) + "\n")

    # ── 可追溯性记录（论文/复现性）：manifest（身份） + failures（失败统计）──
    try:
        mp = write_manifest(args.method, args.dataset, args.sample,
                            args.graph_version, args.retrieval, n,
                            n >= len(items),
                            {"status": "ok", "em": round(em, 4), "f1": round(f1, 4),
                             "n_failed": len(_errors),
                             "n_retried": sum(v for k, v in _retry_hist.items() if k >= 1),
                             "graph": _graph_rec})
        fp = write_failures(args.method, args.dataset, args.sample,
                            args.graph_version, args.retrieval, len(items), n,
                            _retry_hist, _errors, _error_kinds)
        print(f"可追溯记录: {mp.name} / {fp.name}"
              f"（含代码版本、prompt 哈希、环境指纹、失败统计）")
    except Exception as e:
        print(f"[record] ⚠ 可追溯记录写入失败（不影响实验）：{e}")


if __name__ == "__main__":
    main()

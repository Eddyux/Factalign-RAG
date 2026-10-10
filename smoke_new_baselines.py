"""新注册基线的注册自检（离线，秒级；不需要 vLLM、不需要官方源码/权重）。

为 2 个新注册基线（unirag / adaptive_rag）做三件事：

  1. **注册表一致**：run.py 的 ALL_METHODS / CORPUS_METHODS / ADAPTER_METHODS 与
     methods.METHODS 三处必须对齐（漏一处就会出现「--method 认了但走错分支」）
  2. **熔断有效**：官方源码/prompt 未接入时 build() 必须抛 SourceNotReady
     —— 这是最危险的失败模式：半实现基线跑出数字被当成真实基线结果写进表格
  3. **管线通畅**：把放置区常量临时替换成占位串 + 假 LLM/embedding，把 5 个方法
     的 build → query 全流程走一遍，确认「官方源码填好就能跑」（而不是填完才发现
     框架本身有 bug）

只读不改：不写 results/、不碰任何既有方法的数据；临时产物在 data/_selftest/，
跑完删除（--keep 可保留排查）。

用法:
  python smoke_new_baselines.py
  python smoke_new_baselines.py --keep

退出码: 0 = 全部 OK；1 = 有 FAIL。
"""

import argparse
import ast
import json
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np

OK, FAIL, SKIP = "OK", "FAIL", "SKIP"
# 已注册但官方源码未接入的（熔断检查对象）。proh / structrag 已移除，不再在此列。
NEW = ["s2grag"]
ROOT = Path(__file__).resolve().parent

_results = []


def report(step, status, note=""):
    _results.append((step, status, note))
    print(f"[{status}] {step}" + (f" — {note}" if note else ""), flush=True)


# ── 本机环境坑：Anaconda 的 pandas 与 numpy 二进制不匹配（个别 adapter 顶层
#    import pandas 会炸），跟本次注册无关。pandas 真坏掉时塞一个壳进去，
#    只为让 import 通过（DataFrame 注解在 import 期求值）。
def _stub_pandas_if_broken():
    try:
        import pandas  # noqa: F401
        return None
    except Exception as e:
        mod = types.ModuleType("pandas")

        class _DataFrame:      # noqa: N801
            pass

        mod.DataFrame = _DataFrame
        mod.read_parquet = lambda *a, **k: None
        mod.isna = lambda x: x is None
        sys.modules["pandas"] = mod
        return f"pandas 导入失败（{type(e).__name__}），已用空壳顶替（不影响本自检）"


# ── 假 LLM / embedding（按 prompt 里的 MARK_* 关键字返回，全程不联网）──
def fake_chat(system, user, json_mode=False, timeout=None, thinking=False, **kw):
    u = user or ""
    # 官方 S2G-RAG：judge 要 JSON {sufficient, gap_items}；selector 要 {"evidence_global_ids"}
    if "QA/RAG sufficiency judge" in (system or "") or "sufficient" in u.lower():
        return {"sufficient": True, "gap_items": []}
    if "evidence_global_ids" in u and "NUMBERED SENTENCES" in u:
        return {"evidence_global_ids": [1]}
    if "MARK_DECOMPOSE" in u:
        return "sub question one\nsub question two"
    if "MARK_VERIFY" in u:
        return {"sub_fact": "verified sub-fact", "supported": True, "gap": ""}
    if "MARK_REWRITE" in u:
        return "rewritten query"
    if "MARK_SUMMARIZE" in u:
        return "SUMMARY of the cluster"
    if "MARK_STRUCTURIZE" in u:
        return "| entity | value |\n| --- | --- |\n| A | B |"
    if "MARK_SPROPOSE" in u:
        return "I propose: graph"
    if "MARK_SDECIDE" in u:
        return "table"
    if "MARK_SDECOMP" in u:
        return "sub q1\nsub q2"
    if "MARK_SEXTRACT" in u:
        return "extracted key info"
    if "MARK_NORET" in u:
        return "direct answer without retrieval"
    return json_mode and {} or "fake answer"


def fake_answer(question, evidence, dataset=""):
    """假作答器：签名必须与真 `methods.base.generate_answer` 一致（含 dataset）。

    ★ 之前只写了 (question, evidence)，而适配器改成透传 `dataset=` 后这里就炸了
    （本自检正是靠这一点抓到的）——假件的签名要跟着真件走，否则自检会漏检真实调用形状。
    """
    return f"fake answer ({len(evidence)} ev)"


def fake_embed(text, timeout=30):
    v = np.ones(8, dtype=np.float32)
    return v / np.linalg.norm(v)


def fake_embed_batch(texts, timeout=120):
    out = np.tile(np.eye(8, dtype=np.float32)[0], (len(texts), 1))
    return out if len(texts) else np.zeros((0, 8), dtype=np.float32)


def _noop(*a, **k):
    """把放置区守卫替换成空操作（只在管线自检里用，别在别处用）。"""
    return Path(".")


# ── 1. 注册表一致 ──
def check_registry():
    import methods

    tree = ast.parse((ROOT / "run.py").read_text(encoding="utf-8"))
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    all_m, corpus_m, adapter_m = (consts.get("ALL_METHODS"), consts.get("CORPUS_METHODS"),
                                  consts.get("ADAPTER_METHODS"))
    problems = []
    for name in NEW:
        if name not in (all_m or []):
            problems.append(f"{name} 不在 run.py ALL_METHODS")
        if name not in (corpus_m or set()):
            problems.append(f"{name} 不在 run.py CORPUS_METHODS")
        if name not in (adapter_m or ()):
            problems.append(f"{name} 不在 run.py ADAPTER_METHODS")
        if name not in methods.METHODS:
            problems.append(f"{name} 不在 methods.METHODS")
        else:
            obj = methods.METHODS[name]()
            if obj.name != name:
                problems.append(f"{name} 的类 name 属性是 {obj.name!r}")
    report("注册表一致（run.py × methods.METHODS × 新注册方法）",
           OK if not problems else FAIL, "; ".join(problems))
    return not problems


# ── 1b. 作答 prompt 路由：manifest 记的 == 代码真正读的 ──
def check_call_log_split(tmp: Path):
    """逐条调用审计**按阶段分文件**：build 段与 query 段各自成文件、各自 seq 从 1 开始。

    为什么值得自检：建图跑完、作答阶段挂掉是常态，账混在一个文件里就答不了"哪些是建图花的"。
    另：这条自检走的是 `common.llm` 的**真实写入路径**（不是直接调内部函数），
    能一并抓到"切文件后文件句柄没关/seq 没归零"这类只在切换时出现的错。
    """
    import common.llm as L

    if not L.LLM_CALL_LOG_ON:
        report("逐条调用审计按阶段分文件", SKIP, "LLM_CALL_LOG=0 已关闭审计")
        return True
    d = Path(tmp) / "calllog"
    d.mkdir(parents=True, exist_ok=True)
    b = d / "x_build.jsonl"
    q = d / "x_query.jsonl"
    old = L._CALL_LOG["path"], L._CALL_LOG["seq"], L._CALL_LOG["fh"]
    try:
        L.set_call_log_path(b)
        L._call_log_out([{"role": "user", "content": "BUILD"}], "out-build", pt=10, ct=5)
        L.set_call_log_path(q)          # ← run.py 在建图段结束处做的事
        L._call_log_out([{"role": "user", "content": "QUERY"}], "out-query", pt=30, ct=7)
        rb = [json.loads(l) for l in b.read_text(encoding="utf-8").splitlines()]
        rq = [json.loads(l) for l in q.read_text(encoding="utf-8").splitlines()]
        ok = (len(rb) == 1 and len(rq) == 1
              and [r["seq"] for r in rb] == [1] and [r["seq"] for r in rq] == [1]
              and "BUILD" in rb[0]["messages"][0]["content"]
              and "QUERY" in rq[0]["messages"][0]["content"]
              and rb[0]["response"] == "out-build" and rq[0]["response"] == "out-query")
        note = (f"build 文件 {len(rb)} 行(seq={[r['seq'] for r in rb]})、"
                f"query 文件 {len(rq)} 行(seq={[r['seq'] for r in rq]})，输入输出未混杂={ok}")
    finally:                            # 还原，别把审计指向自检目录
        L._CALL_LOG["path"], L._CALL_LOG["seq"], L._CALL_LOG["fh"] = old
        try:
            if old[2] is not None:
                old[2].close()
        except Exception:               # noqa: BLE001
            pass
        L._CALL_LOG["fh"] = old[2]
    report("逐条调用审计按阶段分文件（build/query 各自 seq 从 1 开始）",
           OK if ok else FAIL, note)
    return ok


# 上游 prompt 的**特征长句**（≥8 词连续）：我们的 prompt 里出现任何一条 = 逐字复制了上游。
# 来源：S2G-RAG `utils/prompt_template.py`、HippoRAG 2 `prompts/templates/rag_qa_musique.py`。
# 为什么要把常量**抄进自检**而不是运行时读 baselines/：那两份官方仓库不进版本库
# （见 verify_sync.py 的 EXCLUDE_DIRS），自检必须在没有它们的机器上也能跑。
UPSTREAM_SIGNATURE_PHRASES = (
    # ── S2G-RAG ──
    "you are a factoid question answering assistant",
    "use only the provided evidence",
    "answer with the shortest exact answer span",
    "prefer the canonical alias written in the evidence",
    "not the page title",
    "avoid extra descriptors appositives dates locations parenthetical disambiguation",
    "when a shorter exact answer is sufficient",
    "respond only with the following format nothing else",
    "provide the answer here",
    "provide the rationale here",
    "do not include any additional text headers or explanations outside this format",
    "you are a knowledgeable question answering assistant",
    "based on the context provided if any",
    "provide a brief multihop explanation",
    "prefer the shortest exact answer span",
    # ── HippoRAG 2 rag_qa_musique ──
    "as an accurate and concise question answering assistant",
    "your task is to provide a succinct and accurate answer",
    "based on a given set of retrieved paragraphs",
    "following the analysis of the provided paragraphs",
    "answer the question based on the retrieved paragraphs",
    "please provide your answer on the last line in the following format",
)


def _norm_words(s: str) -> str:
    """小写、去标点、占位符归 X —— 只比**词序列**，避免大小写/标点差异造成漏检。"""
    import re as _re
    s = _re.sub(r"\{[a-z_]+\}", " X ", (s or "").lower())
    return " ".join(_re.sub(r"[^a-z0-9 ]+", " ", s).split())


def check_prompt_not_copied():
    """我们的作答 prompt 必须**没有逐字复制** HippoRAG 2 / S2G-RAG 的 prompt。

    背景：参考官方的**口径**是允许的（本项目本来就按官方口径复刻检索侧），但直接沿用措辞
    既涉及学术诚信、也让"我们的 prompt 是不是抄的"无法自证。这条自检把"特征长句"钉死，
    以后改 prompt 时若顺手把官方句子粘回来，会立刻 FAIL。
    """
    from config import PROMPTS_DIR
    problems = []
    checked = 0
    for p in sorted(PROMPTS_DIR.glob("*.txt")):
        body = _norm_words(p.read_text(encoding="utf-8"))
        checked += 1
        for ph in UPSTREAM_SIGNATURE_PHRASES:
            if ph in body:
                problems.append(f"{p.name} 含上游原文: {ph!r}")
    report("作答 prompt 未逐字复制上游（HippoRAG 2 / S2G-RAG）",
           OK if not problems else FAIL,
           "; ".join(problems[:5]) or
           f"检查 {checked} 份 prompt × {len(UPSTREAM_SIGNATURE_PHRASES)} 条上游特征句，全部无重合")
    return not problems


def check_dataset_chunking():
    """切块口径按数据集解析 + 滑窗落点必须与约定一致。

    两条不变量：
    ① **只有 triviaqa 被覆盖**（1200/100）；hotpotqa/2wiki/musique/popqa 必须与全局
       `EXTRACT_CHUNK_TOKENS/OVERLAP` 逐位相同 —— 否则是"顺手改了别人"的静默回归。
    ② 滑窗落点：步长 = size - overlap，**末块贴住文末即止、不必满 size**。
       1250 → [0..1199] + [1100..1249]；2400 → [0..1199] + [1100..2299] + [2200..2399]。
       这条是踩过坑的：旧实现「再 append 一个贴右端的窗」会产出两个几乎全等的块
       （2400/1200 → [1100..2299] 与 [1200..2399]），白烧一次嵌入。
    用例是**构造的**（不依赖 tokenizer，也不依赖真实语料），所以在没装 transformers 的
    机器上也能跑 —— 本机就缺 transformers，切块会退化成字符级，按 token 断言就没意义了。
    """
    from config import EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP, dataset_chunking
    problems = []
    expect = {"triviaqa": (1200, 100)}
    for ds in ("triviaqa", "hotpotqa", "2wiki", "musique", "popqa"):
        got = dataset_chunking(ds)
        want = expect.get(ds, (EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP))
        if got != want:
            problems.append(f"{ds}: {got} != {want}")
    # 大小写/连字符也要能解析到同一条口径（run.py 传的是命令行原值）
    for variant in ("TriviaQA", "TRIVIAQA", "triviaqa"):
        if dataset_chunking(variant) != (1200, 100):
            problems.append(f"{variant!r} 没解析到 triviaqa 口径")

    # ② 滑窗落点：用桩 tokenizer 精确数 token（每个 `tNNNN` 词 = 1 token）
    import common.llm as _L
    _saved = (_L._encode_with_offsets, _L._decode_ids)

    def _enc(text, kind=None):
        ids, offs, pos = [], [], 0
        for w in text.split(" "):
            i = text.index(w, pos)
            ids.append(len(ids)); offs.append((i, i + len(w))); pos = i + len(w)
        return ids, offs

    _L._encode_with_offsets = _enc
    _L._decode_ids = lambda ids, kind=None: " ".join(f"t{i:04d}" for i in ids)
    try:
        def spans(total, size, ov):
            txt = " ".join(f"t{i:04d}" for i in range(total))
            out = []
            for ct, _cs, _ce in _L.split_by_tokens(txt, size, ov):
                t = ct.split()
                out.append((int(t[0][1:]), int(t[-1][1:])))
            return out

        cases = {
            (1250, 1200, 100): [(0, 1199), (1100, 1249)],
            (2400, 1200, 100): [(0, 1199), (1100, 2299), (2200, 2399)],
            (1200, 1200, 100): [(0, 1199)],
            (4900, 1200, 100): [(0, 1199), (1100, 2299), (2200, 3399), (3300, 4499), (4400, 4899)],
            (4900, 2000, 0):   [(0, 1999), (2000, 3999), (4000, 4899)],
        }
        for (total, size, ov), want in cases.items():
            got = spans(total, size, ov)
            if got != want:
                problems.append(f"切块 {total}@{size}/{ov}: {got} != {want}")
            cov = set()
            for a, b in got:
                cov |= set(range(a, b + 1))
            if cov != set(range(total)):
                problems.append(f"切块 {total}@{size}/{ov} 漏 token {len(set(range(total)) - cov)} 个")
            if any(b - a + 1 > size for a, b in got):
                problems.append(f"切块 {total}@{size}/{ov} 有块超过 size")
    finally:
        _L._encode_with_offsets, _L._decode_ids = _saved

    report("切块口径按数据集解析（triviaqa 1200/100，其余不变）+ 滑窗落点",
           OK if not problems else FAIL,
           "; ".join(problems[:5]) or
           "triviaqa=1200/100，其余 4 个 == 全局值；1250/2400/4900 落点与覆盖全部一致")
    return not problems


def check_hf_endpoint_exported():
    """`HF_ENDPOINT` 必须在**任何 huggingface_hub import 之前**就 export，否则镜像白设。

    踩过的坑：`os.environ.setdefault("HF_ENDPOINT", ...)` 原来写在 `_load_tokenizer()` 里面，
    是**加载 tokenizer 时**才执行；而 `huggingface_hub` 在 **import 时**就把 `HF_ENDPOINT`
    读成模块常量。于是"设了镜像却还在连 huggingface.co 超时"，且报错里看不到原因。
    修法：移到 `common/llm.py` 的模块 body（早于该模块内任何 `from transformers import ...`）。

    三条优先级各验一次（都在子进程里跑，避免本进程已被 import 的模块污染）：
      ① 配置文件 → `HF_ENDPOINT` 真的进了 os.environ；
      ② 已有的真实环境变量**不被配置文件覆盖**；
      ③ 什么都不给 → 代码默认仍是 hf-mirror（离线/内网机器的兜底）。
    """
    import subprocess
    import sys as _sys

    _PROBE = ("import sys; sys.path.insert(0, %r)\n"
              "from common.config_file import apply_config_file\n"
              "apply_config_file('configs/baselines.json', method='zeroshot', verbose=False)\n"
              "import os, common.llm\n"
              "print(os.environ.get('HF_ENDPOINT', ''))\n") % str(ROOT)

    problems = []
    cases = [
        ("配置文件注入", None, "https://hf-mirror.com"),
        ("真实环境变量优先", "https://my-own-mirror.example", "https://my-own-mirror.example"),
    ]
    for label, preset, want in cases:
        env = dict(os.environ)
        env.pop("HF_ENDPOINT", None)
        if preset:
            env["HF_ENDPOINT"] = preset
        try:
            r = subprocess.run([_sys.executable, "-c", _PROBE], capture_output=True,
                               text=True, encoding="utf-8", env=env, timeout=120)
            got = (r.stdout or "").strip().splitlines()[-1] if r.stdout.strip() else ""
        except Exception as e:                                     # noqa: BLE001
            problems.append(f"{label}: 子进程失败 {type(e).__name__}")
            continue
        if got != want:
            problems.append(f"{label}: {got!r} != {want!r}")

    # ③ 代码默认兜底（不读配置文件）
    env = {k: v for k, v in os.environ.items() if k != "HF_ENDPOINT"}
    try:
        r = subprocess.run([_sys.executable, "-c",
                            "import sys; sys.path.insert(0, %r)\n"
                            "import os, common.llm\n"
                            "print(os.environ.get('HF_ENDPOINT', ''))\n" % str(ROOT)],
                           capture_output=True, text=True, encoding="utf-8", env=env, timeout=120)
        got = (r.stdout or "").strip().splitlines()[-1] if r.stdout.strip() else ""
        if got != "https://hf-mirror.com":
            problems.append(f"代码默认兜底: {got!r} != 'https://hf-mirror.com'")
    except Exception as e:                                         # noqa: BLE001
        problems.append(f"代码默认兜底: 子进程失败 {type(e).__name__}")

    report("HF_ENDPOINT 镜像在 import 前 export（配置/环境变量/默认三路优先级）",
           OK if not problems else FAIL,
           "; ".join(problems[:4]) or "配置注入 ✓ / 真实环境变量优先 ✓ / 代码默认兜底 ✓")
    return not problems


def check_prompt_routing():
    """全项目"每个方法 × 每个数据集"真正读的 prompt，必须与 manifest 登记的完全一致。

    为什么值得自检：**manifest 是给审稿人看的复现凭证**，它记错 prompt 比代码写错还坏
    （代码错会掉分，凭证错是"证据造假"）。历史上真出现过 `METHOD_PROMPTS["zeroshot"]
    = ["rag_prompt"]` 而 zeroshot 从没读过 rag_prompt 的情况——离线的本轮检查就能抓到。
    """
    from common.run_record import method_prompt_keys, PROMPT_FILES
    from methods.base import (RAG_PROMPT, TRIVIAQA_RAG_PROMPT, NO_RAG_PROMPT,
                              rag_prompt_for, NO_CONTEXT_METHODS)
    import methods

    ds_names = ("hotpotqa", "musique", "2wiki", "triviaqa", "popqa")
    # 键 → 该文件的实际正文（用于"登记的那份是不是真的被用了"）
    body = {k: p.read_text(encoding="utf-8").strip() for k, p in PROMPT_FILES.items()}
    problems = []
    for name in list(methods.METHODS):
        for ds in ds_names:
            # 代码真正会读的那份：无证据方法不分数据集，其余按数据集分派
            actual_key = ("rag_prompt_nocontext"
                          if name in NO_CONTEXT_METHODS else
                          ("rag_prompt_triviaqa" if ds == "triviaqa" else "rag_prompt"))
            expected = {"rag_prompt": RAG_PROMPT,
                        "rag_prompt_triviaqa": TRIVIAQA_RAG_PROMPT,
                        "rag_prompt_nocontext": NO_RAG_PROMPT}[actual_key]
            if rag_prompt_for(ds, name) != expected:
                problems.append(f"{name}@{ds}: rag_prompt_for 没返回 {actual_key}")
            keys = method_prompt_keys(name, ds)
            if actual_key not in keys:
                problems.append(f"{name}@{ds}: manifest 漏登 {actual_key}")
            # 反向：登记了的每一份，正文都必须真的出现在它读到的 prompt 里
            for k in keys:
                if k == "extract_graph":        # 建图期 prompt，不在作答路径上
                    continue
                if k != actual_key:
                    problems.append(f"{name}@{ds}: manifest 多登了没用到的 {k}")
                elif body.get(k, "") not in expected:
                    problems.append(f"{name}@{ds}: {k} 的正文与实读不符")
    # zeroshot 必须**不随数据集变化**（无证据路径没有单跳/多跳之分）
    zs = {ds: rag_prompt_for(ds, "zeroshot") for ds in ds_names}
    if len(set(zs.values())) != 1:
        problems.append("zeroshot 的 prompt 随数据集变了（无证据路径不该分派）")
    report("作答 prompt 路由（代码实读 == manifest 登记；zeroshot 不分数据集）",
           OK if not problems else FAIL,
           "; ".join(problems[:6]) or
           f"{len(list(methods.METHODS))} 方法 × {len(ds_names)} 数据集全部一致")
    return not problems


# ── 2. 熔断有效（源码未接入 → 必须拒绝出数）──
def check_guards(ctx):
    import methods
    import methods.s2grag_adapter as sg
    from methods.scaffold import SourceNotReady

    # 已就位官方仓库的适配器：把候选路径临时指到不存在的目录，验证「源码缺失必须熔断」
    cand_attr = {"s2grag": ("S2GRAG_REPO_CANDIDATES", sg)}

    problems = []
    for name in NEW:
        obj = methods.METHODS[name]()
        saved = None
        mod = None
        if name in cand_attr:
            attr, mod = cand_attr[name]
            saved = getattr(mod, attr)
            setattr(mod, attr, [ROOT / "data" / "_selftest" / "no_such_repo"])
        try:
            obj.build(ctx, working_dir=str(ROOT / "data" / "_selftest" / name))
        except SourceNotReady as e:
            report(f"熔断 {name}", OK, str(e).splitlines()[0][:90])
            continue
        except Exception as e:
            problems.append(name)
            report(f"熔断 {name}", FAIL,
                   f"抛的不是 SourceNotReady 而是 {type(e).__name__}: {e}")
            continue
        finally:
            if saved is not None:
                setattr(mod, cand_attr[name][0], saved)
        problems.append(name)
        report(f"熔断 {name}", FAIL, "源码未接入却 build 成功了（会跑出假数字！）")
    return not problems


# ── 3. 管线通畅（放置区填好后能否跑通）──
def check_pipelines(ctx, tmp: Path):
    import methods.s2grag_adapter as sg
    import methods.ircot_adapter as ircot
    import methods.corpus_index as ci

    # 假 LLM/embedding（放置区守卫一并空操作，模拟「官方源码已填」）
    for mod in (sg, ircot):
        if hasattr(mod, "llm_chat"):
            mod.llm_chat = fake_chat
        if hasattr(mod, "generate_answer"):
            mod.generate_answer = fake_answer
        for guard in ("require_repo", "require_path"):
            if hasattr(mod, guard):
                setattr(mod, guard, _noop)
    ci.embed_batch = fake_embed_batch
    ci.llm_embed = fake_embed

    # 3a. 底座口径：IRCoT 的「段落合并 + BM25 + 上限」必须与规范实现同序
    #     （IRCoT 官方口径检索的是 Wikipedia 整段，不是我们的整段块 → 不再与 CorpusIndex 比）
    try:
        ircot_obj = ircot.IRCoT()
        ircot_obj.dataset = "hotpotqa"
        ircot_obj.build(ctx)
        q = "who is A"
        titles, paras, idxs = [], [], []
        ircot_obj._retrieve(q, titles, paras, idxs)
        # 规范实现：按 title 合并句子 → 同一 _bm25 → 同一排序 → 取 retrieval_count 篇
        by_title = {}
        for c in ctx:
            t = (c.get("title") or "").strip() or "untitled"
            by_title.setdefault(t, []).extend(c.get("sentences") or [])
        ref_titles = list(by_title)
        ref_paras = [" ".join(by_title[t]) for t in ref_titles]
        scores = ircot._bm25(ircot.remove_wh_words(q), [ircot._tokenize(p) for p in ref_paras])
        ref_order = sorted(range(len(ref_paras)), key=lambda i: -scores[i])[:ircot_obj.retrieval_count]
        ref = [ref_titles[i] for i in ref_order]
        report("口径自检 IRCoT 段落检索（合并+BM25+每轮上限）",
               OK if titles == ref else FAIL, f"ircot={titles} 规范={ref}")
    except Exception as e:
        report("口径自检 IRCoT 段落检索（合并+BM25+每轮上限）", FAIL,
               f"{type(e).__name__}: {e}")

    # 3b. 逐个方法 build → query
    # S2G-RAG（s2grag）：judge → gap → 检索 → 句子级选证 → 统一生成
    try:
        gp = ROOT / "baselines" / "S2G-RAG-main" / "utils" / "prompt_template.py"
        if not gp.exists():
            report("管线 s2grag（judge→gap→检索→选证→生成）", SKIP, f"缺 {gp}")
        else:
            ns = {}
            exec(compile(gp.read_text(encoding="utf-8"), str(gp), "exec"), ns)
            sg._load_official_prompts = lambda repo=None, _ns=ns: _ns     # 用官方 prompt 常量
            s2 = sg.S2GRAG()
            s2.dataset = "hotpotqa"
            s2.build(ctx, working_dir=str(tmp / "s2grag"))
            r = s2.query("Who is A?")
            # 假 judge 总是 sufficient → 首轮空证据会被官方规则强制检索一次，所以 turns_used==2
            ok = bool(r.get("answer")) and r.get("turns_used", 0) >= 1
            report("管线 s2grag（judge→gap→检索→选证→生成）", OK if ok else FAIL,
                   f"turns={r.get('turns_used')} 证据块={r.get('retrieved_blocks')} "
                   f"trace={r.get('sufficiency_trace')} answer={r.get('answer')!r}")
    except Exception as e:
        report("管线 s2grag（judge→gap→检索→选证→生成）", FAIL, f"{type(e).__name__}: {e}")

    # PRoH 的管线冒烟已随该 baseline 一起移除（adapter 已删除，不再注册）。


def _write_fake_hippo(root: Path, flavor: str) -> Path:
    """写一个假 hipporag 包，复刻官方**两代之一**的真实形状（用于离线回归）。

    flavor="inject"：2.0.0a5 那代
        `HippoRAG.__init__(global_config, extraction_llm, embedding_model, index_identity=None)`；
        注入 embedding/LLM 时**必须**给 index_identity（HippoRAG.py:186-191 原文行为）；
        `openie_openai` 的池子**自己传** `max_workers`（patch 类是空操作 → 必须写 BaseConfig）。
        图文件 = graph.pickle。
    flavor="name"：2.0.0-alpha.2 那代
        `__init__(global_config, save_dir, llm_model_name, embedding_model_name, llm_base_url)`
        ——**没有注入口**，自己在 __init__ 里调 `_get_llm_class` / `_get_embedding_model_class`
        （所以适配器只能换掉这两个模块全局名）；池子是**裸** `ThreadPoolExecutor()`。
        图文件 = graph.graphml。
    存图条件两代一致（只在有新增块时存，`if new > 0`）。
    """
    pkg = root / "src" / "hipporag"
    for sub in ("utils", "llm", "embedding_model", "information_extraction"):
        (pkg / sub).mkdir(parents=True, exist_ok=True)
        (pkg / sub / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("from .HippoRAG import HippoRAG\n", encoding="utf-8")

    (pkg / "utils" / "config_utils.py").write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class BaseConfig:\n"
        "    llm_name: str = 'x'\n"
        "    embedding_model_name: str = 'y'\n"
        "    llm_base_url: str = ''\n"
        "    openie_mode: str = 'online'\n"
        "    embedding_max_seq_len: int = 2048\n"
        "    save_dir: str = '.'\n"
        "    retrieval_top_k: int = 5\n"
        "    qa_top_k: int = 5\n"
        + ("    openie_max_workers: int = 8\n" if flavor == "inject" else ""),
        encoding="utf-8")
    (pkg / "utils" / "embed_utils.py").write_text(
        "def retrieve_knn(*a, **k):\n    return None\n", encoding="utf-8")
    (pkg / "llm" / "base.py").write_text(
        "class LLMConfig:\n"
        "    def __init__(self, d):\n"
        "        self.__dict__.update(d)\n"
        "    @classmethod\n"
        "    def from_dict(cls, d):\n"
        "        return cls(d)\n"
        "\n"
        "class BaseLLM:\n"
        "    def __init__(self, global_config=None):\n"
        "        self.global_config = global_config\n"
        "        self.llm_name = getattr(global_config, 'llm_name', None)\n"
        "    def infer(self, messages, **kwargs):\n"
        "        raise NotImplementedError\n", encoding="utf-8")
    (pkg / "embedding_model" / "base.py").write_text(
        "class BaseEmbeddingModel:\n"
        "    def __init__(self, global_config=None):\n"
        "        self.global_config = global_config\n"
        "        self.embedding_model_name = getattr(global_config, 'embedding_model_name', None)\n"
        "    def batch_encode(self, texts, **kw):\n"
        "        raise NotImplementedError\n", encoding="utf-8")

    # alpha.2 的两个工厂：**故意抛错** —— 适配器没换掉它们就会当场暴露
    (pkg / "llm" / "__init__.py").write_text(
        "from .base import BaseLLM, LLMConfig\n"
        "\n"
        "def _get_llm_class(config):\n"
        "    raise RuntimeError('真 _get_llm_class 被调用了（注入没生效）')\n",
        encoding="utf-8")
    (pkg / "embedding_model" / "__init__.py").write_text(
        "from .base import BaseEmbeddingModel\n"
        "\n"
        "def _get_embedding_model_class(embedding_model_name='x'):\n"
        "    raise RuntimeError('真 _get_embedding_model_class 被调用了（注入没生效）')\n",
        encoding="utf-8")

    if flavor == "inject":
        oie_init = ("    def __init__(self, llm_model, max_workers: int = 8):\n"
                    "        self.llm_model = llm_model\n"
                    "        self.max_workers = max_workers\n")
        pool = "ThreadPoolExecutor(max_workers=self.max_workers)"
    else:
        oie_init = ("    def __init__(self, llm_model):\n"
                    "        self.llm_model = llm_model\n")
        pool = "ThreadPoolExecutor()"
    (pkg / "information_extraction" / "openie_openai.py").write_text(
        "import json\n"
        "from concurrent.futures import ThreadPoolExecutor\n"
        "\n"
        "POOL_WORKERS = []\n"
        "\n"
        "class _Out:\n"
        "    def __init__(self, metadata):\n"
        "        self.metadata = metadata\n"
        "\n"
        "class OpenIE:\n"
        + oie_init +
        "    def ner(self, chunk_key, passage):\n"
        f"        with {pool} as ex:\n"
        "            POOL_WORKERS.append(getattr(ex, '_max_workers', -1))\n"
        "            list(ex.map(lambda p: self.llm_model.infer(\n"
        "                [{'role': 'user', 'content': 'list named_entities' + p}]), [passage]))\n"
        "        return _Out({})\n"
        "    def triple_extraction(self, chunk_key, passage, named_entities):\n"
        f"        with {pool} as ex:\n"
        "            POOL_WORKERS.append(getattr(ex, '_max_workers', -1))\n"
        "            list(ex.map(lambda p: self.llm_model.infer(\n"
        "                [{'role': 'user', 'content': '{\\\"triples\\\"}' + p}]), [passage]))\n"
        "        return _Out({})\n", encoding="utf-8")

    graph_file = "graph.pickle" if flavor == "inject" else "graph.graphml"
    # 官方两代在 initialize_graph 里设的属性名不同，适配器 _graph_file_of 两个都认
    graph_attr = (f"        self._graph_pickle_filename = str(Path(self.working_dir) / '{graph_file}')\n"
                  f"        self._graph_out = self._graph_pickle_filename\n"
                  if flavor == "inject" else
                  f"        self._graphml_xml_file = str(Path(self.working_dir) / '{graph_file}')\n"
                  f"        self._graph_out = self._graphml_xml_file\n")
    if flavor == "inject":
        ctor = ("    def __init__(self, global_config, extraction_llm, embedding_model,\n"
                "                 index_identity=None):\n"
                "        if index_identity is not None and (not isinstance(index_identity, str)\n"
                "                                           or not index_identity.strip()):\n"
                "            raise ValueError('index_identity must be a non-empty string when provided.')\n"
                "        injected = embedding_model is not None or extraction_llm is not None\n"
                "        if index_identity is None and injected:\n"
                "            raise ValueError('index_identity is required when injecting an embedding '\n"
                "                             'model, extraction LLM, or text preprocessor.')\n"
                "        IDENTITIES.append(index_identity)\n"
                "        self.extraction_llm = extraction_llm\n"
                "        self.embedding_model = embedding_model\n"
                "        self.llm_model = extraction_llm\n"
                "        self.index_identity = index_identity\n")
        imports = ("from .information_extraction.openie_openai import OpenIE\n")
        openie = ("        self.openie = OpenIE(self.llm_model,\n"
                  "                             max_workers=global_config.openie_max_workers)\n")
    else:
        ctor = ("    def __init__(self, global_config=None, save_dir=None, llm_model_name=None,\n"
                "                 embedding_model_name=None, llm_base_url=None):\n"
                "        IDENTITIES.append(None)\n"
                "        self.extraction_llm = None\n"
                "        self.embedding_model = None\n"
                "        self.llm_model = _get_llm_class(global_config)\n"
                "        self.embedding_model = _get_embedding_model_class(\n"
                "            global_config.embedding_model_name)(\n"
                "            global_config=global_config,\n"
                "            embedding_model_name=global_config.embedding_model_name)\n")
        imports = ("from .llm import _get_llm_class, BaseLLM\n"
                   "from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel\n"
                   "from .information_extraction.openie_openai import OpenIE\n")
        openie = "        self.openie = OpenIE(llm_model=self.llm_model)\n"

    (pkg / "HippoRAG.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "from .utils.embed_utils import retrieve_knn\n"
        + imports +
        "\n"
        "INDEX_CALLS = []\n"
        "IDENTITIES = []\n"
        "\n"
        "class _Solution:\n"
        "    def __init__(self, docs):\n"
        "        self.docs = docs\n"
        "\n"
        "class _Graph:\n"
        "    def __init__(self):\n"
        "        self.nodes = set()\n"
        "    def vcount(self):\n"
        "        return len(self.nodes)\n"
        "\n"
        "class HippoRAG:\n"
        + ctor +
        "        self.global_config = global_config\n"
        "        wd = Path(global_config.save_dir)\n"
        "        wd.mkdir(parents=True, exist_ok=True)\n"
        "        self.working_dir = str(wd)\n"
        "        self.graph = self.initialize_graph()\n"
        "        self.openie_results_path = str(wd / ('openie_results_ner_%s.json'\n"
        "            % global_config.llm_name.replace('/', '_')))\n"
        "        self._chunk_meta_path = wd / 'chunk_metadata.json'\n"
        "        self._known = (set(json.loads(self._chunk_meta_path.read_text()))\n"
        "                       if self._chunk_meta_path.exists() else set())\n"
        "        self._docs = []\n"
        + openie +
        "    def initialize_graph(self):\n"
        # 忠实复刻：官方在 initialize_graph 里设图文件路径（a5: _graph_pickle_filename /
        # alpha.2: _graphml_xml_file）——适配器就是靠这个属性找图来核对的
        + graph_attr +
        "        g = _Graph()\n"
        "        if not getattr(self.global_config, 'force_index_from_scratch', False) \\\n"
        f"                and Path(self._graph_out).exists():\n"
        "            g.nodes = set(json.loads(Path(self._graph_out).read_text()))\n"
        "        return g\n"
        "    def index(self, docs):\n"
        "        INDEX_CALLS.append(len(docs))\n"
        "        self._docs = list(docs)\n"
        "        recs, new = [], 0\n"
        "        for i, d in enumerate(docs):\n"
        "            self.openie.ner(str(i), d)\n"
        "            self.openie.triple_extraction(str(i), d, [])\n"
        "            recs.append({'idx': str(i), 'passage': d,\n"
        "                         'extracted_triples': [['A', 'is', 'B']]})\n"
        "            if d not in self._known:\n"
        "                self._known.add(d)\n"
        "                self.graph.nodes.add('passage::' + d)\n"
        "                new += 1\n"
        "        Path(self.openie_results_path).write_text(\n"
        "            json.dumps({'docs': recs}), encoding='utf-8')\n"
        "        self._chunk_meta_path.write_text(json.dumps(sorted(self._known)),\n"
        "                                         encoding='utf-8')\n"
        "        if new > 0:              # 官方两代一致：只在有新增块时存图\n"
        "            Path(self._graph_out).write_text(json.dumps(sorted(self.graph.nodes)))\n"
        "    def retrieve(self, queries, num_to_retrieve=5):\n"
        "        return [_Solution(self._docs[:num_to_retrieve])]\n", encoding="utf-8")
    return pkg


def _hippo_case(flavor: str, tmp: Path):
    """跑一代官方形状的 build→query（返回 (ok, notes)）。"""
    import importlib
    import methods.hipporag_adapter as ha

    pkg_root = tmp / f"fakehippo_{flavor}"
    _write_fake_hippo(pkg_root, flavor)

    saved = {n: getattr(ha, n) for n in ("chat_messages", "embed_batch", "generate_answer",
                                         "_POOL_PATCHED", "_OPENIE_EXPLICIT_WORKERS")}
    saved_find_repo = ha.HippoRAG._find_repo
    n_llm_calls = []          # 记：注入的 LLM 是否真的被官方抽调用到

    def fake_chat_messages(messages, json_mode=False, timeout=None, **kw):
        txt = " ".join(m.get("content", "") or "" for m in messages if isinstance(m, dict))
        n_llm_calls.append(1)
        if "named_entities" in txt:
            return '{"named_entities": ["A", "B"]}', 11, 4
        if '"triples"' in txt:
            return '{"triples": [["A", "is", "B"]]}', 13, 6
        return "fake", 7, 2

    notes = []
    sys.path.insert(0, str(pkg_root / "src"))
    try:
        importlib.import_module("hipporag")
        hm = sys.modules["hipporag.HippoRAG"]     # 包属性被类盖住 → 从 sys.modules 拿
        hoe = sys.modules["hipporag.information_extraction.openie_openai"]
        ha.chat_messages = fake_chat_messages
        ha.embed_batch = lambda texts, **kw: np.zeros((len(texts), ha.EMBED_DIM), np.float32)
        ha.generate_answer = fake_answer
        ha.HippoRAG._find_repo = lambda self: pkg_root
        ha._POOL_PATCHED = False
        ha._OPENIE_EXPLICIT_WORKERS = False

        # 官方更新后把「索引生产者」的类身份（module + __qualname__）写进索引清单；
        # build / load 两条路径必须算出同一个字符串，否则查询轮 init 直接 manifest mismatch
        class _B:
            pass
        c1 = ha._component_classes(_B, _B, _B)
        c2 = ha._component_classes(_B, _B, _B)
        ident_stable = all(f"{a.__module__}.{a.__qualname__}" == f"{b.__module__}.{b.__qualname__}"
                           for a, b in zip(c1, c2))
        notes.append(f"类身份跨 build/load 稳定={ident_stable}（{c1[0].__qualname__}）")

        obj = ha.HippoRAG()
        wd = tmp / f"hipporag_{flavor}"
        obj.build([{"title": "Doc A", "text": "A met B in 1990. " * 40},
                   {"title": "Doc B", "text": "B visited C. " * 40}],
                  working_dir=str(wd))
        n_index = hm.INDEX_CALLS[-1] if hm.INDEX_CALLS else 0
        pool = sorted(set(hoe.POOL_WORKERS))
        graph_file = wd / "graph.pickle"
        notes.append("官方仓库形状=HippoRAG 2 现行版（注入 + index_identity + graph.pickle）")
        notes.append(f"index() 被调用={n_index} 块")
        notes.append(f"注入的 LLM 真被抽调用={len(n_llm_calls)} 次")
        notes.append(f"池子并发={pool}")
        notes.append(f"图文件={graph_file.name}{'有' if graph_file.exists() else '无'}"
                     f"/{obj._rag.graph.vcount()} 节点")
        notes.append(f"index_identity={'传入' if hm.IDENTITIES[-1] == ha._index_identity() else '没传'}"
                     f"/openie_max_workers="
                     f"{obj._rag.global_config.openie_max_workers}")

        r = obj.query("Who is A?")
        n_ev = len(r.get("evidence") or [])
        notes.append(f"query answer={r.get('answer')!r} n_ev={n_ev}")

        # 残骸复现：只剩向量库/chunk_metadata、图文件丢了 → 官方判定「无新增块」不存图，
        # 适配器必须**直接失败**（不许写 marker 假装建好，否则整轮全 unknown）
        graph_existed = graph_file.exists()
        graph_file.unlink()
        residue = ""
        try:
            obj.build([{"title": "Doc A", "text": "A met B in 1990. " * 40},
                       {"title": "Doc B", "text": "B visited C. " * 40}],
                      working_dir=str(wd))
        except RuntimeError as e:
            residue = str(e)
        residue_ok = "没落盘" in residue
        notes.append(f"残骸目录被发现={residue_ok}")

        ok = (ident_stable and n_index > 0 and bool(n_llm_calls) and graph_existed
              and pool == [ha.HIPPORAG_MAX_ASYNC] and n_ev > 0 and residue_ok
              and hm.IDENTITIES[-1] == ha._index_identity()
              and obj._rag.global_config.openie_max_workers == ha.HIPPORAG_MAX_ASYNC)
        return ok, notes
    finally:
        for k, v in saved.items():
            setattr(ha, k, v)
        ha.HippoRAG._find_repo = saved_find_repo
        sys.path.remove(str(pkg_root / "src"))
        for name in [n for n in sys.modules if n == "hipporag" or n.startswith("hipporag.")]:
            sys.modules.pop(name, None)


def check_hipporag_flow(tmp: Path):
    """回归：HippoRAG 适配器在官方仓库（HippoRAG 2 现行版）上能 build→query 走通。

    为什么必须有：这文件出过两次致命回归，全是"能静默出 0 分"的那种——
      ① build() 里的 `self._rag.index(docs)` 被误删 → 什么都没抽、marker 照写、query 全 unknown；
      ② 官方仓库目录换了（2.0.0-alpha.2 ↔ 2.0.0a5）、构造 API 与图文件名都不同，
         适配器却按老那代写 → 崩或静默错。所以这里：
         · 让假包**记录 index() 被调用**、**记录池子的 max_workers**、**记录注入的 LLM 是否被调用**；
         · 假包按官方现行版的形状写：注入时不给 index_identity 就 ValueError（HippoRAG.py:186-191）、
           池子自己传 max_workers、图文件 graph.pickle；
         · 复现"图没落盘"的残骸目录，要求适配器当场失败。
    """
    import methods.hipporag_adapter as ha

    # 老版（构造函数没有 index_identity）→ _build_rag 必须退回不传，而不是崩
    class _BareCtor:
        def __init__(self, **kw):
            if "index_identity" in kw:
                raise TypeError("__init__() got an unexpected keyword argument 'index_identity'")
            self.kw = kw
    try:
        fb = ha._build_rag(_BareCtor, object(), None, None)
        fallback_note = f"老版退回不传={'ok' if 'index_identity' not in fb.kw else 'FAIL'}"
    except Exception as e:                                   # noqa: BLE001
        fallback_note = f"老版退回失败: {type(e).__name__}: {e}"

    try:
        ok, notes = _hippo_case("inject", tmp)
    except Exception as e:                                   # noqa: BLE001
        ok, notes = False, [f"{type(e).__name__}: {e}"]
    notes.append(fallback_note)
    ok = ok and fallback_note.endswith("ok")
    report("HippoRAG build→query 回归（假包·HippoRAG 2 现行版）："
           "真建图/真注入/真落盘/残骸必失败",
           OK if ok else FAIL, "；".join(notes))
    return ok


def check_hipporag_json_repair():
    """回归：官方解析器只认 `{"named_entities": [...]}`（openie_openai.py:29-44）。

    线上撞到的是 `OpenIE response does not contain a valid JSON object with 'named_entities'`——
    模型给裸数组或换了个字段名。temperature=0 下重试同一 prompt 多半复现同一份，
    所以必须**先修形状**（HIPPORAG_FIX_JSON），修不了才走"重试 → 摘 error 不打断整轮"。
    """
    import json as _json
    import methods.hipporag_adapter as ha

    N = ha._normalize_openie_payload
    cases = [
        ("裸数组", '["Antiochus IX", "Cleopatra IV"]', "ner", ["Antiochus IX", "Cleopatra IV"]),
        ("换名字 entities", '{"entities": ["A", "B"]}', "ner", ["A", "B"]),
        ("前后带解释文字", 'Sure:\n["A", "B"]\nDone.', "ner", ["A", "B"]),
        ("裸数组混 int/bool", '[1, true, "A"]', "ner", ["1", "A"]),
        ("三元组裸数组", '[["A","is","B"]]', "triple", [["A", "is", "B"]]),
        ("三元组换名 relations", '{"relations": [["A","is","B"]]}', "triple", [["A", "is", "B"]]),
    ]
    ok_all = True
    bad = []
    for desc, raw, kind, expect in cases:
        out = N(raw, kind)
        try:
            field = "named_entities" if kind == "ner" else "triples"
            got = _json.loads(out)[field]
        except Exception as e:                               # noqa: BLE001
            got, out = f"解析失败 {type(e).__name__}", out
        if got != expect:
            ok_all = False
            bad.append(f"{desc}: {got!r}")
    # 合规输出必须**字节级原样**（不能顺手改动正常样本）
    same = N('{"named_entities": ["A", "B"]}', "ner") == '{"named_entities": ["A", "B"]}'
    # 真不是 JSON 的不能瞎编（要留给重试/摘 error 那条路）
    untouched = N("I cannot find any entities here.", "ner") == "I cannot find any entities here."
    # kind 判定：从 prompt 认 NER / 三元组
    kind_ok = (ha._openie_kind([{"content": 'key "named_entities"'}]) == "ner"
               and ha._openie_kind([{"content": 'key "triples"'}]) == "triple")
    # ★ 回归（线上 5012 块全中）：官方三元组 prompt 把**实体列表当输入**塞进 user 帧
    #   （openie_openai.py:113 `named_entity_json=json.dumps({"named_entities": ...})`
    #   + templates/triple_extraction.py:14-21 的 `{named_entity_json}`），所以三元组调用里
    #   **同时**出现 named_entities 和 triples。kind 若先判 named_entities → 每一次三元组
    #   调用都被判成 ner → 合规 `{"triples": [...]}` 被改写成 `{"named_entities": []}`
    #   → 官方抛 `OpenIE response does not contain a valid JSON object with 'triples'`。
    real_triple_prompt = [
        {"role": "system", "content": "Respond with a JSON list of triples."},
        {"role": "user", "content": 'it has a named entity list and a triple list. '
                                    '{"named_entities": ["A", "B"]}'},
    ]
    kind_real = ha._openie_kind(real_triple_prompt) == "triple"
    # 合规三元组必须原样穿过：**即使 kind 判成 ner 也不许改写**（保命规则①）；
    # 裸三元组数组也不许被包成 `{"named_entities": []}`（保命规则②）。
    good = '{"triples": [["A", "is", "B"]]}'
    want = {"triples": [["A", "is", "B"]]}
    safe = True
    for kind_try in ("triple", "ner"):
        try:
            safe = safe and _json.loads(N(good, kind_try)) == want
        except Exception:                                    # noqa: BLE001
            safe = False
    try:
        safe = safe and _json.loads(N('[["A", "is", "B"]]', "ner")) == want
    except Exception:                                        # noqa: BLE001
        safe = False
    ok = ok_all and same and untouched and kind_ok and kind_real and safe
    notes = [f"6 种畸形全部修成官方形状={ok_all}" + (f"（坏的：{bad}）" if bad else ""),
             f"合规输入字节级不变={same}", f"非 JSON 不瞎编={untouched}", f"kind 判定={kind_ok}",
             f"三元组 prompt 内嵌实体列表仍判 triple={kind_real}",
             f"合规三元组不被改写（kind 错也不改）={safe}"]
    report("HippoRAG OpenIE JSON 形状修复（裸数组/换名/解释文字 → 官方字段）",
           OK if ok else FAIL, "；".join(notes))
    return ok


def check_hipporag_openie_tolerance(tmp: Path):
    """回归：官方更新后 `batch_openie` 遇到**任意一个块**有 `metadata['error']` 就
    `raise RuntimeError`（openie_openai.py:214-216 NER / :252-254 三元组），而写 json 的
    `save_openie_results` 排在它后面 → 一整轮几百万 token 全废（实测 3012 块里 2 块失败
    就把建图打断）。适配器必须在**单块**层面重试，仍失败就把 error 摘掉（回到旧版语义：
    这块没有实体/事实边、文本仍作 passage 节点），这样官方 batch 级校验不会触发。
    """
    import importlib
    import methods.hipporag_adapter as ha

    pkg_root = tmp / "fakehippo_tol"
    if not (pkg_root / "src" / "hipporag").exists():
        _write_fake_hippo(pkg_root, "inject")

    saved = {n: getattr(ha, n) for n in ("_POOL_PATCHED", "_OPENIE_EXPLICIT_WORKERS")}
    sys.path.insert(0, str(pkg_root / "src"))
    try:
        importlib.import_module("hipporag")
        mod = sys.modules["hipporag.information_extraction.openie_openai"]
        calls = []

        def fake_ner(self, chunk_key, passage):
            calls.append((chunk_key, passage))
            once_failed = "POISON_ONCE" in passage and calls.count((chunk_key, passage)) < 2
            always = "POISON_ALWAYS" in passage
            return mod._Out({"error": "boom"} if (once_failed or always) else {})

        mod.OpenIE.ner = fake_ner
        mod.OpenIE.triple_extraction = lambda self, k, p, ents: mod._Out({})
        ha._POOL_PATCHED = False
        ha._OPENIE_EXPLICIT_WORKERS = False
        with ha._FAIL_LOCK:
            ha._OPENIE_FAILS["ner"].clear()
            ha._OPENIE_FAILS["triple"].clear()
        ha._patch_official_openie()          # 包住上面刚放的 fake_ner

        oie = mod.OpenIE(llm_model=None, max_workers=4)
        r1 = oie.ner("k1", "POISON_ONCE abc")     # 第一次失败 → 重试成功
        r2 = oie.ner("k2", "POISON_ALWAYS xyz")   # 一直失败 → 用尽尝试 + 摘 error + 登记
        with ha._FAIL_LOCK:
            bad = set(ha._OPENIE_FAILS["ner"])
        n_once = calls.count(("k1", "POISON_ONCE abc"))
        n_always = calls.count(("k2", "POISON_ALWAYS xyz"))
        # 反向验证：把「摘 error」关掉，error 就还在 → 官方 batch 级校验必然 raise
        real_clear = ha._clear_openie_error
        ha._clear_openie_error = lambda out: None
        try:
            r3 = oie.ner("k3", "POISON_ALWAYS zzz")
        finally:
            ha._clear_openie_error = real_clear
        neg_ok = bool((r3.metadata or {}).get("error"))
        notes = [f"失败块尝试次数={n_always}（上限 {ha.HIPPORAG_OPENIE_TRIES}）",
                 f"一次失败后成功={n_once == 2}",
                 f"失败块登记={sorted(bad)}",
                 f"error 已摘（官方 batch 不会 raise）="
                 f"{'error' not in (r1.metadata or {}) and 'error' not in (r2.metadata or {})}",
                 f"留痕={bool((r2.metadata or {}).get('factalign_openie_failed'))}",
                 f"反向验证（关掉摘除则 error 仍在）={neg_ok}"]
        ok = (n_always == ha.HIPPORAG_OPENIE_TRIES and n_once == 2 and neg_ok
              and ha.HIPPORAG_OPENIE_TRIES >= 3
              and "error" not in (r1.metadata or {})
              and "error" not in (r2.metadata or {})
              and (r2.metadata or {}).get("factalign_openie_failed") is True
              and bad == {"k2"})
        report("HippoRAG OpenIE 硬失败容错（单块重试 → 摘 error，不打断整轮）",
               OK if ok else FAIL, "；".join(notes))
        return ok
    except Exception as e:                                   # noqa: BLE001
        report("HippoRAG OpenIE 硬失败容错（单块重试 → 摘 error，不打断整轮）",
               FAIL, f"{type(e).__name__}: {e}")
        return False
    finally:
        for k, v in saved.items():
            setattr(ha, k, v)
        sys.path.remove(str(pkg_root / "src"))
        for name in [n for n in sys.modules if n == "hipporag" or n.startswith("hipporag.")]:
            sys.modules.pop(name, None)


def check_corpus_sample_guard(tmp: Path):
    """回归：语料图目录名不含题数，必须靠 meta.sample 拦住「不同 --sample 混用同一张图」。

    300 题和 500 题的语料是**两个不同语料**，混用会让整套结果口径错且毫无提示。
    """
    import json
    import numpy as np
    import corpus

    saved = corpus.CORPUS_DIR
    root = tmp / "fakecorpus"
    (root / "ds" / "v1").mkdir(parents=True, exist_ok=True)
    d = root / "ds" / "v1"
    np.save(str(d / "pvecs.npy"), np.zeros((1, 4), dtype=np.float32))
    for f in ("entities.json", "propositions.json", "chunks.json"):
        (d / f).write_text("[]", encoding="utf-8")
    corpus.CORPUS_DIR = root
    notes = []
    try:
        # ① 旧目录（没有 meta.json）→ 只警告、不抛
        old_ok = False
        try:
            corpus.load_corpus("ds", 300, 1)
            old_ok = True
        except Exception as e:                               # noqa: BLE001
            notes.append(f"旧目录不该抛: {type(e).__name__}: {e}")
        # ② meta 记 500 题、现在要 300 题 → 必须拦下
        (d / "meta.json").write_text(json.dumps(
            {"extract_version": corpus.EXTRACT_VERSION, "chat_model": corpus.CHAT_MODEL,
             "sample": 500}), encoding="utf-8")
        mismatch = ""
        try:
            corpus.load_corpus("ds", 300, 1)
        except FileNotFoundError as e:
            mismatch = str(e)
        # ③ 题数一致 → 放行
        (d / "meta.json").write_text(json.dumps(
            {"extract_version": corpus.EXTRACT_VERSION, "chat_model": corpus.CHAT_MODEL,
             "sample": 300}), encoding="utf-8")
        same_ok = True
        try:
            corpus.load_corpus("ds", 300, 1)
        except Exception as e:                               # noqa: BLE001
            same_ok = False
            notes.append(f"题数一致不该抛: {type(e).__name__}: {e}")
        notes.append(f"旧目录无 meta 只警告={old_ok}")
        notes.append(f"500 题图被 300 题请求拦下={'500 题' in mismatch}")
        notes.append(f"题数一致放行={same_ok}")
        ok = old_ok and ("500 题" in mismatch) and same_ok
        report("语料图题数口径守卫（不同 --sample 不许共用同一张语料图）",
               OK if ok else FAIL, "；".join(notes))
        return ok
    finally:
        corpus.CORPUS_DIR = saved


def check_log_segments(tmp: Path):
    """回归：分段日志必须①**逐行落盘**（不等 close/退出）②文件头自带版本+环境+口径
    ③同一个文件不被 stdout/stderr 两个 tee 写两遍。

    踩过：日志文件是 8KB 块缓冲 → 建图跑完了 build log 还是空的（要等 close/进程退出）；
    以及日志里查不到"哪版代码/prompt、什么模型、哪些环境变量被改过"。
    """
    import common.log_setup as L

    out_bak, err_bak, active_bak = sys.stdout, sys.stderr, L._active_tee
    root_bak = L.ROOT
    notes = []
    try:
        L.ROOT = tmp                                  # 日志写到临时目录
        tee = L.start_log("logtest", "ds", 7, extra="v1r1", segments=True)
        b = tee.logs["build"]
        print("[build] 建图中：普通 print（flush=False）")
        mid = b.read_text(encoding="utf-8")
        live_ok = "建图中：普通 print" in mid        # ① 未 switch 就已落盘
        L.switch_log("query", notes=["=== 本次检索使用的图: fake ==="])
        q = tee.logs["query"]
        btxt, qtxt = b.read_text(encoding="utf-8"), q.read_text(encoding="utf-8")
        hdr_ok = ("[ver] EXTRACT_VERSION" in btxt and "[cfg] chat =" in btxt
                  and "[run] method = logtest" in btxt)
        q_hdr_ok = "[ver] EXTRACT_VERSION" in qtxt and "本次检索使用的图" in qtxt
        # ③ 头/段标记都不该重复（两个 tee 指向同一文件）
        dup_ok = (btxt.count("[run] method = logtest") == 1
                  and qtxt.count("[run] method = logtest") == 1
                  and qtxt.count("段日志开始") == 1)
        notes.append(f"建图未结束已落盘={live_ok}（{len(mid)} 字节）")
        notes.append(f"build 头有版本/环境/口径={hdr_ok}")
        notes.append(f"query 头自带且带图说明={q_hdr_ok}")
        notes.append(f"无重复写入={dup_ok}")
        ok = live_ok and hdr_ok and q_hdr_ok and dup_ok
        report("日志分段（逐行落盘 + 头自带版本/环境 + 不重复）",
               OK if ok else FAIL, "；".join(notes))
        return ok
    except Exception as e:                                   # noqa: BLE001
        report("日志分段（逐行落盘 + 头自带版本/环境 + 不重复）",
               FAIL, f"{type(e).__name__}: {e}")
        return False
    finally:
        for st in (sys.stdout, sys.stderr):
            if isinstance(st, L._SplitTee) and st._file is not None:
                try:
                    st._file.flush()
                    st._file.close()
                except Exception:
                    pass
                st._file = None
        sys.stdout, sys.stderr = out_bak, err_bak
        L._active_tee = active_bak
        L.ROOT = root_bak


def check_tee_fileno(tmp: Path):
    """回归：`_Tee` 必须透传 `fileno()`。

    没有它会炸在 C 层直接取 fd 的地方，`faulthandler.dump_traceback_later()`/
    `enable()` 就是（`AttributeError: '_Tee' object has no attribute 'fileno'`
    会把建图整体搞挂——已经踩过一次）。
    """
    import faulthandler
    from common.log_setup import _Tee

    fh = open(tmp / "tee_probe.log", "a", encoding="utf-8")
    tee = _Tee(sys.__stdout__, fh)
    try:
        fd = tee.fileno()
    except Exception as e:
        report("log_setup._Tee 透传 fileno()", FAIL, f"{type(e).__name__}: {e}")
        fh.close()
        return False
    try:
        # 修复前就是这一行抛 AttributeError（只注册，长 timeout + 立刻取消 → 不产生输出）
        faulthandler.dump_traceback_later(3600, repeat=False, file=tee)
        faulthandler.cancel_dump_traceback_later()
        ok = fd == sys.__stdout__.fileno()
    except Exception as e:
        report("log_setup._Tee 透传 fileno()", FAIL,
               f"faulthandler 注册失败：{type(e).__name__}: {e}")
        fh.close()
        return False
    fh.close()
    report("log_setup._Tee 透传 fileno()", OK if ok else FAIL,
           f"fileno()={fd}（=stdout fd），faulthandler 可注册")
    return ok


def check_terminal_filter():
    """回归：终端过滤器只挡终端那一路，日志文件仍收全量。"""
    import io

    from common import log_setup

    term, logf = io.StringIO(), io.StringIO()
    tee = log_setup._Tee(term, logf)
    log_setup.set_terminal_filter(lambda d: "Orig Question:" not in d and "[proh] " not in d)
    try:
        tee.write("[proh] 内部噪音\n")
        tee.write("Orig Question: x\nLevel 0:\n")
        tee.write("[proh][1/20] EM=True | gold='a' pred='a'\n")
        tee.write("普通输出\n")
    finally:
        log_setup.set_terminal_filter(None)
    t, l = term.getvalue(), logf.getvalue()
    ok = ("[proh] 内部噪音" not in t and "Orig Question" not in t
          and "[proh][1/20]" in t and "普通输出" in t
          and "[proh] 内部噪音" in l and "Orig Question" in l)
    report("日志终端过滤器（只挡终端、文件留全量）", OK if ok else FAIL,
           f"终端={t!r:.60} 文件长度={len(l)}")
    return ok


def check_input_truncation():
    """回归：超长 prompt 在客户端被截断，且**保头保尾、只砍中段**。

    踩过的坑：PRoH 2wiki 的 query prompt 到 36k token > max_model_len 24576 → vLLM 400
    → 整题 unknown。修法是发送前按 max_model_len 截断中段（RAG prompt 的规则在头、
    问题/Answer 在尾，两头都必须留）。
    """
    from common.llm import fit_messages

    head = "RULES: answer with the shortest complete answer.\n"
    mid = "EVIDENCE LINE. " * 4000          # 中段证据（可牺牲）
    tail = "\nQuestion: who?\n\nAnswer:"
    msgs = [{"role": "system", "content": "you are a QA system"},
            {"role": "user", "content": head + mid + tail}]
    out, dropped = fit_messages(msgs, max_new=1024, limit=8000)
    text = out[-1]["content"]
    head_ok = text.startswith(head.strip()[:20])
    tail_ok = text.rstrip().endswith("Answer:")
    ok = dropped > 0 and head_ok and tail_ok and "truncated" in text
    # 不超长时必须原样返回（不许多此一举）
    short = [{"role": "user", "content": "hi"}]
    out2, dropped2 = fit_messages(short, max_new=1024, limit=8000)
    ok = ok and dropped2 == 0 and out2[0]["content"] == "hi"
    report("LLM 输入超长客户端截断（保头保尾）", OK if ok else FAIL,
           f"截掉 {dropped} tokens、头在={head_ok}、尾在={tail_ok}、短 prompt 不动={dropped2 == 0}")
    return ok


def check_embed_prefix_template():
    """回归：查询前缀必须与官方模板**逐字符**一致。

    踩过的坑：qwen3 分支曾写成 `f"...\\nQuery: {text}"`（`Query:` 后多一个空格），
    而**模型随附的** `config_sentence_transformers.json` 给的 query 模板是
    `"Instruct: {task}\\nQuery:"` —— sentence-transformers 把 prompt 与正文**直接拼接**，
    所以实际输入是 `Instruct: {task}\\nQuery:{query}`：`Instruct:` 后有空格、
    **`Query:` 后没有**。f-string 里 `{text}` 前的空格是**字面字符**而不是格式符，
    多这一个空格 tokenizer 就切出不同的 token 序列，查询向量口径就与 ST 不一致；
    而这种事肉眼看代码极难发现（第一遍就漏了）。

    ⚠️ 出处更正：README 的 `get_detailed_instruct()` 写的是 `Query: {query}`（**有**空格），
    与模型随附的 ST 配置**不一致**。本项目采用**后者** —— 那是 ST 实际施加的模板。
    基准字面量取自 `<模型目录>/config_sentence_transformers.json`（本地可核）。

    这里用一个**独立手写的**官方字面量当基准：谁改了 `llm.py` 的前缀拼接，
    这条就会失败，而不是静默漂移。
    """
    import common.llm as L

    saved_fmt = L.EMBED_QUERY_FORMAT
    bad, checked = [], 0
    try:
        if not L.EMBED_QUERY_TEMPLATE:          # 自定义模板时前缀由模板决定，跳过
            # 任务文字用配置里的值，只把**模板形状**写成官方字面量（形状才是会写错的东西）
            L.EMBED_QUERY_FORMAT = "qwen3"
            got = L.query_prefix("probe")
            want = f"Instruct: {L.EMBED_QUERY_TASK}\nQuery:probe"
            checked += 1
            if got != want:
                bad.append(f"qwen3: 实际 {got!r} ≠ 官方 {want!r}")

            L.EMBED_QUERY_FORMAT = "bge"
            got = L.query_prefix("probe")
            # 官方 bge 指令句，注意**结尾有一个空格**（官方 model card 原文）
            want_instr = "Represent this sentence for searching relevant passages: "
            checked += 1
            if got != want_instr + "probe":
                bad.append(f"bge: 实际 {got!r} ≠ 官方 {want_instr + 'probe'!r}")
    finally:
        L.EMBED_QUERY_FORMAT = saved_fmt

    ok = not bad
    report(f"embedding 查询前缀与官方模板逐字符一致（{checked} 种格式）",
           OK if ok else FAIL,
           "qwen3 `Query:`后无空格、bge 指令句结尾带空格 —— 都对"
           if ok else " | ".join(bad))
    return ok


def check_config_injection_ordering(tmp):
    """回归：`--config` 注入必须真的生效（且晚了要报错，不能静默用默认值）。

    踩过的坑：`common/paths.py` 顶层 `from config import ROOT`，而 `--config` 的注入
    路径恰好经过 `common.paths`（config_file → paths）→ `config` 在注入**之前**就被
    加载，于是配置文件里的 CHAT_MODEL/EMBED_DIM/… 全部静默失效：不报错、不告警，
    只有跑完发现口径不对。修法 = paths.py 自己推导目录 + `apply_config_file` 加硬检查。
    """
    import subprocess
    import sys
    import textwrap

    cfg = tmp / "inject_probe.json"
    cfg.write_text(json.dumps({
        "embed_model": "PROBE/embed-model", "embed_dim": 4321,
        "embed_max_tokens": 111, "embed_query_format": "qwen3",
        "methods": {"selfask": {"max_hops": 7}},
    }), encoding="utf-8")

    # ① 正常路径：预扫描 → apply_config_file → import config，注入必须生效
    probe_ok = textwrap.dedent(f"""
        import sys, os
        sys.path.insert(0, r"{ROOT}")
        from common.config_file import apply_config_file
        apply_config_file(r"{cfg}", verbose=False)
        import config as C
        import common.llm as L
        assert C.EMBED_MODEL == "PROBE/embed-model", C.EMBED_MODEL
        assert C.EMBED_DIM == 4321, C.EMBED_DIM
        assert C.EMBED_MAX_TOKENS == 111, C.EMBED_MAX_TOKENS
        assert C.EMBED_QUERY_FORMAT == "qwen3", C.EMBED_QUERY_FORMAT
        assert L.EMBED_DIM == 4321, L.EMBED_DIM
        assert C.CHUNK_TOKENIZER == C.EMBED_TOKENIZER, (C.CHUNK_TOKENIZER, C.EMBED_TOKENIZER)
        print("OK")
    """)
    # ② 顺序错了（config 已先被 import）必须 raise，而不是静默忽略
    probe_late = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, r"{ROOT}")
        import config
        from common.config_file import apply_config_file
        try:
            apply_config_file(r"{cfg}", verbose=False)
        except RuntimeError:
            print("OK")
        else:
            print("NO_GUARD")
    """)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    good = subprocess.run([sys.executable, "-c", probe_ok], capture_output=True,
                          text=True, env=env, cwd=str(ROOT))
    late = subprocess.run([sys.executable, "-c", probe_late], capture_output=True,
                          text=True, env=env, cwd=str(ROOT))
    ok = (good.returncode == 0 and "OK" in good.stdout
          and late.returncode == 0 and "OK" in late.stdout)
    report("配置注入时序（config 不被提前冻结）", OK if ok else FAIL,
           f"注入生效={'OK' in good.stdout}（{good.stderr.strip().splitlines()[-1][:90] if good.returncode else ''}）、"
           f"晚注入被拦={'OK' in late.stdout}")
    return ok


def check_config_keys_are_read(tmp):
    """回归：`configs/*.json` 里的每个键都必须**真的有人读**。

    踩过的坑有两个，症状完全一样（配置写了、静默不生效）：
      ① `config.py` 里 `EXTRACT_CHUNK_TOKENS = 400` 写死，而 `configs/default.json`
         一直在提供 `"extract_chunk_tokens": 400` —— 改配置文件毫无作用；
      ② `methods.<m>.*` 的前缀与方法名不一致（`hypergraphrag` 的旋钮其实叫 `HGR_*`）。
    做法：扫 `configs/*.json` 的所有非 `_` 键 → 算出它要注入的环境变量名 → 在项目源码里
    找 `environ...get("<NAME>"` 。找不到就说明这个键是死键（要么写错、要么代码没接）。
    """
    import json as _json
    import re as _re
    from common.config_file import config_to_env, load_config_file

    files = sorted((ROOT / "configs").glob("*.json"))
    # 预扫源码里所有「读环境变量」的写法：
    #   os.environ.get("X" / os.environ["X" / _e("X" / _ei("X" / _ef("X" / _eb("X"
    # 即：任何以字符串字面量开头、且形如大写环境变量名的调用实参。
    read_keys: set[str] = set()
    for src in list(ROOT.glob("*.py")) + list((ROOT / "common").glob("*.py")) \
            + list((ROOT / "methods").glob("*.py")) + list((ROOT / "tools").glob("*.py")):
        try:
            txt = src.read_text(encoding="utf-8", errors="ignore")
        except Exception:                                        # noqa: BLE001
            continue
        read_keys.update(_re.findall(r'[\(\[,\s]\s*["\']([A-Z][A-Z0-9_]{2,})["\']', txt))
    # 已知"配置里有、但代码不读 os.environ"的键：在 config.py 里是写死的常量
    # （改它要改代码），列出来免得它们盖住真正可疑的键。
    known_not_env = {"BACKEND", "OLLAMA_BASE_URL", "DATA_DIR", "ROOT"}

    dead, checked = [], 0
    for f in files:
        try:
            cfg = load_config_file(f)
        except Exception as e:                                   # noqa: BLE001
            dead.append(f"{f.name}: 读不了（{e}）")
            continue
        # 方法私有键：逐个方法展开（前缀要按 method_prefixes 算）
        methods = list((cfg.get("methods") or {}).keys())
        per_method = []
        for m in methods:
            if str(m).startswith("_"):
                continue
            per_method += list(config_to_env(cfg, method=m).keys())
        top = list(config_to_env(cfg, method="").keys())
        for k in set(top + per_method):
            if k.startswith("_") or k in known_not_env:
                continue
            checked += 1
            if k not in read_keys:
                dead.append(f"{f.name}: {k}")
    # 只报告，不判失败（SKIP 级）：新增键与代码落地允许有短暂先后差。
    # 但**已经接好的键**被改名/删掉时，可疑清单里就会出现它——这才是要防的回归。
    ok = not dead
    report("配置文件里的键都有人读（无死键）", OK if ok else SKIP,
           f"检查 {len(files)} 份配置 / {checked} 个键" + (f"；可疑：{dead}" if dead else ""))
    return True          # 不阻断，但把可疑键打出来


def check_selfask_loop():
    """Self-Ask `query()` 的**上下文累积** + 官方停止条件 + 首轮强制检索分支。

    官方 `promptf()` 的核心是**状态机**：每轮 `cur_prompt += ret_text`，
    检索结果再以 `\\nIntermediate answer: X.` 拼回去，靠模型自己写
    `So the final answer is:` 停。这里用假 LLM 断言三件事：

      ① 多跳：每轮 prompt 必须**以上一轮 prompt 为前缀**（漏累积 = 多跳退化成单跳）；
      ② 配对：`Follow up:` 条数必须等于 `Intermediate answer:` 条数
         （出现「有追问、没答案」的裸 Follow up = 模型下一轮看不到依据）；
      ③ 首轮强制检索分支（模型第一句就写最终答案）：模型原话必须先按官方
         `cur_prompt += ret_text` 拼进 prompt；检索结果**不能**写成
         `Intermediate answer: <内容>.` —— 那会变成没有 Follow up 的悬空答案。
    """
    import methods.selfask_adapter as sa

    saved = {n: getattr(sa, n) for n in ("llm_chat", "generate_answer",
                                        "generate_from_context")}
    PROBLEM = "When was the capital of Freedonia founded?"
    NL = "\n"
    calls: list[str] = []
    cut = {"n": 0}

    def _live(p):
        return p[cut["n"]:]

    def _mk(replies, reactive=None):
        """replies 按 `Intermediate answer: <内容>` 条数索引；reactive 优先（看累积内容）。"""
        def _fake(system, user, **kw):
            live = _live(user)
            n = live.count(sa.INTERMEDIATE + " ")
            calls.append(user)
            if reactive is not None:
                return reactive(live, n)
            return replies[n] if n < len(replies) else replies[-1]
        return _fake

    def _build_state():
        m = sa.SelfAsk()
        m.dataset = "hotpotqa"
        # 块内容要**真能回答**子问题："capital" 那一跳必须选出一句含 capital 的句子，
        # 否则下面 reactive 假模型看到第一跳没答上就不会继续追问（那是模型行为，不是 bug）。
        m._chunks = [("Freedonia", 0, "The capital of Freedonia is Vantry.", [0]),
                     ("Freedonia", 1, "Freedonia was founded in 1904.", [1]),
                     ("Vantry", 2, "Vantry is in the north.", [2])]
        m._texts = [c[2] for c in m._chunks]
        m._toks = [sa._tokenize(t) for t in m._texts]
        m._fewshot, _ = sa.load_official_fewshot()
        cut["n"] = len(m._fewshot.format(PROBLEM) + sa.NEEDED)
        return m

    try:
        sa.generate_answer = lambda q, ev, dataset="", **k: "GROUNDED"
        sa.generate_from_context = lambda *a, **k: "SNIPPET"
        # ── ① ② 多跳累积 + 配对 ──
        calls.clear()
        sa.llm_chat = _mk([" Yes." + NL + "Follow up: What is the capital of Freedonia?",
                           NL + "Follow up: In what year was Freedonia founded?",
                           NL + "So the final answer is: 1904"])
        r = _build_state().query(PROBLEM)
        prefixes_ok = all(calls[i].startswith(calls[i - 1]) for i in range(1, len(calls)))
        pair_ok = True
        for p in calls:
            live = _live(p)
            if live.count("Follow up:") > live.count(sa.INTERMEDIATE):
                pair_ok = False
        multi_ok = r["hops_used"] == 2 and len(r["sub_questions"]) == 2

        # ── ③ 强制检索后**必须发生在上一轮检索结果之上**的下一跳 ──
        #    假模型**看累积内容反应**：只有当前面已经有"有内容的" Intermediate answer、
        #    且该回答里出现了 Freedonia 时，才追问第二个子问题。
        #    → 若收尾时把**最后一次模型输出**丢了（`cur_prompt += ret_text` 漏拼），
        #      模型看到的上下文不自洽，这一跳就不成立；若强制分支把检索结果写成
        #      `Intermediate answer: <内容>.`，模型反而**不会**往下追问。
        calls.clear()

        def _reactive(live, n):
            # 模型只在「prompt 停在等待它续写的位置」时才继续：追问后等着填 Intermediate
            # answer（含强制分支那种**只有提示没有内容**的写法）、或刚填完答案等着它接着问。
            tail = live.rstrip()
            ends_ok = tail.endswith("Are follow up questions needed here:") \
                or tail.endswith("Follow up:") \
                or tail.endswith("Intermediate answer:") \
                or tail.endswith(".")
            if n == 0:
                return " Yes." + NL + "Follow up: What is the capital of Freedonia?"
            last_int = live.rsplit(sa.INTERMEDIATE, 1)[-1]
            # n==1 才允许往下走（确定性：最多两跳），且**第一跳的答案得真答上了**
            # （问"capital"、答里出现"capital"）—— 否则说明累积链断了，模型只能收尾。
            if n == 1 and "capital" in last_int and ends_ok:
                return NL + "Follow up: In what year was Freedonia founded?"
            return NL + "So the final answer is: 1904"

        sa.llm_chat = _mk([], reactive=_reactive)
        r3 = _build_state().query(PROBLEM)
        chain_ok = r3["hops_used"] >= 2 and len(r3["sub_questions"]) >= 2

        # ── ⑤ 首轮强制检索：裸 ret 要先拼进 prompt，且检索结果不写成悬空答案 ──
        calls.clear()
        sa.llm_chat = _mk([" No." + NL + "So the final answer is: 1904"])
        r2 = _build_state().query(PROBLEM)
        p1 = _live(calls[1]) if len(calls) > 1 else ""
        # 强制分支的 prompt 必须**正好停在** `\nIntermediate answer:`（官方「搜索引擎无结果」
        # 写法：只有提示、没有内容）。少了 `ret` → 模型原话没进 prompt；多了 ` <内容>.`
        # → 变成没有 Follow up 的悬空答案。
        forced_ok = (
            len(calls) >= 2
            and "No." in p1                                  # 模型原话被累积（官方 += ret_text）
            and p1.rstrip().endswith("Intermediate answer:")  # 且正好停在无内容写法
            and r2["trace_rounds"]
            and r2["trace_rounds"][0]["query_used"] == PROBLEM  # 记真正检索的那句话
            and r2["trace_rounds"][0].get("forced_query") is True
        )
        # 最终作答仍走统一口径（grounded）
        grounded_ok = r["answer"] == "GROUNDED" and r2["answer"] == "GROUNDED"
    finally:
        for n, v in saved.items():
            setattr(sa, n, v)

    all_ok = (prefixes_ok and pair_ok and multi_ok and chain_ok
              and forced_ok and grounded_ok)
    report("Self-Ask 循环（官方状态机：累积 / 配对 / 首轮强制检索）",
           OK if all_ok else FAIL,
           f"多跳={r['hops_used']}跳 前缀累积={prefixes_ok} Follow up/Intermediate 配对={pair_ok} "
           f"多跳成立={multi_ok} 强制检索后的下一跳={chain_ok}({r3['hops_used']}跳) "
           f"首轮强制检索分支={forced_ok} 最终走统一作答={grounded_ok}")


def check_stats_sees_corpus_in_index():
    """建图凭证的 chunk 计数要能看见「语料包在检索底座里」的方法（s2grag）。

    真实事故：s2grag 只把语料单元存进 `self._index`（`CorpusIndex.units`），**不落盘**
    （`run.py:455-460` 已注明它"纯内存、从不落盘"）。而 `collect_stats` 的探针只认顶层
    list 属性（chunks/passages/_chunks/docs）→ 一个都命不中 → `summary` 为空 →
    验收 `tools/verify_run.py` 同时挂两条：
      「图规模统计非空」FAIL +「chunk 数与 merged 一致」WARN（缺一方 图=None）
    明明检索用的就是这些单元，数得出来 —— 是**凭证收集的盲区**，不是方法的问题。
    """
    from tools.write_build_meta import collect_stats
    from methods.corpus_index import CorpusIndex

    def _units(n):
        return [{"text": f"t{i}", "title": "T", "locs": [], "level": 0, "meta": {}}
                for i in range(n)]

    class _S2G:                       # 形状与 methods/s2grag_adapter.S2GRAG 一致
        name = "s2grag"
        _P = {}

        def __init__(self, n):
            self._index = CorpusIndex(units=_units(n))

    class _TopLevel:                  # 顶层已有 chunks → 不能被 _index 覆盖
        name = "x"

        def __init__(self):
            self.chunks = list(range(100))
            self._index = CorpusIndex(units=_units(5))

    ok = True
    detail = []
    for n in (4956, 7):
        s = (collect_stats(method_obj=_S2G(n)).get("summary") or {}).get("chunks")
        ok = ok and s == n
        detail.append(f"units={n}→chunks={s}")
    s2 = (collect_stats(method_obj=_TopLevel()).get("summary") or {}).get("chunks")
    ok = ok and s2 == 100
    detail.append(f"顶层 chunks=100 优先→{s2}")
    s3 = (collect_stats(method_obj=_S2G(0)).get("summary") or {})
    ok = ok and not s3
    detail.append(f"空语料不编数字→{s3}")
    report("建图凭证能数出「语料存在检索底座里」的 chunk 数（s2grag）",
           OK if ok else FAIL, "；".join(detail))


def check_token_reset_no_lost_accounting():
    """`reset_stats()` 之后，**已用过统计的线程**的 token 不能丢账。

    真实缺陷：桶按线程 thread-local 存、注册表是全局 list。`reset_stats()` 只
    `clear()` 注册表时，各线程的 `_phase_tls.buckets` 还在 → `_get_buckets()` 里
    "桶是不是刚建的" 这个判据不再成立 → 该线程之后的调用**永不重新登记** →
    `get_stats()` 少算它所有 token。这类"丢账"不报错，只让成本表偏小，最难发现。

    修法用**全局世代号**：reset 时 +1，线程发现自己的世代号落后就补登记。
    （光加 per-thread 布尔标记不够 —— 那个标记也在 thread-local 里，主线程改不到别的
    线程，worker 会一直以为自己登记着。这一点在实现时实测踩过。）

    这里锁住两件事：① reset 前已用过的 worker，reset 后新花的 token 仍进 get_stats；
    ② 多次 reset 不重复登记（同一桶只出现一次，否则 get_stats 会把同一桶加两遍）。
    """
    import threading
    import common.llm as L

    saved_vllm = L._chat_vllm

    def _fake(system, user, json_mode, timeout, thinking=True,
              max_tokens=None, flatten=True, stop=None):
        return "out", len(user), 5, "out"

    ok = True
    detail = []
    try:
        L._chat_vllm = _fake
        L.reset_stats()
        started, go = threading.Event(), threading.Event()
        got = {}

        def worker():
            with L.phase("query"):
                L.chat("s", "A" * 100)
            got["first"] = L.get_stats()["query"]["prompt_tokens"]
            started.set()
            go.wait(5)
            with L.phase("query"):
                L.chat("s", "B" * 100)
            got["second"] = L.get_stats()["query"]["prompt_tokens"]

        t = threading.Thread(target=worker, name="token-reset-probe")
        t.start()
        started.wait(5)
        L.reset_stats()                     # worker 已用过统计之后再 reset
        go.set()
        t.join()
        # 判据：worker 第二次调用花的 100 必须**对它自己可见**（get_stats 合并了它的桶）。
        # 修之前 worker 的桶在 reset 后不再登记 → 第二次调用查 get_stats 得 0。
        # 用增量（second - first）：reset 会把 worker 已有的账清零，绝对值会回落。
        ok = ok and got["first"] == 100 and got["second"] == 100
        detail.append(f"reset 前={got['first']} reset 后再花仍可见={got['second']}（两次都须 100）")
        # ② 重复 reset 不应让同一桶进注册表两次
        L.reset_stats()
        L.reset_stats()
        dup = len(L._bucket_registry) != len(set(map(id, L._bucket_registry)))
        ok = ok and not dup
        detail.append(f"重复 reset 后注册表={len(L._bucket_registry)} 无重复={not dup}")
    finally:
        L._chat_vllm = saved_vllm
        L.reset_stats()
    report("reset_stats 不让已用过的线程丢账（跨世代的 token 归属）",
           OK if ok else FAIL, "；".join(detail))


def _tiny_corpus():
    return [
        {"title": "Doc A", "sentences": ["A is a person.", "A was born in B city.",
                                         "B city is in C country."],
         "text": "A is a person. A was born in B city. B city is in C country."},
        {"title": "Doc B", "sentences": ["C country is in Europe.", "D is the capital."],
         "text": "C country is in Europe. D is the capital."},
    ]


def check_staticmethod_no_self():
    """静态方法里读 `self` —— 只能在运行时炸，而且会被 run.py 吞成一行「不可用」。

    真实事故：`proprag_adapter._chunk_corpus` 是 `@staticmethod`，但为了按数据集选切块
    口径加了 `dataset=self.dataset` → 一跑就 `NameError: name 'self' is not defined`，
    `run.py:464` 只打印 `[proprag] 不可用：name 'self' is not defined`（**不写
    traceback、不写结果行**）→ 整张表里 proprag 直接缺行，看不出原因。

    这里用 AST 静态查全仓（不 import，避免官方重依赖）：`@staticmethod` 函数体**自己那层**
    引用 `self` 即报错。**嵌套函数/类的 `self` 不算** —— 那是它们自己的形参
    （如 hypergraphrag 把 `NanoVectorDBStorage.query` 换成 `async def _query_off_loop(self, ...)`，
    以及 lightrag `_BgeInterface.__deepcopy__(self)`，都是合法的）。
    """
    import ast

    class _Scan(ast.NodeVisitor):
        def __init__(self, path):
            self.path = path
            self.bad: list[str] = []

        def _check(self, node):
            decs = {d.id for d in node.decorator_list if isinstance(d, ast.Name)}
            if "staticmethod" not in decs:
                return
            # 只看**本层**语句，不下钻 FunctionDef/Lambda/ClassDef（它们各有自己的 self）
            stack = list(node.body)
            while stack:
                n = stack.pop()
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda,
                                  ast.ClassDef)):
                    continue
                if isinstance(n, ast.Name) and n.id == "self":
                    self.bad.append(f"{self.path}:{node.lineno} {node.name}()")
                    return
                stack.extend(ast.iter_child_nodes(n))

        def visit_FunctionDef(self, node):
            self._check(node)
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    bad: list[str] = []
    for p in sorted(ROOT.glob("methods/*.py")) + sorted(ROOT.glob("*.py")) \
            + sorted(ROOT.glob("common/*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        sc = _Scan(p.relative_to(ROOT))
        sc.visit(tree)
        bad += sc.bad
    report("静态方法里不读 self（否则运行时 NameError 被 run.py 吞成一行「不可用」）",
           OK if not bad else FAIL,
           "全仓 0 处" if not bad else "命中: " + "; ".join(bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="保留 data/_selftest 便于排查")
    args = ap.parse_args()

    note = _stub_pandas_if_broken()
    if note:
        report("环境（pandas 壳）", SKIP, note)

    os.chdir(ROOT)          # 自检里的相对路径（tmp/data 产物）一律相对项目根解析
    tmp = ROOT / "data" / "_selftest"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    ctx = _tiny_corpus()

    ok = check_registry()
    ok = check_dataset_chunking() and ok
    ok = check_hf_endpoint_exported() and ok
    ok = check_prompt_routing() and ok
    ok = check_prompt_not_copied() and ok
    ok = check_call_log_split(tmp) and ok
    ok = check_guards(ctx) and ok
    check_pipelines(ctx, tmp)
    ok = check_hipporag_flow(tmp) and ok
    ok = check_hipporag_json_repair() and ok
    ok = check_hipporag_openie_tolerance(tmp) and ok
    ok = check_corpus_sample_guard(tmp) and ok
    ok = check_log_segments(tmp) and ok
    ok = check_terminal_filter() and ok
    ok = check_input_truncation() and ok
    ok = check_embed_prefix_template() and ok
    ok = check_tee_fileno(tmp) and ok
    ok = check_config_injection_ordering(tmp) and ok
    check_config_keys_are_read(tmp)
    check_selfask_loop()
    check_staticmethod_no_self()
    check_stats_sees_corpus_in_index()
    check_token_reset_no_lost_accounting()

    n_fail = sum(1 for _s, st, _n in _results if st == FAIL)
    n_ok = sum(1 for _s, st, _n in _results if st == OK)
    n_skip = sum(1 for _s, st, _n in _results if st == SKIP)
    print(f"\n合计: OK={n_ok} FAIL={n_fail} SKIP={n_skip}")
    if n_fail:
        print("有 FAIL：注册或框架有问题，先修再往里填官方源码。")
    elif not ok:
        print("有 FAIL（同上）。")
    else:
        print("注册就绪：三个检查全过。接下来按 EXPERIMENT_TABLE.md 的清单填官方源码。")
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

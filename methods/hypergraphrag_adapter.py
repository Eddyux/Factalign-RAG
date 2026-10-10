"""HyperGraphRAG 适配器（语料级：build() 建一次知识超图，query() 逐题检索）。

HyperGraphRAG = LightRAG 衍生的知识超图 RAG（NeurIPS 2025）。适配要点：
- 构造器支持自定义 llm_model_func / embedding_func：chat 走 common.llm.chat_messages
  （本地 vLLM，token 统一计入 build/query 分桶），embedding 走 embed_batch（本地 bge）
- 默认 chunk 1200 token 会撞 bge 512 上限 → 本项目**不切块**（HGR_CHUNK_TOKENS=0 → 2000/0，
  整段一块；同 lightrag）。旧版压到 400 token，现已统一。
- 存储锁绑定 event loop（LightRAG 系通病）：起专用常驻 loop 线程，
  ainsert / aquery 经 run_coroutine_threadsafe 派发到同一个 loop
- NanoVectorDB 的同步 numpy 扫描是并发串行点：monkeypatch 成 asyncio.to_thread
  （query 只读 matrix，线程安全），loop 只做调度
- 口径：QueryParam.response_type 注入短答案指令，对齐 base.py RAG_PROMPT 风格

═══ 与官方的差异清单（论文里要注明；**不改 baselines/ 源码，全是运行时替换/构造参数**）═══
  1. **token 尺子 = 本地 bge**（`_patch_hgr_tokenizer()`）：官方 `chunking_by_token_size`
     用 tiktoken（默认 gpt-4o-mini/o200k_base）计数切块，我们用 bge（项目统一口径）。
     运行时替换官方 `encode_/decode_tokens_by_tiktoken` + `truncate_list_by_token_size`
     的**全部绑定名**（扫所有已加载的 hypergraphrag* 模块；只改 utils 定义处无效，
     因为 operate.py 是 from .utils import ... 的名字拷贝）。换来"我们的整段块
     不会被官方再切"（否则 tiktoken 数超阈的块会被再切一刀、双重重叠）。
     设 `HGR_TOKENIZER=tiktoken` 回到官方原版；`HGR_TRUNCATE_TOKENS`（默认 3000）
     是换 bge 后 truncate 的上限（官方默认 4000 是按 tiktoken 调的）。
  2. **chunk 1200/100 → 不切块（整段）**：一个 context（标题+正文）= 一个块。
     官方默认 chunk 会超 bge 512 上限（vLLM embed 直接 400）。配置层写
     `EXTRACT_CHUNK_TOKENS=0`，`config.py` 归一成 `CHUNK_NO_SPLIT=2000` 再传下来
     —— **不能直接传 0**：官方 `chunking_by_token_size` 的 `range` 步长会变成 0，
     直接 `ValueError: range() arg 3 must not be zero`。
  3. **entity_extract_max_gleaning 2 → 0**：官方每块最多 **4** 次抽取调用
     （首抽 + 2 次 gleaning + 最多 1 次 if_loop，operate.py:326-339）；
     设 0 = 每块 1 次，是最快的建图档位（换更全的抽取改 1）。
  4. **addon_params={"example_number": 1}**：官方默认 `{}` → `example_number=None` → 用
     **全部** few-shot 示例（本 fork 的 prompt.py 里是 3 个）；我们减到 1 个（8B 模型下
     长 few-shot 只增 token/延迟）。注意：这是全方法统一下调示例规模的设定。
  5. **llm_model_max_token_size 32768 → HGR_PROMPT_MAX_TOKENS(16384)**：摘要 prompt 的
     截断上限必须小于 chat 窗口，否则每次摘要都被 vLLM 400（表现为建图卡死、GPU 0%）。
     设 `HGR_PROMPT_MAX_TOKENS=32768` 即对齐官方 fork；**16k 窗口下必须降到 ≤12288**。
  6. **并发/批次上调**：embedding_batch_num 32→256、embedding_func_max_async 16→32、
     llm_model_max_async 16→HGR_MAX_ASYNC（默认 32）。只改调度，不改方法语义。
  7. **embedding 前 `fit_embed()`**：实体/关系描述过长时按 `EMBED_MAX_TOKENS`（4096）截断保住向量。
  8. **建图前把语料整段当 doc 喂入、不让官方再切**（官方是吃整篇文档再按
     chunk_token_size 自己切）；marker/重试/
     响应缓存是适配器新增的工程件（断点续、失败补抽）。
  9. **tiktoken_model_name 显式传值**：只在 `HGR_TOKENIZER=tiktoken` 时生效，避免官方
     改默认值后口径悄悄漂移。

依赖: git clone https://github.com/LHRLAB/HyperGraphRAG.git 到 baselines/
（HyperGraphRAG-main 也行）+ pip install -r 其 requirements.txt
"""

import asyncio
import ast
import csv
import hashlib
import io
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

from methods.base import BaseMethod
from common.llm import (chat_retry, embed_batch, bucket_ctx, cur_phase,
                        fit_embed, gate_for, get_phase_bucket, set_default_phase,
                        split_by_tokens)
from config import (BASELINES_DIR, CHAT_MODEL, EXTRACT_CHUNK_TOKENS,
                    EXTRACT_CHUNK_OVERLAP, EMBED_DIM,
                    normalize_chunk_tokens,
                    EMBED_TOKENIZER as _EMBED_TOKENIZER_DEFAULT,
                    CHUNK_TOKENIZER as _CHUNK_TOKENIZER_DEFAULT,
                    EMBED_MAX_TOKENS as _EMBED_MAX_TOKENS_DEFAULT)

# LLM 并发上限：既传给 fork 的 llm_model_max_async，也当自适应闸的封顶
HGR_MAX_ASYNC = int(os.environ.get("HGR_MAX_ASYNC", "32"))

# 摘要 prompt 的输入截断上限 = 官方 `llm_model_max_token_size`（fork 默认 32768）。
# 必须**小于** chat 窗口（`--max-model-len`），否则每次摘要都 400（vLLM 在 API 层直接拒，
# 引擎不执行，表现为建图卡死 + GPU 0%）。
# 本项目默认 **16384**（原 12000）：描述越长、可进摘要的上下文越多，但每条请求的 KV
# 也随之线性涨（128 KiB/token）→ 并发会降。要在 16k 窗口下跑就设 ≤12288；
# 对齐官方 fork 用 `HGR_PROMPT_MAX_TOKENS=32768`（前提：chat 窗口够大）。
HGR_PROMPT_MAX_TOKENS = int(os.environ.get("HGR_PROMPT_MAX_TOKENS", "16384"))

# ── 官方长度限制旋钮（默认 = 官方仓库原值；对齐实验用 configs/official-alignment.json）──
# 官方 hypergraphrag.py:135 `entity_summary_to_max_tokens = 500`，在 operate.py:64-84
# 身兼三职：① 触发阈值——描述不足 500 token 就**不摘要**（直接返回原文）
#           ② 摘要 prompt 的输入截断 = llm_model_max_token_size（不是这个值）
#           ③ 摘要输出的 max_tokens 上限 = 500
# 之前没显式传，走 fork 的隐式默认 500（值对但不透明）；现在钉死并可调。
HGR_SUMMARY_MAX_TOKENS = int(os.environ.get("HGR_SUMMARY_MAX_TOKENS", "500"))

# 官方 hypergraphrag.py:134 `entity_extract_max_gleaning = 2` → 每块最多 3 次抽取
# 调用（首抽 + 2 次补抽）。本项目默认 0（每块 1 次，建图最快）。
# 对齐官方：`HGR_MAX_GLEANING=2`
HGR_MAX_GLEANING = int(os.environ.get("HGR_MAX_GLEANING", "0"))

# 官方 base.py:28-32 QueryParam 的三个上下文预算，官方均为 4000：
#   max_token_for_text_unit / max_token_for_global_context / max_token_for_local_context
# 不传时走 fork 默认（同样 4000）；显式化以便与官方逐项对账。
HGR_QUERY_MAX_TOKENS = int(os.environ.get("HGR_QUERY_MAX_TOKENS", "4000"))


async def _vllm_complete(prompt, system_prompt=None, history_messages=None,
                         keyword_extraction=False, **kwargs):
    """自定义 llm_model_func（gpt_4o_mini_complete 同款签名）：走 common.llm.chat_messages。

    返回 str；token 经 bucket_ctx 归到发起查询/建索引的线程。
    """
    kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for m in history_messages or []:
        if isinstance(m, dict) and m.get("content") is not None:
            messages.append({"role": m.get("role", "user"), "content": m["content"]})
    messages.append({"role": "user", "content": prompt})

    rf = kwargs.get("response_format")
    # 同 lightrag：抽取解析器吃 tuple 格式，别按 prompt 含 "json" 就强制 JSON 输出，
    # 否则解析器会把 JSON 按 tuple 切碎（0 实体）。只有显式要求时才开。
    json_mode = bool(keyword_extraction
                     or (rf and "json" in str(rf).lower()))

    # 响应级缓存只覆盖 build 阶段：query 阶段的关键词抽取若被缓存，fork 解析
    # 失败返回 fail_response 时重试会命中同一份坏输出，重试就失效了
    key = None
    if _LLM_CACHE is not None and cur_phase() == "build":
        key = hashlib.sha1(json.dumps(
            {"model": CHAT_MODEL, "json_mode": json_mode, "msgs": messages,
             "max_tokens": kwargs.get("max_tokens")},
            ensure_ascii=False).encode("utf-8")).hexdigest()
        hit = _LLM_CACHE.get(key)
        if hit is not None:
            return hit[0]

    content, _pt, _ct = await chat_retry(
        messages, gate=gate_for(HGR_MAX_ASYNC, "hypergraphrag"),
        json_mode=json_mode, timeout=300)
    if content is None:
        raise VLLMTimeoutError()
    if key is not None:
        with _LLM_CACHE_LOCK:
            _LLM_CACHE[key] = (content, _pt, _ct)
            with open(_LLM_CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"k": key, "content": content,
                                    "pt": _pt, "ct": _ct},
                                   ensure_ascii=False) + "\n")
    return content

# 建索引失败兜底重试次数（响应级缓存 llm_cache.jsonl 让重试只补抽失败部分）
# ⚠️ 这里**必须**读环境变量：`configs/baselines.json` 的 `hgr.build_retries` 会
#    展开成 `HGR_BUILD_RETRIES` 注入。以前这里是硬编码字面量 3，导致那个配置键
#    成了**死键**（配置改了不生效，冒烟测试的死键检查会报它）。
HGR_BUILD_RETRIES = int(os.environ.get("HGR_BUILD_RETRIES", "3"))


class VLLMTimeoutError(RuntimeError):
    """vLLM 调用 5 次均超时失败（基础设施问题；区别于方法自身的解析失败）。"""

# 响应级 LLM 缓存（模块级：_vllm_complete 是模块级函数，无实例上下文）。
# fork 的 extract_entities 把全部 chunk 一次性并发抽取，一个失败整个 ainsert
# 白跑；缓存让重跑时已成功的抽取直接命中，只补抽失败的 chunk。
_LLM_CACHE = None          # dict: key -> (content, pt, ct)
_LLM_CACHE_PATH = None     # Path
_LLM_CACHE_LOCK = threading.Lock()


def _init_llm_cache(working_dir: Path):
    global _LLM_CACHE, _LLM_CACHE_PATH
    _LLM_CACHE_PATH = Path(working_dir) / "llm_cache.jsonl"
    _LLM_CACHE = {}
    if _LLM_CACHE_PATH.exists():
        try:
            for line in _LLM_CACHE_PATH.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    _LLM_CACHE[d["k"]] = (d["content"], d["pt"], d["ct"])
        except Exception:
            _LLM_CACHE = {}


# ══════════════════════════════════════════════════════════════════════════════
# 统一尺子：把官方内部的 tiktoken 换成**项目的 bge tokenizer**
# ══════════════════════════════════════════════════════════════════════════════
# 项目口径是**不切块**：一个 context（标题+正文）= 一个整段块（所有方法同粒度 =
# 可比性前提）。配置层写 `EXTRACT_CHUNK_TOKENS=0`，`config.py` 归一成
# `CHUNK_NO_SPLIT=2000` 后才传下来 —— **不能直接传 0**：官方
# `chunking_by_token_size`（operate.py:35-53）是
#     tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
#     for start in range(0, len(tokens), max_token_size - overlap_token_size): ...
# 步长 `0 - 0 = 0` → `ValueError: range() arg 3 must not be zero`，建索引直接崩。
#
# 但光传大值还不够：官方 `chunking_by_token_size` 会拿每篇 doc 再用 **tiktoken**
# 重新量一遍，两把尺子读数不同（同一段文本可能 1800，也可能 2300）→ 块一旦超过
# 我们传的 2000 就会被官方**再切一刀**：索引块数 ≠ 喂入 docs 数、两套重叠叠加、
# 抽取调用变多，而且那部分块的检索单元变成"半块"，与其它方法的粒度不一致。
#
# 修法（就是现在这套）：**把官方那把尺子换成 bge**——patch 掉官方的
# `encode_string_by_tiktoken` / `decode_tokens_by_tiktoken` / `truncate_list_by_token_size`
# 三个调用点（utils 定义处 + operate 的 import 绑定名都要换，见 `_patch_hgr_tokenizer`）。
# 之后官方内部量的就是 bge token，与我们同一把尺子：`range(0, len, 2000)` 对
# ≤2000 bge 的块只有 start=0 → **1 doc = 1 chunk**。
#
# 这是**对官方实现的改动**（论文里要注明）：改的只是分词器，切块算法/步长/重叠
# 参数的语义不变。
#
# 两条保险（打印，不拦）：
#   · 建索引前 `measure_rechunk_risk()` 复量一遍超阈比例
#   · 建索引后 `verify_chunk_parity()` 对账官方落盘块数 == 喂入 docs 数
#
# 官方 chunk_token_size / chunk_overlap_token_size 默认 1200/100；本项目统一改用
# `config.py` 的 `EXTRACT_CHUNK_TOKENS` / `EXTRACT_CHUNK_OVERLAP`（默认 0/0 →
# 归一成 2000/0）。
# ★ 默认**派生自全局 config.py** —— 改 config.py 一处，所有方法 + HGR 内部的官方切块器
#   一起变（"单一尺子 + 单一语料单元"的前提）。
# ⚠️ 这两个常量必须与传给官方的 `chunk_token_size`/`chunk_overlap_token_size` **逐字一致**，
#   否则我们切的块会被官方按它的阈值再切一刀（双重重叠）。
HGR_CHUNK_TOKENS = normalize_chunk_tokens(
    int(os.environ.get("HGR_CHUNK_TOKENS", str(EXTRACT_CHUNK_TOKENS))))
HGR_CHUNK_OVERLAP = int(os.environ.get("HGR_CHUNK_OVERLAP", str(EXTRACT_CHUNK_OVERLAP)))
# ── 被 monkey-patch 替换的 `truncate_list_by_token_size` 的**默认参数** ──
# ⚠️ 这个默认值**在官方调用点上从不生效**：`operate.py:871/913/960/1049/1097` 五处
#    每一个都**显式传** `max_token_size=query_param.max_token_for_*`。
#    所以**真正生效的是 `HGR_QUERY_MAX_TOKENS`（= 那三个 QueryParam 预算）**，不是这个值。
#    默认改成跟随 `HGR_QUERY_MAX_TOKENS`：一来消除"两个旋钮两个数"的歧义，
#    二来万一将来某个调用点漏传，拿到的也是同一个数，而不是一个悄悄不同的 3000。
# 官方原默认是 4000（按 tiktoken 调的）；改 `HGR_QUERY_MAX_TOKENS` 即改实际口径。
HGR_TRUNCATE_TOKENS = int(os.environ.get("HGR_TRUNCATE_TOKENS", str(HGR_QUERY_MAX_TOKENS)))
# ── 尺子 / 向量（换 embedding 模型时要一起调的旋钮）──
# HGR_EMBED_DIM：传给官方 `EmbeddingFunc(embedding_dim=)`，必须 = 服务端真实维度
#   （默认跟随全局 `EMBED_DIM`）。
# HGR_EMBED_MAX_TOKENS：官方第二次切/嵌前的单条上限（bge 512 上限 → 默认 480，
#   跟随全局 `EMBED_MAX_TOKENS`）。
# HGR_TOKENIZER：官方内部计数/切块用哪把尺子。默认跟随全局 `EMBED_TOKENIZER`
#   （即默认 `bge`=EMBED_MODEL 的 tokenizer）。设 `tiktoken` = **官方原版口径**
#   （此时 HGR_TRUNCATE_TOKENS 应按官方的 4000 设，见 `_patch_hgr_tokenizer`）。
HGR_EMBED_DIM = int(os.environ.get("HGR_EMBED_DIM", str(EMBED_DIM)))
# 传给官方 `EmbeddingFunc(max_token_size=)` 的声明值。**截断本身由全局 `fit_embed()`
# 按 `EMBED_MAX_TOKENS` 执行**（那是 embedder 的真实单条上限，不该在适配器里再分叉），
# 这里只是把同一档位告诉官方，让它内部的 512 闸门跟着换。
HGR_EMBED_MAX_TOKENS = int(os.environ.get("HGR_EMBED_MAX_TOKENS",
                                          str(_EMBED_MAX_TOKENS_DEFAULT)))


def _tiktoken_model_name() -> str:
    """只在 tiktoken 未被 patch 掉（如 `HGR_TOKENIZER=tiktoken`）时使用的模型名。

    默认 "gpt-4o-mini" = 官方 `hypergraphrag.py:131` 默认值；显式传出去避免官方
    改默认值后口径漂移。默认路径下官方那三个函数已被 bge 替换，这个值不起作用。
    """
    return os.environ.get("HGR_TIKTOKEN_MODEL", "gpt-4o-mini")


def _tiktoken_encoding():
    """tiktoken encoding（**仅用于 `HGR_TOKENIZER=tiktoken` 的对账/实测**）。"""
    try:
        import tiktoken
    except Exception:
        return None
    try:
        return tiktoken.encoding_for_model(_tiktoken_model_name())
    except Exception:
        return None


def hgr_tokenizer_mode() -> str:
    """官方内部切块/计数用哪把尺子（默认跟随全局 `CHUNK_TOKENIZER` = bge）。

    `tiktoken` = 官方原版口径（用于与官方数字对账；此时 `HGR_TRUNCATE_TOKENS`
    也应设回官方的 4000，否则上下文量口径混着两把尺子）。
    """
    return os.environ.get("HGR_TOKENIZER", _CHUNK_TOKENIZER_DEFAULT).lower()


def _patch_hgr_tokenizer() -> bool:
    """把官方 hypergraphrag 的 token 计数/切块换成 bge tokenizer。返回是否成功。

    替换策略：**扫描所有已加载的 `hypergraphrag*` 模块，凡是仍指向原函数的绑定名
    一并替换**。为什么不能只改 `utils` 的定义处——`operate.py` 是
    `from .utils import encode_string_by_tiktoken, ...` 把名字**复制**进自己的
    命名空间，改 utils 对它无效（官方 fork 里还有 evaluation/ 副本，将来也可能
    多出别的调用点，逐个手改容易漏）。

    语义等价性：bge 与 tiktoken 都是"分词 → 计数 → 切片 → decode 回文本"，
    切片算法完全一致；换的只是 token 边界，保证与我们的预切块同尺子。

    **不改 baselines/ 里的源码**，纯运行时替换。
    """
    if hgr_tokenizer_mode() != "bge":
        return False
    try:
        import hypergraphrag.utils as _ut      # noqa: F401  确保模块已加载
    except ImportError:
        return False
    try:
        from common.llm import _get_tokenizer, fit_embed
    except Exception:
        return False
    try:
        tok = _get_tokenizer()
    except Exception as e:
        print(f"[hypergraphrag] ⚠ bge tokenizer 加载失败（{type(e).__name__}: {e}）→ "
              f"官方内部继续用 tiktoken，存在块被再切的风险", flush=True)
        return False

    # bge 词表可能没有某些生僻字符 → 统一过一遍 fit_embed（它按 bge 切），再 encode 成 ids
    def _encode(content: str, model_name: str | None = None) -> list[int]:
        return tok.encode(fit_embed(content, 10 ** 9))

    def _decode(tokens: list[int], model_name: str | None = None) -> str:
        return tok.decode(tokens)

    def _truncate(list_data: list, key, max_token_size: int = HGR_TRUNCATE_TOKENS):
        """逐项累加 token 数，超上限即停（官方 `utils.py:206-215` 的同款语义）。

        `max_token_size` 由官方调用点显式传入（= `QueryParam.max_token_for_*`
        = 我们的 `HGR_QUERY_MAX_TOKENS`）；这里的默认值只是兜底，见上面常量注释。
        """
        if max_token_size <= 0:
            return []
        out, total = [], 0
        for data in list_data:
            n = len(_encode(key(data)))
            if total + n > max_token_size:
                break
            out.append(data)
            total += n
        return out

    pairs = ((_ut.encode_string_by_tiktoken, _encode),
             (_ut.decode_tokens_by_tiktoken, _decode),
             (getattr(_ut, "truncate_list_by_token_size", None), _truncate))
    pairs = tuple((o, n) for o, n in pairs if o is not None)
    n_repl = 0
    for mod in list(sys.modules.values()):
        name = getattr(mod, "__name__", "") or ""
        if not name.startswith("hypergraphrag"):
            continue
        try:
            items = list(vars(mod).items())
        except TypeError:
            continue
        for k, v in items:
            for orig, new in pairs:
                if v is orig:
                    try:
                        setattr(mod, k, new)
                        n_repl += 1
                    except Exception:          # noqa: BLE001
                        pass
    print(f"[hypergraphrag] 官方 token 尺子=tiktoken→bge（替换 {n_repl} 处绑定名，"
          f"不等同于只改 utils）→ 与预切块 {HGR_CHUNK_TOKENS}/{HGR_CHUNK_OVERLAP} 同尺子，"
          f"官方内部 1 doc=1 chunk", flush=True)
    return n_repl > 0


def prechunk_corpus(corpus_contexts: list[dict], verbose: bool = True) -> tuple[list[str], str]:
    """**按项目统一口径预切块**：`title + text` → 按 `HGR_TOKENIZER` 的尺子切 `HGR_CHUNK_*`。

    ★ 尺寸与尺子都走 HGR **自己的**方法级旋钮，**不走**全局 `EXTRACT_CHUNK_*` /
    `CHUNK_TOKENIZER`。原因：全局键服务「所有方法同一语料单元」，而 HGR 的块必须与
    **官方内部** `chunk_token_size` + 计数尺子严格一致——否则我们按 400 切的块会被官方
    按它的阈值再切一刀 → 双重重叠。用**同一个旋钮**做这两件事就不可能错配。
    「官方内部会不会再切」由 `measure_rechunk_risk()` 实测、`verify_chunk_parity()` 建完对账。

    返回 (docs, 口径标签)；标签写进 marker，切块参数变了才重建。
    """
    ruler = hgr_tokenizer_mode()
    docs: list[str] = []
    for c in corpus_contexts:
        full = f"{c.get('title', '')}\n{c.get('text', '')}".strip()
        for chunk_text, _cs, _ce in split_by_tokens(full, HGR_CHUNK_TOKENS,
                                                    HGR_CHUNK_OVERLAP, kind=ruler):
            if chunk_text.strip():
                docs.append(chunk_text.strip())
    tag = f"{ruler}:{HGR_CHUNK_TOKENS}/{HGR_CHUNK_OVERLAP}"
    if verbose:
        print(f"[hypergraphrag] 预切块={tag} → {len(docs)} 块（HGR 方法级口径）", flush=True)
    return docs, tag


def measure_rechunk_risk(docs: list[str], sample: int = 0) -> str:
    """复量"官方会不会在我们的块上再切一刀"——现在两边同尺子，期望 0%。

    用 `HGR_TOKENIZER` 那把尺子量（与 `prechunk_corpus` 同一把）：我们的块是
    `split_by_tokens(..., HGR_CHUNK_TOKENS, HGR_CHUNK_OVERLAP, kind=ruler)` 切的，
    每块数必然 ≤ `HGR_CHUNK_TOKENS` → 不会超阈。
    （`docs` 是预切块结果，逐块复核一遍，防止上游有人手改过文本。）
    """
    writer = hgr_tokenizer_mode()
    try:
        from common.llm import _get_tokenizer
        tok = _get_tokenizer(writer)
    except Exception as e:
        return f"复核跳过：{writer} tokenizer 加载失败（{type(e).__name__}）"
    import random
    idx = list(range(len(docs)))
    if sample and sample < len(docs):
        random.seed(0)
        idx = sorted(random.sample(idx, sample))
    over, worst = 0, 0
    for i in idx:
        n = len(tok.encode(docs[i], add_special_tokens=False))
        worst = max(worst, n)
        if n > HGR_CHUNK_TOKENS:
            over += 1
    if over:
        return (f"⚠ 复核异常：{over}/{len(idx)} 块超过 {HGR_CHUNK_TOKENS} {writer} token"
                f"（最大 {worst}）——预切块上游有问题，需排查")
    return (f"复核 OK（{len(idx)} 块，{writer} 尺子）：无块超 {HGR_CHUNK_TOKENS}，最大 {worst} "
            f"→ 官方内部不会重切")


def verify_chunk_parity(chunk_store: Path, expected: int) -> str:
    """对账：官方落盘的实际块数 vs 我们喂进去的 docs 数。

    相等 → 官方内部**没有**再切（1 doc = 1 chunk），口径按预期；
    不等 → 官方把某些块又切了（双重重叠），必须按差值排查（通常是预切块尺子没对上）。
    """
    if not chunk_store.exists():
        return "对账跳过：还没有 kv_store_text_chunks.json"
    try:
        stored = json.loads(chunk_store.read_text(encoding="utf-8"))
    except Exception as e:
        return f"对账失败：读不了 {chunk_store.name}（{type(e).__name__}）"
    n = len(stored)
    if n == expected:
        return f"对账 OK：官方落盘块数 {n} == 喂入 docs {expected}（官方内部未再切）"
    return (f"⚠ 对账不符：官方落盘块数 {n} != 喂入 docs {expected}"
            f"（差 {n - expected:+d}）→ 官方内部把部分块又切了一刀（双重重叠）："
            f"这部分检索单元与本项目语料单元口径不一致，处理选项见本文件头说明")


def _parse_hgr_context(context: str) -> tuple[list, list]:
    """从 `only_need_context=True` 拿到的 context 里取出**检索命中的实体与超边**。

    官方拼装处 `operate.py:696-722`，两个 CSV 的表头在 `:775` / `:787`：
        -----Entities-----
        ```csv
        id,entity,type,description           ← 实体名 = 第 2 列（下标 1）
        0,Paris,city,...
        ```
        -----Relationships-----
        ```csv
        id,hyperedge,related_entities        ← HyperGraphRAG 是**超边**，不是二元关系
        0,"(Paris, France, ...)",['Paris', 'France']
        ```
        -----Sources-----
    返回 `(entities, paths)`：`entities` = 命中的实体名（去重保序），
    `paths` = 命中的超边 `[{"hyperedge": 描述, "related_entities": 关联实体}]`。

    ⚠️ 这段是**解析而非复制官方算法**：官方 `only_need_context` 只返回拼好的字符串，
    结构化数据在 `_build_query_context` 内部、没有对外接口可拿。解析失败一律返回空
    （绝不影响答题主流程）。
    """
    def _rows(section: str) -> list:
        seg = context.split(section, 1)
        if len(seg) < 2:
            return []
        body = seg[1].split("-----", 1)[0]        # 截到下一个段标记
        out = []
        try:
            for row in csv.reader(io.StringIO(body)):
                if not row:
                    continue
                head = row[0].strip().lstrip("`").strip()   # 跳过 ```csv 与表头 id
                if not head or head in ("csv", "id"):
                    continue
                out.append([c.strip() for c in row])
        except Exception:                                  # noqa: BLE001
            return out
        return out

    def _as_list(s: str) -> list:
        """官方把 `e["related_nodes"]` 原样写进 CSV 第三列（operate.py:787），
        值是 **Python list 的 repr**（如 `"['Paris', 'France']"`）→ 还原成真 list。"""
        t = (s or "").strip()
        if not t:
            return []
        try:
            v = ast.literal_eval(t)
            if isinstance(v, (list, tuple)):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:                                      # noqa: BLE001
            pass
        return [t]

    entities = list(dict.fromkeys(
        r[1] for r in _rows("-----Entities-----") if len(r) >= 2 and r[1]))
    paths = []
    for r in _rows("-----Relationships-----"):
        if len(r) >= 2 and r[1]:
            paths.append({"hyperedge": r[1][:300],             # 超边描述可能很长，截断
                          "related_entities": _as_list(r[2] if len(r) >= 3 else "")})
    return entities, paths


class HyperGraphRAG(BaseMethod):
    name = "hypergraphrag"

    def __init__(self):
        self._rag = None
        self._loop = None
        self._loop_thread = None

    # ── 专用常驻 event loop（LightRAG 系存储锁绑定 loop，不能 asyncio.run 用完就关）──

    def _ensure_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()

            def _run():
                asyncio.set_event_loop(self._loop)
                # 同 lightrag：默认 executor 只有 min(32, cpu+4) 个 worker，
                # 阻塞 HTTP 走 asyncio.to_thread → 实际并发被卡住（vLLM KV 只用 ~10%）。
                # 放大到 128，并发上限交回 llm_model_max_async / vLLM KV。
                from concurrent.futures import ThreadPoolExecutor
                self._loop.set_default_executor(
                    ThreadPoolExecutor(max_workers=128, thread_name_prefix="hgr-http"))
                self._loop.run_forever()

            self._loop_thread = threading.Thread(target=_run, daemon=True,
                                                 name="hypergraphrag-loop")
            self._loop_thread.start()
        return self._loop

    def _dispatch(self, coro):
        """派发到常驻 loop 并等结果；统计归调用线程（bucket_ctx 随 contextvars 复制）。"""
        loop = self._ensure_loop()
        token = bucket_ctx.set(get_phase_bucket())
        try:
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        finally:
            bucket_ctx.reset(token)

    def _find_repo(self) -> Path:
        for n in ("HyperGraphRAG", "HyperGraphRAG-main"):
            p = BASELINES_DIR / n
            if p.exists():
                return p
        raise RuntimeError(
            f"未找到 {BASELINES_DIR}/HyperGraphRAG（或 -main）。请先 "
            f"`cd baselines && git clone https://github.com/LHRLAB/HyperGraphRAG.git`"
        )

    # ── VDB 查询挪出 loop 线程（多并发查询的串行点之一）──

    _vdb_query_patched = False

    @staticmethod
    def _patch_vdb_query_to_threadpool():
        """把 NanoVectorDB 的同步 numpy 全量扫描挪出常驻 loop 线程。

        --workers 的并发题全部共用一个 loop；VDB query 里 _client.query 是同步
        CPU 扫描（nano-vectordb 0.0.4 矩阵余弦，万级向量 × 1024 维），跑在 loop
        线程上会让并发题互相阻塞。patch 成 asyncio.to_thread：query 只读 matrix
        （线程安全），loop 只做调度，扫描在线程池并行。
        """
        if HyperGraphRAG._vdb_query_patched:
            return
        from hypergraphrag.storage import NanoVectorDBStorage

        async def _query_off_loop(self, query: str, top_k=5):
            embedding = await self.embedding_func([query])
            embedding = embedding[0]
            results = await asyncio.to_thread(
                self._client.query,
                query=embedding,
                top_k=top_k,
                better_than_threshold=self.cosine_better_than_threshold,
            )
            return [{**dp, "id": dp["__id__"], "distance": dp["__metrics__"]}
                    for dp in results]

        NanoVectorDBStorage.query = _query_off_loop
        HyperGraphRAG._vdb_query_patched = True

    # ── 静音 kg_query 的关键词输出刷屏（保留进度条）──

    _print_patched = False
    _orig_operate_print = None

    @staticmethod
    def _patch_print():
        """fork 的 kg_query 里 print(final_result) 会把每题的关键词抽取输出整段
        打到终端（16 并发 × 300 题 = 满屏刷）。替换 operate 模块的 print：
        - 抽取进度 ticker（消息以 \r 结尾）保留（建索引进度条）
        - 报错打印保留
        - 其余（关键词抽取输出）静音。tqdm 走 stderr，不受影响。
        """
        if HyperGraphRAG._print_patched:
            return
        import builtins
        import hypergraphrag.operate as _op
        # print 是 builtin，不在 operate 模块的命名空间里（直接 _op.print 会
        # AttributeError）；取原函数用 getattr 兜底，再设置模块全局 print 遮蔽
        # builtins——operate.py 内的 print 调用会解析到模块属性
        HyperGraphRAG._orig_operate_print = getattr(_op, "print", builtins.print)

        def _filtered(*args, **kwargs):
            s = str(args[0]) if args else ""
            if s.endswith("\r") or "error" in s.lower():
                HyperGraphRAG._orig_operate_print(*args, **kwargs)
            # 其余（kg_query 的关键词抽取输出）静音

        _op.print = _filtered
        HyperGraphRAG._print_patched = True

    # ── 建索引 ────────────────────────────────────────────────

    def load(self, working_dir: str = "data/hypergraphrag_corpus"):
        """只加载已有索引，直接进查询。不做任何建图/断点/重建判断。"""
        working_dir = Path(working_dir)
        if not (working_dir / "kv_store_full_docs.json").exists():
            raise RuntimeError(
                f"[hypergraphrag] 索引不存在：{working_dir}\n"
                f"请先跑：python run.py --method hypergraphrag --dataset ... 建索引")
        repo = self._find_repo()
        import sys
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            from hypergraphrag import HyperGraphRAG as _HGR
            from hypergraphrag.utils import EmbeddingFunc
        except ImportError as e:
            raise RuntimeError(
                "HyperGraphRAG 依赖未装：cd baselines/HyperGraphRAG-main && "
                f"pip install -r requirements.txt（{e}）"
            ) from e

        self._patch_vdb_query_to_threadpool()
        self._patch_print()

        _hg = logging.getLogger("hypergraphrag")
        _hg.propagate = False
        if not any(isinstance(h, logging.StreamHandler) for h in _hg.handlers):
            _hg_ch = logging.StreamHandler()
            _hg_ch.setLevel(logging.WARNING)
            _hg_ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            _hg.addHandler(_hg_ch)

        async def embed_func(texts):
            if isinstance(texts, str):
                texts = [texts]
            texts = [fit_embed(t) for t in texts]
            return await asyncio.to_thread(embed_batch, texts)

        # ★ 把官方内部的 token 尺子换成 bge（与我们的预切块同尺子）——必须在建实例前
        _patch_hgr_tokenizer()
        self._rag = _HGR(
            working_dir=str(working_dir),
            llm_model_func=_vllm_complete,
            llm_model_name=CHAT_MODEL,
            embedding_func=EmbeddingFunc(embedding_dim=HGR_EMBED_DIM,
                                         max_token_size=HGR_EMBED_MAX_TOKENS,
                                         func=embed_func),
            chunk_token_size=HGR_CHUNK_TOKENS,
            chunk_overlap_token_size=HGR_CHUNK_OVERLAP,
            # 官方那三个计数函数已被 patch 成 bge；这个值只在 HGR_TOKENIZER=tiktoken 时生效
            tiktoken_model_name=_tiktoken_model_name(),
            embedding_batch_num=256,
            embedding_func_max_async=32,
            # 摘要 prompt 的截断上限必须 < chat 窗口，否则每次摘要都 400（见文件头常量）
            llm_model_max_token_size=HGR_PROMPT_MAX_TOKENS,
            # 加载已有索引时的实例：并发与 build 保持一致（8B 模型 8 并发稳）
            llm_model_max_async=HGR_MAX_ASYNC,
        )
        return self

    def build(self, corpus_contexts: list[dict], working_dir: str = "data/hypergraphrag_corpus"):
        """N 题上下文合成语料，建知识超图。working_dir 已完整存在则直接加载。"""
        repo = self._find_repo()
        import sys
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            from hypergraphrag import HyperGraphRAG as _HGR
            from hypergraphrag.utils import EmbeddingFunc
        except ImportError as e:
            raise RuntimeError(
                "HyperGraphRAG 依赖未装：cd baselines/HyperGraphRAG-main && "
                f"pip install -r requirements.txt（{e}）"
            ) from e

        # VDB 同步扫描挪出 loop 线程（多并发查询的串行点），建实例前 patch
        self._patch_vdb_query_to_threadpool()
        # 静音 kg_query 的关键词输出刷屏（保留 tqdm 与 \r 进度 ticker）
        self._patch_print()

        # HyperGraphRAG 是旧版 LightRAG fork，本身用 tqdm 进度条（已经是进度条，不用改）；
        # 只需把它的 logger.info 从终端静音（否则 propagate 到 root 会逐行刷屏），
        # 报错（WARNING/ERROR）照常打印到终端。
        _hg = logging.getLogger("hypergraphrag")
        _hg.propagate = False
        _hg_ch = logging.StreamHandler()
        _hg_ch.setLevel(logging.WARNING)
        _hg_ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        _hg.addHandler(_hg_ch)

        async def embed_func(texts):
            if isinstance(texts, str):
                texts = [texts]
            # 实体/关系描述可能超 bge 512 上限（vLLM 400 拒绝 → 该实体向量被丢弃），
            # 嵌入前截断到 480，保住向量
            texts = [fit_embed(t) for t in texts]
            return await asyncio.to_thread(embed_batch, texts)

        # 语料整段成块（不切块，项目统一口径），见文件头说明。
        # 先切块：补写 marker 前要按块数判断索引是否完整
        docs, _tok_tag = prechunk_corpus(corpus_contexts)
        # ★ 预切块之后立刻把官方内部那把尺子也换成 bge（必须在 ainsert 前、建实例前）
        _patch_hgr_tokenizer()
        # 复核：两边同尺子后不该有块超阈（期望 0%）
        print(f"[hypergraphrag] {measure_rechunk_risk(docs)}", flush=True)

        working_dir = Path(working_dir)
        # 完成标记：插入即写盘，中断留下的半截索引会被当成完整索引加载；
        # 切块/抽取口径变了 → 索引口径变了 → 重建。签名必须涵盖**所有会改变抽取结果**的
        # 旋钮，否则换了参数还会静默复用旧图（踩过：prompt_max 12000→16384 不影响旧签名）。
        marker = working_dir / "build_complete.marker"
        marker_sig = (f"chunk={HGR_CHUNK_TOKENS}/{HGR_CHUNK_OVERLAP}"
                      f"|ruler={hgr_tokenizer_mode()}"
                      f"|prompt_max={HGR_PROMPT_MAX_TOKENS}"
                      f"|glean={HGR_MAX_GLEANING}"
                      f"|summary_max={HGR_SUMMARY_MAX_TOKENS}")
        # 图文件完整但缺 marker（上次 ainsert 末尾异常/中断）→ 块数与本次语料一致才
        # 补写（防半截索引被当完整）；对不上会掉进下方重建分支
        # fork 存储布局：full_docs/text_chunks 是 JSON KV（kv_store_*.json），
        # 实体/超边在 NanoVectorDB（vdb_entities.json / vdb_hyperedges.json，
        # 不是 kv_store_full_entities.json —— 那个文件不存在，写错了补写分支永不触发）
        _graph_files = ["kv_store_full_docs.json", "vdb_entities.json",
                        "vdb_hyperedges.json", "graph_chunk_entity_relation.graphml"]
        if working_dir.exists() and not marker.exists() and \
                all((working_dir / f).exists() for f in _graph_files):
            try:
                n_stored = len(json.loads(
                    (working_dir / "kv_store_full_docs.json").read_text(encoding="utf-8")))
            except Exception:
                n_stored = -1
            if n_stored == len(set(docs)):
                marker.write_text(marker_sig, encoding="utf-8")
                print(f"[hypergraphrag] 检测到完整图但缺 marker，补写 marker（跳过重建）")
        if working_dir.exists():
            old_sig = marker.read_text(encoding="utf-8") if marker.exists() else ""
            if old_sig != marker_sig:
                print(f"[hypergraphrag] 索引状态变化（{old_sig or '未完成'} → {marker_sig}），清除 {working_dir} 重建")
                shutil.rmtree(working_dir, ignore_errors=True)
        working_dir.mkdir(parents=True, exist_ok=True)

        _init_llm_cache(working_dir)

        self._rag = _HGR(
            working_dir=str(working_dir),
            llm_model_func=_vllm_complete,
            llm_model_name=CHAT_MODEL,
            embedding_func=EmbeddingFunc(embedding_dim=HGR_EMBED_DIM,
                                         max_token_size=HGR_EMBED_MAX_TOKENS,
                                         func=embed_func),
            # 与自带方法同口径：默认**不切块**（0 → CHUNK_NO_SPLIT=2000、overlap 0，整段一块）
            chunk_token_size=HGR_CHUNK_TOKENS,
            chunk_overlap_token_size=HGR_CHUNK_OVERLAP,
            # ★ 显式钉死官方内部切块用的 tokenizer（不吃官方默认），保证与预切块同尺子
            tiktoken_model_name=_tiktoken_model_name(),
            # gleaning（补抽）轮数：官方 2（每块最多 3 次 LLM 调用，建图慢的主因）；
            # 本项目默认 0（每块 1 次调用，最快档位）。对齐官方用 HGR_MAX_GLEANING=2
            entity_extract_max_gleaning=HGR_MAX_GLEANING,
            # 实体/关系描述的摘要上限（官方 500）：描述不足这个 token 数就不摘要。
            # 显式钉死，不吃 fork 隐式默认；对齐官方用 HGR_SUMMARY_MAX_TOKENS=500
            entity_summary_to_max_tokens=HGR_SUMMARY_MAX_TOKENS,
            # 小模型设定：few-shot 示例减到 1 个（官方默认 2 个，每个 ~750 token，
            # 示例占抽取 prompt 的 ~70%）。8B 模型下长 few-shot 只增加 token/延迟，
            # 收益有限。注意：这是**全方法统一下调样本规模**的设定，论文需注明。
            addon_params={"example_number": 1},
            # 抽取 LLM 并发：8B 模型显存吃紧，6 并发稳妥（16 会排队超时）
            embedding_batch_num=256,
            embedding_func_max_async=32,
            llm_model_max_async=HGR_MAX_ASYNC,
            # 摘要 prompt 的截断上限必须 < chat 窗口，否则每次摘要都 400（见文件头常量）
            llm_model_max_token_size=HGR_PROMPT_MAX_TOKENS,
        )

        set_default_phase("build")
        try:
            # fork 的 extract_entities 把所有 chunk 一次性并发抽取且单块无
            # try/except：一个 chunk 失败整个 ainsert 崩。响应级缓存
            # （llm_cache.jsonl）让重试只补抽失败部分，这里再按
            # kv_store_full_docs 的 content 集合找缺失块，重试只插缺失的
            # （fork 的 ainsert 按内容 md5 hash 判重，失败块内容没进索引，
            # 重插不会被挡——和 lightrag 的 filename 判重不同，无需换 id）。
            def _missing() -> list[int]:
                store = working_dir / "kv_store_full_docs.json"
                if not store.exists():
                    return list(range(len(docs)))
                try:
                    stored = {str(v.get("content", "")) for v in
                              json.loads(store.read_text(encoding="utf-8")).values()}
                except Exception:
                    return list(range(len(docs)))
                return [i for i, d in enumerate(docs)
                        if d not in stored and d.strip() not in stored]

            pending = list(range(len(docs)))
            last_err = None
            for attempt in range(1, HGR_BUILD_RETRIES + 1):
                if not pending:
                    break
                try:
                    self._dispatch(self._rag.ainsert([docs[i] for i in pending]))
                    last_err = None
                except Exception as e:
                    last_err = e
                    if attempt < HGR_BUILD_RETRIES:
                        wait = 2 ** attempt * 10
                        print(f"[hypergraphrag] 建索引失败（{type(last_err).__name__}）: {last_err}"
                              f"；{wait}s 后重试 {attempt + 1}/{HGR_BUILD_RETRIES}"
                              f"（响应缓存命中部分不重抽）", flush=True)
                        time.sleep(wait)
                pending = _missing()
            if last_err is not None and pending:
                # 重试耗尽仍缺：写 failed_docs.json 留痕、marker 照写——
                # 建图总能完成，不整体抛错（对应题可能缺证据，但不会白跑整批）
                (working_dir / "failed_docs.json").write_text(
                    json.dumps([{"i": i, "head": docs[i][:80]} for i in pending],
                               ensure_ascii=False, indent=2),
                    encoding="utf-8")
                print(f"[hypergraphrag] ⚠️ {len(pending)} 块重试 {HGR_BUILD_RETRIES} 轮仍失败，"
                      f"清单已写 failed_docs.json（对应题会受影响）", flush=True)
            marker.write_text(marker_sig, encoding="utf-8")
            # 口径对账（跑完立刻可见）：官方落盘的块数必须 == 我们喂入的 docs 数。
            # 相等 = 官方内部没再切（1 doc = 1 chunk）；不等 = 双重重叠，必须排查尺子。
            print(f"[hypergraphrag] {verify_chunk_parity(working_dir / 'kv_store_text_chunks.json', len(docs))}",
                  flush=True)
        finally:
            set_default_phase(None)
        return self

    # ── 检索 ──────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """hybrid 检索（他们自己的图检索）→ 答案生成统一走我们的 generate_answer。

        建图 prompt 用他们自己的；回答口径用 base.py 的 RAG_PROMPT（与其余基线一致）。
        """
        if self._rag is None:
            raise RuntimeError("请先 build() 建索引")
        self._patch_vdb_query_to_threadpool()  # 兜底：build 没走过的路径也确保 patch
        self._patch_print()
        from hypergraphrag import QueryParam
        from hypergraphrag.prompt import PROMPTS
        from methods.base import generate_answer
        fail_resp = PROMPTS.get("fail_response", "").strip()

        async def _q_context():
            # only_need_context：跳过答案生成 LLM 调用（operate.py 在检索后直接
            # return context）；但关键词抽取（hint_prompt）仍有一次 LLM 调用
            # max_token_for_*：官方 base.py:28-32 三个上下文预算**均为 4000**。
            # 显式钉死以便与官方逐项对账（不传 = fork 隐式默认，同样 4000）
            cparam = QueryParam(
                mode="hybrid", only_need_context=True,
                max_token_for_text_unit=HGR_QUERY_MAX_TOKENS,
                max_token_for_global_context=HGR_QUERY_MAX_TOKENS,
                max_token_for_local_context=HGR_QUERY_MAX_TOKENS,
            )
            resp = await self._rag.aquery(question, param=cparam)
            return resp if isinstance(resp, str) else str(resp)

        # 检索重试只针对超时类失败（LLM 排队超时等基础设施问题），最多 3 次
        # （首查 + 2 重试）。fork 自身返回 fail_response（关键词解析失败）或其他
        # 异常是方法本身的行为：不重试、不生成，按 unknown 记分——重试等于给
        # baseline 额外答题机会，破坏对比公平性
        context = ""
        for attempt in range(3):
            try:
                context = self._dispatch(_q_context())
                break
            except (VLLMTimeoutError, TimeoutError, asyncio.TimeoutError) as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                print(f"[hypergraphrag] 检索超时 3 次：{e}", flush=True)
                break
            except Exception as e:
                print(f"[hypergraphrag] 检索异常（不重试）：{e}", flush=True)
                break
        if not context.strip() or context.strip() == fail_resp:
            return {"answer": "unknown", "evidence": [], "evidence_locs": [],
                    "entities": [], "paths": []}

        # ★ 可记录的东西**就在 context 里**：`only_need_context=True` 让官方在检索后
        #   直接 `return context`（官方 operate.py:596-597），适配器本来就这么调的
        #   → 解析命中实体/超边是**零额外检索、零额外 token** 的。
        entities, paths = _parse_hgr_context(context)

        # 生成走我们自己的 prompt（短答案口径 + best guess + 关思考），
        # 在调用线程执行 → token 直接计入该题的 query 桶
        answer = generate_answer(question, [{"text": context}], dataset=getattr(self, "dataset", ""))
        return {"answer": answer,
                "evidence": [context[:2000]],
                "evidence_locs": [], "entities": entities, "paths": paths}

    def run(self, question, contexts):
        """per-question 路径不建议（每题独立建索引太慢），统一走语料级 build/query。"""
        raise NotImplementedError(
            "HyperGraphRAG 建议语料级：先 build(全部上下文) 一次，再逐题 query。"
        )

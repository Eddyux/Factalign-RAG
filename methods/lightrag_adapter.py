"""LightRAG 适配器（语料级：build() 建一次索引，query() 逐题检索）。

语料级口径：N 题上下文合成一个语料建索引，逐题 query（与 factalign/standardrag 同语料）。
按 lightrag-hku 1.5.x 实测（1.5.6；旧 1.x 的接口约定已废弃，见下方说明）。

依赖: pip install lightrag-hku

1.5.x 关键变化（与 1.x 旧版适配的差异）:
- llm_model_func 约定是 openai_complete 风格：第一个位置参数是 prompt
  （model 由 LightRAG 全局配置决定），必须 async，返回纯文本 str。
  官方默认 openai_complete_if_cache 是 (model, prompt, ...) 旧签名，
  1.5.x 用 openai_complete 包了一层桥接。本适配器自写 _vllm_complete
  走 common.llm.chat_messages，token 统一计入 build/query 分桶统计。
- EmbeddingFunc 的 func 必须 async，返回 np.ndarray（LightRAG 会 await
  并校验 .size 与 embedding_dim）。
- 存储锁绑定创建它的 event loop：官方明确警告 asyncio.run 建索引后 loop
  关闭、再跨线程查询会崩。本适配器起一个专用常驻 loop 线程，建索引和
  查询统一经 run_coroutine_threadsafe 派发到同一个 loop。
- 应答缓存由 LightRAG 的 use_llm_func_with_cache 层管理（hashing_kv 参数
  只是历史遗留，pop 掉即可）。

═══ 与官方 LightRAG 的差异清单（论文里要注明；**不改 baselines/ 源码，全是构造参数/运行时 patch**）═══
  1. **LLM/embedding 走本地 vLLM**：自写 `_vllm_complete`（openai_complete 同款签名）+ 注入
     `EmbeddingFunc(bge, 1024)`；token 统一计入 build/query 分桶（官方的 provider 不计数）。
  2. **块粒度 = 本项目语料单元**（默认**不切块**：`LIGHTRAG_CHUNK_TOKENS=0` → 归一成
     `CHUNK_NO_SPLIT=2000`、overlap 0，整段 passage 当一个块，每个块当一篇 doc 喂进去）。
     官方默认 `chunk_token_size=1200` / overlap 100 会超 bge 512 上限；旧版我们预切成
     400/40，现已统一为不切块。
  3. **`entity_extraction_use_json=True`**：官方对 tuple 格式解析器更稳，但我们的 provider
     强制 JSON 输出；官方文档也说 JSON 模式对小模型抽取质量/兼容性更好。
  4. **`entity_extract_max_gleaning=0`**：官方默认 1 → 每块最多 2 次抽取调用；设 0 = 每块 1 次
     （与 HyperGraphRAG 侧同档位，建图更快）。
  5. **并发**：`max_parallel_insert` / `llm_model_max_async` 显式上调（官方推荐 ≤10）；
     响应级/文档级缓存与缺文档兜底重试是适配器新增的工程件。
  6. **embedding 前 `fit_embed()`**：实体/关系描述过长时按 `EMBED_MAX_TOKENS`（2048）截断保住向量。
  7. **官方内部 token 尺子 = 本地 bge**（首选官方注入口 `tokenizer=`，老版本回退类级替换；
     **不改源码**）：
     - 官方 chunker 全吃 `self.tokenizer`（`chunker/token_size.py:174/250` 等），默认
       `TiktokenTokenizer`(gpt-4o-mini/o200k_base)。我们喂的块若被它重数超过
       `chunk_token_size` 就会被**再切一刀**（步长 = size−overlap）→ 索引块数 > 喂入 docs 数、那部分检索单元
       变成"半块"。换成 bge 后两边同一把尺子 → `1 doc = 1 chunk`。
     - 做法：`lightrag.utils.Tokenizer(model_name=EMBED_MODEL_SERVED, tokenizer=<bge 接口>)`
       传给构造器（官方 `__post_init__` 只在 `self.tokenizer is None` 时才造 tiktoken 版）。
       内层只需 `encode`/`decode`（`TokenizerInterface`，utils.py:2855-2877），基类自带的
       `split_by_token_limit`/`truncate_by_token_limit` 只用 `encode`，不必自己实现；
       另外给内层加了 `__deepcopy__`（构造会 `asdict` 深拷贝 global_config）。
       不支持该参数的版本自动回退到类级替换（encode/decode + safe split/truncate 一并换，
       避免 bge 的 token id 被喂进 tiktoken 的 `decode_with_offsets` 快路径）。
     - 设 `LIGHTRAG_TOKENIZER=tiktoken` 回到官方原版。
     - ⚠️ 换尺子会连带改 **query 期**的上下文预算口径（`truncate_list_by_token_size` 等
       也按 bge 计数），但 `max_entity/relation/total_tokens` 我们没跟着下调
       （HyperGraphRAG 侧把 4000→3000 做了对齐）→ 论文里按实际报告上下文量。
     - ⚠️ 官方 chunker 产出的块文本是 `bge.decode(bge.encode(doc))`（**decode 往返**，
       不是与预切块逐字相同）：抽取/检索用的是这份文本，块向量则由我们另按原文计算。
  8. **引擎版本**：适配器按 pip 安装的 lightrag-hku 版本走（本机无法核对 pip 版；
     `baselines/LightRAG-main` 是 1.5.8）。建完索引建议用 `tools/audit_chunk_parity.py`
     对账 `kv_store_text_chunks.json` 的块数 vs 喂入 docs 数。
"""

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import threading
from pathlib import Path

from methods.base import BaseMethod, generate_answer
from common.llm import (chat_retry, embed_batch, bucket_ctx, fit_embed, gate_for,
                        get_phase_bucket, set_default_phase, split_by_tokens)
from config import (CHAT_MODEL, EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP,
                    EMBED_DIM, CHUNK_TOKENIZER as _CHUNK_TOKENIZER_DEFAULT,
                    EMBED_MAX_TOKENS as _EMBED_MAX_TOKENS_DEFAULT)

# ── 尺子 / 向量（换 embedding 模型时要一起调的旋钮）──
# LIGHTRAG_EMBED_DIM：官方 `EmbeddingFunc(embedding_dim=)`，必须 = 服务端真实维度
#   （默认跟随全局 `EMBED_DIM`）。
# LIGHTRAG_EMBED_MAX_TOKENS：官方 `EmbeddingFunc(max_token_size=)` 的声明值
#   （默认跟随全局 `EMBED_MAX_TOKENS`）。真实截断由全局 `fit_embed()` 执行。
# LIGHTRAG_TOKENIZER：官方内部 tiktoken 尺子是否换掉。默认跟随全局 `CHUNK_TOKENIZER`
#   （`bge`=换成本地 embedder tokenizer，与预切块同尺子；`tiktoken`=官方原版口径）。
LIGHTRAG_EMBED_DIM = int(os.environ.get("LIGHTRAG_EMBED_DIM", str(EMBED_DIM)))
LIGHTRAG_EMBED_MAX_TOKENS = int(os.environ.get("LIGHTRAG_EMBED_MAX_TOKENS",
                                               str(_EMBED_MAX_TOKENS_DEFAULT)))
# 分块（官方默认 1200/100）；本项目统一改用 `config.py` 的
# `EXTRACT_CHUNK_TOKENS` / `EXTRACT_CHUNK_OVERLAP`（默认**不切块**：0 → 2000/0）。
# ★ 默认**派生自全局 config.py** —— 改 config.py 一处，所有方法 + LightRAG 内部的官方
#   切块器一起变。⚠️ 必须与传给官方的 `chunk_token_size`/`chunk_overlap_token_size` 逐字一致，
#   否则官方会二次切块（双重重叠）。
LIGHTRAG_CHUNK_TOKENS = int(os.environ.get("LIGHTRAG_CHUNK_TOKENS", str(EXTRACT_CHUNK_TOKENS)))
LIGHTRAG_CHUNK_OVERLAP = int(os.environ.get("LIGHTRAG_CHUNK_OVERLAP", str(EXTRACT_CHUNK_OVERLAP)))
# gleaning（补抽）轮数：官方默认 1（每块最多 2 次 LLM 调用）；本项目默认 0（每块 1 次）
LIGHTRAG_MAX_GLEANING = int(os.environ.get("LIGHTRAG_MAX_GLEANING", "0"))
_LIGHTRAG_TOKENIZER_DEFAULT = _CHUNK_TOKENIZER_DEFAULT


def _lr_tokenizer_mode() -> str:
    """LightRAG 内部尺子模式：`bge`（换成本地 embedder tokenizer）或 `tiktoken`（官方原版）。"""
    return os.environ.get("LIGHTRAG_TOKENIZER", _LIGHTRAG_TOKENIZER_DEFAULT).lower()



async def _vllm_complete(prompt, system_prompt=None, history_messages=None,
                         keyword_extraction=False, entity_extraction=False, **kwargs):
    """自定义 llm_model_func（openai_complete 同款签名）：走 common.llm.chat_messages。

    LightRAG 1.5.x 会把本函数包上 hashing_kv 后经优先级队列（asyncio task）
    调用；返回 str。token 经 bucket_ctx 归到发起查询/建索引的线程。
    """
    kwargs.pop("hashing_kv", None)  # 应答缓存由 use_llm_func_with_cache 层管理

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for m in history_messages or []:
        if isinstance(m, dict) and m.get("content") is not None:
            messages.append({"role": m.get("role", "user"), "content": m["content"]})
    messages.append({"role": "user", "content": prompt})

    rf = kwargs.get("response_format")
    # 注意：不要按 prompt 里是否含 "json" 强制 json_object！
    # LightRAG 的抽取解析器默认吃 tuple 格式（<|#|> 分隔 4 字段），
    # 强制 JSON 输出会被它按 tuple 切碎（"found 39/4 fields" 就是这病）。
    # 只有当 LightRAG 自己要求（keyword_extraction / 显式 response_format）才开。
    json_mode = bool(keyword_extraction or entity_extraction
                     or (rf and "json" in str(rf).lower()))
    # 重试 + 自适应并发统一走 common.llm.chat_retry（与 hypergraphrag 同款）：
    # 阻塞 HTTP 在线程池里（不卡 LightRAG 的常驻 loop）；并发只由 vLLM 指标决定
    # （waiting/KV 减半、空闲加性增），**重试不改并发**；失败只重试 1 次、
    # 输出上限兜底 4096 且重试时减半；单次调用总预算 600s。
    content, _pt, _ct = await chat_retry(
        messages, gate=gate_for(LLM_MAX_ASYNC, "lightrag"),
        json_mode=json_mode, timeout=300)
    if content is None:
        raise RuntimeError("vLLM 调用失败（重试用尽）")
    return content


# 建索引并发与失败兜底（--workers 只管逐题查询，管不到建索引）
MAX_PARALLEL_INSERT = 32   # 文档级并行（LightRAG 官方推荐 ≤10，32 偏激进；
                           # 若日志出现锁竞争/内存升高/写入冲突，退回 16）
LLM_MAX_ASYNC = 64         # 抽取 LLM 并发上限，同时是自适应闸的封顶：vLLM waiting/KV 高
                           # 会自动减半、引擎空闲再加性增（见 common.llm.AdaptiveLimiter）。
                           # Qwen2.5 的 KV 占用约为 Llama-3.1 的一半，所以上限给得比 HGR 高
RETRY_ROUNDS = 3          # 缺文档兜底重试轮数（每轮并发减半）


def _drop_doc_status(miss_fps: set[str], working_dir: Path) -> int:
    """删 doc_status 里缺失块（含 r{n}_ 重试残留）的旧记录。

    为什么必须删：doc_status 记录入队时就写 content_hash，失败只改 status
    不删 hash——重插时 LightRAG 的 content-hash 判重（get_existing_doc_by_
    content_hash）会命中旧记录直接剔除，换 id 只绕过 filename 判重、
    绕不过 content_hash 判重（1.5.6 和 1.5.7 都这样）。删掉记录后重插
    不再被挡。返回删除条数。
    """
    p = working_dir / "kv_store_doc_status.json"
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return 0

    def _match(fp):
        if fp in miss_fps:
            return True
        for n in range(1, RETRY_ROUNDS + 1):
            pre = f"r{n}_"
            if fp.startswith(pre) and fp[len(pre):] in miss_fps:
                return True
        return False

    to_del = [k for k, v in data.items() if _match(str(v.get("file_path", "")))]
    for k in to_del:
        del data[k]
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(to_del)


def _missing_docs(docs: list[str], working_dir: Path) -> list[int]:
    """按 kv_store_full_docs 里实际存储的正文集合找没入库的文档下标。

    LightRAG 对抽取失败/超时的文档只打 ERROR 日志然后静默跳过，索引里永久缺它。
    直接比对「存储的正文集合」而非哈希：content-hash 去重会让相同正文的后续
    文档不进索引（其正文已在另一条记录下，检索不受影响），这种情况不算缺失；
    哈希口径也可能因版本/strip 而不同，正文比对最稳。
    """
    store = Path(working_dir) / "kv_store_full_docs.json"
    if not store.exists():
        return list(range(len(docs)))
    try:
        stored_contents = {str(v.get("content", "")) for v in
                           json.loads(store.read_text(encoding="utf-8")).values()}
    except Exception:
        return list(range(len(docs)))
    return [i for i, d in enumerate(docs)
            if d not in stored_contents and d.strip() not in stored_contents]


class LightRAG(BaseMethod):
    name = "lightrag"

    def __init__(self):
        self._rag = None
        self._loop = None
        self._loop_thread = None
        # 默认统一走本项目 generate_answer（prompt 与自带方法同口径）；
        # True 时用 LightRAG 自己的生成器（端到端原生态对比用，run.py --native-generator）
        self.native_generator = False

    # ── 专用常驻 event loop（1.5.x 存储锁绑定 loop，不能 asyncio.run 用完就关）──

    def _ensure_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()

            def _run():
                asyncio.set_event_loop(self._loop)
                # 关键：asyncio 默认 executor 只有 min(32, cpu+4) 个 worker，
                # 而本适配器的阻塞 HTTP 走 asyncio.to_thread → 实际并发请求被卡在
                # 12~32 个（vLLM 侧 KV cache 只用 ~10%，GPU 在等活干）。
                # 显式放大到 128：真正的并发上限交回 llm_model_max_async / vLLM KV。
                from concurrent.futures import ThreadPoolExecutor
                self._loop.set_default_executor(
                    ThreadPoolExecutor(max_workers=128, thread_name_prefix="lr-http"))
                self._loop.run_forever()

            self._loop_thread = threading.Thread(target=_run, daemon=True,
                                                 name="lightrag-loop")
            self._loop_thread.start()
        return self._loop

    def _dispatch(self, coro):
        """把协程派发到常驻 loop 并等结果。

        统计归调用线程：先在调用线程取统计桶 set 进 bucket_ctx，
        contextvars 随 run_coroutine_threadsafe 复制到 loop 线程，
        loop 线程里的 chat_messages 会记回调用线程的桶——单题 token 差值才能算对。
        """
        loop = self._ensure_loop()
        token = bucket_ctx.set(get_phase_bucket())
        try:
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        finally:
            bucket_ctx.reset(token)

    # ── 建索引 ────────────────────────────────────────────────

    def _make_rag(self, working_dir: Path, llm_async: int, parallel: int):
        """构造 LightRAG 实例。兜底重试轮用更小并发，避免再打满 KV 排队超时。"""
        try:
            from lightrag import LightRAG as _LR
            from lightrag.utils import EmbeddingFunc
        except ImportError as e:
            raise RuntimeError("需先 `pip install lightrag-hku`") from e

        async def embed_func(texts):
            """1.5.x 要求 async + 返回 np.ndarray（会 await 并校验维度）。
            encode 放线程池，避免阻塞常驻 loop。"""
            if isinstance(texts, str):
                texts = [texts]
            # 实体/关系描述可能超 bge 512 上限（vLLM 400 拒绝 → 该实体向量被丢弃），
            # 嵌入前截断到 480，保住向量
            texts = [fit_embed(t) for t in texts]
            return await asyncio.to_thread(embed_batch, texts)

        kw = dict(
            working_dir=str(working_dir),
            llm_model_func=_vllm_complete,
            llm_model_name=CHAT_MODEL,
            embedding_func=EmbeddingFunc(embedding_dim=LIGHTRAG_EMBED_DIM,
                                         max_token_size=LIGHTRAG_EMBED_MAX_TOKENS,
                                         func=embed_func),
            # 与自带方法同口径：默认**不切块**（0 → CHUNK_NO_SPLIT=2000、overlap 0，整段一块）
            # 对齐官方：LIGHTRAG_CHUNK_TOKENS=1200 LIGHTRAG_CHUNK_OVERLAP=100
            chunk_token_size=LIGHTRAG_CHUNK_TOKENS,
            chunk_overlap_token_size=LIGHTRAG_CHUNK_OVERLAP,
            # JSON 抽取模式：官方文档明确说"显著提升小模型的抽取质量与兼容性"。
            # 开启后 LightRAG 自己传 response_format=json_object（我们的 provider
            # 强制 JSON 输出），解析器走 JSON 路径——不再依赖 tuple 格式（
            # Qwen2.5 描述里带逗号就会把 tuple 解析器炸掉，即 "found 39/4 fields"）
            entity_extraction_use_json=True,
            # gleaning（补抽）默认 1 → 每块最多 2 次 LLM 调用；设 0 = 只做初始抽取，
            # 与 HyperGraphRAG 侧保持同一档位（建图更快，抽取完整度影响很小）
            # 官方默认 1（= 每块最多 2 次调用）→ 对齐官方用 LIGHTRAG_MAX_GLEANING=1
            entity_extract_max_gleaning=LIGHTRAG_MAX_GLEANING,
            # 建索引并发（--workers 只管逐题查询，管不到建索引）
            # 新版 pipeline 的文档级并行开关是 max_parallel_insert（默认 3，推荐 ≤10），
            # 这才是 Running 2~3 的元凶；llm_model_max_async 在 pipeline 里作用有限
            max_parallel_insert=parallel,   # 文档级并行（首次 MAX_PARALLEL_INSERT，重试轮减半）
            llm_model_max_async=llm_async,  # 抽取 LLM 并发（首次 LLM_MAX_ASYNC，重试轮减半）
            embedding_func_max_async=32,    # embedding 并发
            embedding_batch_num=256,        # 每批 embed 向量数
        )

        # ★ 统一尺子（首选）：官方注入口 `tokenizer=`——官方 `__post_init__` 只在
        #   `self.tokenizer is None` 时才造 TiktokenTokenizer，传了就用我们的 bge 实现。
        #   一次注入，chunker / truncate / 第二道 512 闸门全部跟着换，且不留 patch 副作用。
        bge_tokenizer = self._make_bge_tokenizer()
        if bge_tokenizer is not None:
            try:
                return _LR(tokenizer=bge_tokenizer, **kw)
            except TypeError as e:          # 老版本没有 tokenizer 参数 → 走类级兜底
                print(f"[lightrag] 该版本不支持 tokenizer= 注入（{e}）→ 回退类级替换", flush=True)
                self._patch_lightrag_tokenizer()
        return _LR(**kw)

    # ── 统一 token 尺子：把 LightRAG 内部的 tiktoken 换成本地 bge ──
    # 主路径 = **官方注入口 `tokenizer=`**（`lightrag.py:818` + `1737-1741`：只有
    # `self.tokenizer is None` 才造 TiktokenTokenizer）；兜底 = 类级替换（老版本没有该参数时）。
    _TOKENIZER_PATCHED = False

    @staticmethod
    def _make_bge_tokenizer():
        """构造官方 `Tokenizer` 包装的 bge 实现（`_make_rag` 用）。拿不到就返回 None。

        为什么用官方注入而不是 patch 类方法：
        - 一次注入，`self.tokenizer` 这一整片调用点全跟着换（chunker 的 encode/decode、
          `truncate_list_by_token_size`、`enforce_chunk_token_limit_before_embedding`
          的第二道 512 闸门、`ainsert_custom_chunks/`custom_kg 的 tokens 计数）；
        - 不会留下"bge 的 id 被喂进 `tiktoken.decode_with_offsets`"这种**靠运气**的路径
          （`TiktokenTokenizer.split_by_token_limit` 内部走 tiktoken 快路径，类级 patch
          只换 encode/decode 时它仍会拿 bge id 去 tiktoken 解，结果不等才回退 super()）。
        内层实现无状态 → 天然线程安全；`Tokenizer` 自带通用 `split_by_token_limit` /
        `truncate_by_token_limit`（只用 encode），不需要我们再实现。
        """
        if _lr_tokenizer_mode() != "bge":
            return None
        try:
            from lightrag.utils import Tokenizer as _Tokenizer
            from common.llm import _get_tokenizer, EMBED_MODEL_SERVED
        except Exception:
            return None
        try:
            tok = _get_tokenizer()
        except Exception as e:
            print(f"[lightrag] ⚠ bge tokenizer 加载失败（{type(e).__name__}: {e}）→ "
                  f"回退官方 tiktoken 尺子，存在块被再切的风险", flush=True)
            return None

        class _BgeInterface:
            """`TokenizerInterface`（utils.py:2855-2877 只要 encode/decode，要求线程安全）。

            `__deepcopy__`/`__copy__` 返回 self：LightRAG 构造时会 `asdict` 深拷贝
            global_config（utils.py:3503-3517），不拦一下会去深拷贝整个 HF tokenizer。
            实现无状态（每次现取缓存的 tokenizer）→ 返回自身是安全的。
            """

            def encode(self, content, **kw):
                # 'disallowed_special' 是 tiktoken 专有入参：bge 不需要（也不接受）→ 丢掉
                kw.pop("disallowed_special", None)
                return _get_tokenizer().encode(content, add_special_tokens=False)

            def decode(self, tokens, **kw):
                kw.pop("disallowed_special", None)
                return _get_tokenizer().decode(list(tokens), skip_special_tokens=False)

            def __deepcopy__(self, memo):
                return self

            def __copy__(self):
                return self

        return _Tokenizer(model_name=str(EMBED_MODEL_SERVED), tokenizer=_BgeInterface())

    @staticmethod
    def _patch_lightrag_tokenizer() -> bool:
        """兜底：类级把 `TiktokenTokenizer.encode/decode`（以及两个 safe split）换掉。

        只在**老版本 LightRAG 不支持 `tokenizer=` 构造参数**时才会用到。除 encode/decode
        外必须连带替换 `split_by_token_limit` / `truncate_by_token_limit` 为基类实现，
        否则它们内部的 tiktoken `decode_with_offsets` 会拿到 bge 的 token id（词表更大
        不报错，但结果错、白跑一遍）。设 `LIGHTRAG_TOKENIZER=tiktoken` 回到官方原版。
        """
        if _lr_tokenizer_mode() != "bge":
            return False
        if LightRAG._TOKENIZER_PATCHED:
            return True
        try:
            from lightrag.utils import TiktokenTokenizer as _TT, Tokenizer as _Base
            from common.llm import _get_tokenizer
        except Exception:
            return False
        try:
            tok = _get_tokenizer()
        except Exception:
            return False

        def _encode(self, content, **kw):
            kw.pop("disallowed_special", None)
            return tok.encode(content, add_special_tokens=False)

        def _decode(self, tokens, **kw):
            kw.pop("disallowed_special", None)
            return tok.decode(list(tokens), skip_special_tokens=False)

        _TT.encode = _encode
        _TT.decode = _decode
        # safe split/truncate 必须一起换成基类实现：`TiktokenTokenizer` 那两个重写走的是
        # tiktoken 的 `decode_with_offsets` 快路径，会把 bge 的 token id 喂给 tiktoken
        # （词表更大所以不报错，结果错+白跑）。老版本基类没有这些方法时跳过。
        for meth in ("split_by_token_limit", "truncate_by_token_limit"):
            base_meth = getattr(_Base, meth, None)
            if base_meth is not None and hasattr(_TT, meth):
                setattr(_TT, meth, base_meth)
        LightRAG._TOKENIZER_PATCHED = True
        print("[lightrag] token 尺子=tiktoken→bge（兜底：类级替换 encode/decode + "
              "safe split/truncate）→ 与预切块同尺子，官方内部 1 doc=1 chunk",
              flush=True)
        return True

    def load(self, working_dir: str = "data/lightrag_corpus"):
        """只加载已有索引，直接进查询。不做任何建图/断点/重建判断。

        索引不存在直接报错——建索引的事归 run.py 管，本方法永不触发抽取。
        """
        working_dir = Path(working_dir)
        if not (working_dir / "kv_store_full_docs.json").exists():
            raise RuntimeError(
                f"[lightrag] 索引不存在：{working_dir}\n"
                f"请先跑：python run.py --method lightrag --dataset ... 建索引")
        # 查询期静音 LightRAG 的 INFO 刷屏（build 里配进度条 handler，
        # load 只路由 WARNING+ 到终端，INFO 全静默）
        _lg = logging.getLogger("lightrag")
        _lg.propagate = False
        if not any(isinstance(h, logging.StreamHandler) for h in _lg.handlers):
            _ch = logging.StreamHandler()
            _ch.setLevel(logging.WARNING)
            _ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            _lg.addHandler(_ch)

        self._rag = self._make_rag(working_dir, LLM_MAX_ASYNC, MAX_PARALLEL_INSERT)
        self._dispatch(self._rag.initialize_storages())
        return self

    def build(self, corpus_contexts: list[dict], working_dir: str = "data/lightrag_corpus"):
        """把 N 题上下文合成一个语料，建 LightRAG 索引。

        insert 过程中 LightRAG 自己做 LLM 实体/关系抽取；索引存在 working_dir，
        目录已存在时会加载旧索引而不是重建（断点续）。
        """
        # 语料预切块：用 bge tokenizer 按本项目语料单元切（默认**不切块**：
        # LIGHTRAG_CHUNK_TOKENS=0 → 归一成 CHUNK_NO_SPLIT=2000、overlap 0；整段 passage 一条"文档"）。
        # 旧版为 400/40。
        # 块 ≤ `LIGHTRAG_CHUNK_TOKENS`（同一把尺子）→ LightRAG 内部切块器整条通过
        # （1 块 = 1 文档），检索单元与自带方法逐块一致（彻底消除分词器差异）。
        # ★ 尺寸与尺子都走 LightRAG **自己的**方法级旋钮，不走全局 `EXTRACT_CHUNK_*` /
        # `CHUNK_TOKENIZER`：块尺寸必须与官方内部 `chunk_token_size` + 计数尺子严格一致，
        # 否则我们切的块会被官方按它的阈值再切一刀 → 双重重叠。同旋钮做两件事 = 不可能错配。
        docs = []
        ids = []
        file_paths = []
        for c in corpus_contexts:
            title = (c.get('title') or '').strip()
            qid = str(c.get('qid', ''))
            full = f"{title}\n{c.get('text','')}".strip()
            for chunk_text, _cs, _ce in split_by_tokens(full, LIGHTRAG_CHUNK_TOKENS,
                                                        LIGHTRAG_CHUNK_OVERLAP,
                                                        kind=_lr_tokenizer_mode()):
                if chunk_text.strip():
                    docs.append(chunk_text)
                    # 唯一 id（题目 id + 标题 + 序号）：避免内容 MD5 撞 hash
                    doc_id = f"{qid}_{title}_{len(docs)}"
                    ids.append(doc_id)
                    # file_path 必须唯一：LightRAG 1.5.x 按 filename 严格去重，
                    # 同标题（hotpotqa 共享段落/同文多块）会被判重并直接丢弃
                    # （dup 记录 status=FAILED、0 chunk，内容不进索引）
                    file_paths.append(doc_id)

        # 完成标记：LightRAG 插入即写盘，中断留下的半截索引会被当成完整索引加载。
        # 没有标记 → 上次 build 没跑完 → 清掉重建；切块参数变了 → 索引口径变了 → 重建。
        working_dir = Path(working_dir)
        marker = working_dir / "build_complete.marker"
        marker_sig = (f"chunk={_lr_tokenizer_mode()}:"
                      f"{LIGHTRAG_CHUNK_TOKENS}/{LIGHTRAG_CHUNK_OVERLAP}")
        # 图文件完整但缺 marker（上次 ainsert 末尾异常/中断）→ 补写 marker，不重建
        _graph_files = ["kv_store_full_docs.json", "kv_store_full_entities.json",
                        "kv_store_full_relations.json", "graph_chunk_entity_relation.graphml"]
        if working_dir.exists() and not marker.exists() and \
                all((working_dir / f).exists() for f in _graph_files):
            # 半截索引的 4 个文件也可能都在（插入即写盘）——补写前按文档数校验，
            # 对不上会掉进下方重建分支，不把残缺索引当完整
            try:
                n_stored = len(json.loads(
                    (working_dir / "kv_store_full_docs.json").read_text(encoding="utf-8")))
            except Exception:
                n_stored = -1
            if n_stored == len(set(docs)):
                marker.write_text(marker_sig, encoding="utf-8")
                print(f"[lightrag] 检测到完整图但缺 marker，补写 marker（跳过重建）")
        if working_dir.exists():
            old_sig = marker.read_text(encoding="utf-8") if marker.exists() else ""
            if old_sig != marker_sig:
                print(f"[lightrag] 索引状态变化（{old_sig or '未完成'} → {marker_sig}），清除 {working_dir} 重建")
                shutil.rmtree(working_dir, ignore_errors=True)
        working_dir.mkdir(parents=True, exist_ok=True)

        # 日志策略：完整 INFO 写文件；终端把「抽取/合并/落盘」三阶段进度渲染成单行进度条
        # （\r 覆盖，不逐行刷屏）；报错（WARNING/ERROR）照常打印。
        class _LightRAGProgressHandler(logging.Handler):
            def __init__(self, total_chunks: int = 0):
                super().__init__()
                # total_chunks：build() 的总块数（len(docs)）。LightRAG 的
                # Extracting/Merging 日志是「批内文件」进度（一批 ~500 个，
                # total_files = 批大小），批进度换算成全局进度，剩余按总量算
                self._total = total_chunks
                self._done_before = {"Extracting stage": 0, "Merging stage": 0,
                                     "processing file": 0}
                self._last_n = {"Extracting stage": None, "Merging stage": None,
                                "processing file": None}
                self._last_m = {"Extracting stage": 0, "Merging stage": 0,
                                "processing file": 0}
                self._prog = {"Extracting stage": None, "Merging stage": None,
                              "processing file": None}
                self._labels = [("Extracting stage", "抽取"), ("Merging stage", "合并"),
                                ("processing file", "落盘")]
                self._on_bar = False
                self._done = False

            def finish(self):
                """build 结束：进度条换行收尾，之后只路由报错、不再渲染进度
                （query 阶段的 INFO 日志会反复触发 emit，不拦会拿旧值刷屏）。"""
                if self._on_bar:
                    sys.stderr.write("\n")
                    sys.stderr.flush()
                    self._on_bar = False
                self._done = True

            def emit(self, record):
                msg = record.getMessage()
                if record.levelno >= logging.WARNING:
                    # 报错：先换行（别覆盖进度条），再打
                    if self._on_bar:
                        sys.stderr.write("\n")
                        self._on_bar = False
                    sys.stderr.write(f"{record.levelname}: {msg}\n")
                    sys.stderr.flush()
                    return
                if self._done:
                    return  # build 已结束，INFO 不再渲染进度条
                for key, _label in self._labels:
                    if key in msg:
                        m = re.search(rf"{key}\s+(\d+)/(\d+)", msg)
                        if m:
                            n, m_total = int(m.group(1)), int(m.group(2))
                            # 批内进度 → 全局：新一批从 1 重新数（n 回退）时
                            # 把上一批的完成数累加；批总量 < 总块数才算批内
                            if m_total < self._total:
                                last = self._last_n[key]
                                if last is not None and n < last:
                                    self._done_before[key] += self._last_m[key]
                                self._last_n[key] = n
                                self._last_m[key] = m_total
                                self._prog[key] = (self._done_before[key] + n,
                                                   self._total)
                            else:
                                self._prog[key] = (n, m_total)
                        break
                parts = []
                for key, label in self._labels:
                    v = self._prog[key]
                    if v:
                        n, total = v
                        parts.append(f"{label} {n}/{total}(剩{total - n})")
                if parts:
                    sys.stderr.write("\r" + " | ".join(parts))
                    sys.stderr.flush()
                    self._on_bar = True

        _lg = logging.getLogger("lightrag")
        for _h in list(_lg.handlers):
            _lg.removeHandler(_h)
        _fmt = logging.Formatter("%(levelname)s: %(message)s")

        # 文件：全量日志
        _fh = logging.FileHandler(str(working_dir / "lightrag_build.log"), encoding="utf-8")
        _fh.setLevel(logging.INFO)
        _fh.setFormatter(_fmt)
        _lg.addHandler(_fh)

        # 终端：单行进度条 + 报错（总块数来自上面的 docs，剩余按总量算）
        _pb = _LightRAGProgressHandler(len(docs))
        _lg.addHandler(_pb)

        _lg.setLevel(logging.INFO)
        _lg.propagate = False

        self._rag = self._make_rag(working_dir, LLM_MAX_ASYNC, MAX_PARALLEL_INSERT)

        # 建索引期间的 LLM/embedding 调用发生在 loop 线程上，没有 phase 上下文，
        # 强制归到 build 桶
        set_default_phase("build")
        try:
            # 真正要建索引才打印总量（索引已存在时 ainsert 秒过，不用误导）
            if not marker.exists():
                print(f"[lightrag] 开始建索引：共 {len(docs)} 块待抽取", flush=True)
            try:
                self._dispatch(self._rag.initialize_storages())
                self._dispatch(self._rag.ainsert(docs, ids=ids, file_paths=file_paths))
            except Exception as e:
                print(f"[lightrag] 首次建索引异常：{e}，进入兜底重试", flush=True)

            # ── 抽取失败兜底 ──
            # LightRAG 对抽取超时/失败的文档只打 ERROR 日志然后静默跳过，
            # 索引里会永久缺它。事后按 kv_store_full_docs 的 content_hash
            # （= md5 正文）找出缺的文档，用减半并发的新实例重插；
            # 轮数耗尽仍缺的写 failed_docs.json 留痕（marker 照写，缺题知情）。
            # 并发下限 8（不是 2）：重试轮只为缓解 vLLM 排队，降到 8 已足够；
            # 下限 2 会让建图后半段锁死在 2 并发慢慢磨（16→8→4→2）。
            # 抽取内容/质量不变，只影响补抽速度。
            llm_async, parallel = LLM_MAX_ASYNC, MAX_PARALLEL_INSERT
            for round_no in range(1, RETRY_ROUNDS + 1):
                miss = _missing_docs(docs, working_dir)
                if not miss:
                    break
                llm_async = max(8, llm_async // 2)
                parallel = max(8, parallel // 2)
                print(f"[lightrag] {len(miss)} 个文档未入库，第 {round_no}/{RETRY_ROUNDS} 轮重试"
                      f"（llm_async={llm_async} parallel={parallel}）", flush=True)
                retry_rag = self._make_rag(working_dir, llm_async, parallel)
                try:
                    self._dispatch(retry_rag.initialize_storages())
                    # 重插前必须先删 doc_status 旧记录（含上一轮 r{n}_ 残留）：
                    # 记录的 content_hash 会让重插被 content-hash 判重剔除，
                    # 换 id 只绕过 filename 判重、绕不过 content-hash 判重
                    # （1.5.6/1.5.7 都这样）——不删则「重试永远失败」。
                    n_del = _drop_doc_status({file_paths[i] for i in miss},
                                             working_dir)
                    if n_del:
                        print(f"[lightrag] 已清 doc_status 旧记录 {n_del} 条", flush=True)
                    # 重插再换 ids/file_paths（r{n}_ 前缀）双保险绕 filename 判重
                    retry_ids = [f"r{round_no}_{ids[i]}" for i in miss]
                    retry_fps = [f"r{round_no}_{file_paths[i]}" for i in miss]
                    self._dispatch(retry_rag.ainsert(
                        [docs[i] for i in miss],
                        ids=retry_ids,
                        file_paths=retry_fps))
                    # 查询要用最新实例：旧实例的内存态不含本轮重插的文档
                    self._rag = retry_rag
                except Exception as e:
                    print(f"[lightrag] 第 {round_no} 轮重试异常：{e}", flush=True)
            else:
                miss = _missing_docs(docs, working_dir)
            if miss:
                (working_dir / "failed_docs.json").write_text(
                    json.dumps([{"id": ids[i], "file_path": file_paths[i],
                                 "head": docs[i][:80]} for i in miss],
                               ensure_ascii=False, indent=2),
                    encoding="utf-8")
                print(f"[lightrag] ⚠️ {len(miss)} 个文档重试 {RETRY_ROUNDS} 轮仍失败，"
                      f"清单已写 failed_docs.json（对应题会受影响）", flush=True)
            marker.write_text(marker_sig, encoding="utf-8")  # 兜底结束才打标记
        finally:
            _pb.finish()  # 进度条收尾：query 阶段的 INFO 日志不再拿旧值重绘
            set_default_phase(None)
        return self

    # ── 检索 ──────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """在已建索引上检索 + 生成。

        native_generator=False（默认）：检索用 LightRAG 原生（hybrid），答案统一走
        generate_answer（本项目 RAG_PROMPT），与自带方法/其它 baseline 完全同 prompt——
        检索差异不被生成 prompt 差异污染。
        native_generator=True：LightRAG 自己的生成（run.py --native-generator）。
        """
        if self._rag is None:
            raise RuntimeError("请先 build() 建索引")
        try:
            from lightrag import QueryParam
        except ImportError as e:
            raise RuntimeError("需先 `pip install lightrag-hku`") from e

        if self.native_generator:
            # ── 原生态：LightRAG 自己的生成（口径化指令）──
            param = QueryParam(
                mode="hybrid",
                enable_rerank=False,
                response_type=("Short Answer: output ONLY the exact name, date, number or "
                               "entity, using the fewest words the answer requires; no "
                               "sentence, no explanation, no citation."),
                user_prompt=("Give the SHORTEST COMPLETE answer: the exact name, date, number, "
                             "or entity. No markdown, no references, no explanation. "
                             "Even if the evidence is incomplete, make your BEST GUESS — "
                             "never answer unknown or say you don't know."),
            )

            async def _q():
                resp = await self._rag.aquery_llm(question, param=param)
                return resp

            resp = self._dispatch(_q())
            if not isinstance(resp, dict):
                return {"answer": str(resp).strip(), "evidence": [],
                        "evidence_locs": [], "entities": []}
            answer = (resp.get("llm_response") or {}).get("content") or ""
            data = resp.get("data") or {}
            evidence = []
            for c in data.get("chunks") or []:
                if isinstance(c, dict):
                    t = c.get("content") or c.get("text")
                    if t:
                        evidence.append(str(t)[:500])
            ents = [str(e.get("entity_name")) for e in (data.get("entities") or [])
                    if isinstance(e, dict) and e.get("entity_name")]
            return {"answer": str(answer).strip(), "evidence": evidence[:20],
                    "evidence_locs": [], "entities": ents}

        # ── 统一生成：只取检索 chunks，答案由 generate_answer 出 ──
        # enable_rerank=False：本基线没配 rerank 模型，显式关掉消掉每题的 WARNING；
        # 与官方「开关默认开但无模型→实际不生效」的检索行为等价
        # only_need_context=True：跳过 LightRAG 自己的生成（本来也弃用），
        # 每题省一次 LLM 调用；raw_data（chunks）照常返回
        param = QueryParam(mode="hybrid", enable_rerank=False, only_need_context=True)

        async def _q():
            # aquery_llm 一次检索同时返回结构化数据（chunks/entities）+ LLM 回答；
            # 这里只取检索结果（chunks），LLM 回答弃用
            resp = await self._rag.aquery_llm(question, param=param)
            return resp

        resp = self._dispatch(_q())
        if not isinstance(resp, dict):
            return {"answer": generate_answer(question, [], dataset=getattr(self, "dataset", "")), "evidence": [],
                    "evidence_locs": [], "entities": []}
        data = resp.get("data") or {}
        evidence = []
        for c in data.get("chunks") or []:
            if isinstance(c, dict):
                t = c.get("content") or c.get("text")
                if t:
                    evidence.append({"text": str(t)[:500]})
        ents = [str(e.get("entity_name")) for e in (data.get("entities") or [])
                if isinstance(e, dict) and e.get("entity_name")]
        # 与 standardrag 同口径：证据限 10 条，避免 generate_answer 拼爆 prompt
        evidence = evidence[:10]
        answer = generate_answer(question, evidence, dataset=getattr(self, "dataset", ""))
        return {"answer": answer, "evidence": [e["text"] for e in evidence],
                "evidence_locs": [], "entities": ents}

    def run(self, question, contexts):
        """per-question 路径不建议（每题独立建索引太慢），统一走语料级 build/query。"""
        raise NotImplementedError(
            "LightRAG 建议语料级：先 build(全部上下文) 一次，再逐题 query。"
        )

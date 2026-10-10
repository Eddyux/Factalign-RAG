"""HippoRAG 适配器（语料级：build() 建一次知识图，query() 逐题检索）。

HippoRAG = OpenIE 三元组知识图 + Personalized PageRank(PPR) 检索（ICLR 2024）。
语料级口径：N 题上下文合成语料、**不切块**（一个 context = 一个整段块），逐题 query（与 factalign/standardrag 同语料）。

token 统计：通过注入自定义 LLM / embedding 走 common.llm.chat_messages / embed /
embed_batch，使 HippoRAG 的 OpenIE 抽取、rerank、query embedding、答案生成全部计入
build/query 分桶统计（与自带方法同口径）。建索引期间的抽取发生在 HippoRAG 自己的
线程池里，用 set_default_phase("build") 把无 phase 上下文的线程归到 build 桶。

官方仓库：只支持 **HippoRAG 2 现行版**（`baselines/HippoRAG-main`，setup.py `2.0.0a5`）。
它必须能注入 `extraction_llm` / `embedding_model`（`HippoRAG.py:97-99`），注入时**必须**
给 `index_identity`（`:191`），OpenIE 池子自己传 `max_workers`（`openie_openai.py:183/:221`，
来自 `config.openie_max_workers`，默认 8），图文件 `graph.pickle`，并有索引清单与
`StateConsistencyError` 校验。选目录见 `_find_repo()`（`HIPPORAG_REPO` 优先）；
`build_complete.marker` 签名里带 `repo=`：换目录 → 签名不符 → 走"状态变化"分支
（覆盖率够就续跑、不够就重建），不会拿别的目录凑合。
（早先兼容过的 2.0.0-alpha.2 —— 无注入口、裸池、`graph.graphml` —— 已删，不再支持。）

并发：真实并发 = `HIPPORAG_MAX_ASYNC`（默认 64）。官方池子**自己**传 max_workers，
所以关键是把它写进 `config.openie_max_workers`（`_openie_cfg_kwargs()`，只在官方 BaseConfig
真有该字段时给）；另外仍保留对「裸池版」的 `_pool_factory` 补丁作为兜底（对已传参的空操作）。
运行时读模块源码探测走哪条，并打印实际生效的并发，见 `_patch_official_openie()`。
另加适配器侧信号量兜住 retrieve 阶段 DSPyFilter 的 LLM，以及同步 KV 背压——
细节见下面「并发旋钮总表」。**不改 baselines/ 源码。**

抽取失败：官方解析抛异常时把 `metadata['error']` 填上，而新版 `batch_openie` 只要有一个块带
error 就 `raise`，写 json 的 `save_openie_results` 又排在它后面 → 整轮 token 全废。
所以适配器**只对失败的那个块**重试：`HIPPORAG_OPENIE_TRIES`（默认 3，与 HGR/LightRAG 的
3 轮兜底同口径），3 次都用完仍失败就摘掉 error（旧版语义：这块没有实体/事实边、文本仍作
passage 节点）并照旧汇报数量。成功的块一次都不会多跑。
`HIPPORAG_RETRY_ROUNDS`（默认 0）才是"建图后整轮补抽"，默认关闭。

**尝试次数口径（会被问到的）**：一块 = 最多 `HIPPORAG_OPENIE_TRIES`（默认 3）次**抽取**；
每次抽取内部，`_infer_vllm` 在**传输失败（超时/无返回）**时最多重发
`HIPPORAG_LLM_ATTEMPTS`（默认 5）次请求 —— 解析失败那层不会走这个重发（有内容就直接返回，
由上面的单块重试管）。要严格"一次尝试"，设 `HIPPORAG_OPENIE_TRIES=1` +
`HIPPORAG_LLM_ATTEMPTS=1`。

═══ 与官方的差异清单（论文里要注明；**不改 baselines/ 源码**）═══
  1. **块粒度 = 整段（不切块）**：官方这版 `TextPreprocessor.preprocess` 是**恒等映射**
     （`preprocessing.py:15-27`，1 doc = 1 Chunk，`preprocess_chunk_*` 是死配置）→
     我们喂什么它就是什么，官方内部**不会再切**（已核对）。
  2. **embedding_max_seq_len 512**（官方默认 2048）：实际由注入的 `_BgeEmbedding` 决定，
     该字段对我们的 embedding 不生效；真正约束是 bge 512 上限。
  3. **注入 LLM/embedding**：OpenIE 抽取、rerank、query embedding 全部走 common.llm，
     统一计入 build/query 分桶；官方 `infer` 三元组契约保持不变。
  4. **`HIPPORAG_FIX_JSON`（默认开）**：在 `_infer_vllm` 里把实体/三元组形状修好，绕开
     官方 `sum([len(e) ...])`（本版在 `HippoRAG.py:1455-1495`，统计语句 `:1467`）遇到
     非字符串元素就崩、导致整轮 OpenIE 结果不落盘的问题；合规输出字节级不变。
  5. **断点续**：`HIPPORAG_RESUME_COVERAGE`（默认 0.99）决定上一轮没跑完时能否复用
     已落盘的 OpenIE 结果，避免"无 marker → 删目录 → 全量重抽"。
  6. **KNN 设备**：默认照官方走 GPU（`utils/embed_utils.py:21`），可用
     `HIPPORAG_KNN_CPU=1` 让建图后半段的同义边 KNN 走 CPU（GPU 被 vLLM 占满时会 OOM）。
  7. **并发/背压**：OpenIE 抽取并发顶到 `HIPPORAG_MAX_ASYNC`（老版走 `_pool_factory`
     补裸池参数，新版直接把 `openie_max_workers` 写进 BaseConfig，见文件头「并发」段）；
     适配器侧另有在途信号量与同步 KV 背压。

依赖: git clone https://github.com/OSU-NLP-Group/HippoRAG.git 到 baselines/
（HippoRAG-main 也行）。

注意：HippoRAG-main 的包顶层导入会连带 torch/transformers/gritlm/litellm/boto3
（HippoRAG.py、llm/__init__.py、embedding_model/__init__.py 都是顶层 import，
"只装轻依赖"在这版源码上不成立）。本机 base 环境 torch 已损坏（缺顶层 __init__.py，
pip list 显示 2.1.0 但 import 即崩）。要跑 hipporag 需先解决：
1) 修复/重装 torch（与 numpy 2.x 兼容需 >=2.3），以及 transformers/sentence_transformers/
   gritlm/litellm/boto3；
2) 或给上述 4 个文件打轻依赖补丁（重型 import 改 try/except 懒加载，online 模式不碰
   这些类，行为不变）。
API 调用本身已与源码核对一致（HippoRAG.py:45-57 构造、BaseConfig 字段、
infer 三元组契约、batch_encode 调用形态、index/retrieve 签名）。
"""

import importlib
import ast
import inspect
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

from methods.base import BaseMethod, generate_answer
from common.llm import (split_by_tokens, chat_messages, embed, embed_batch,
                        set_default_phase, EMBED_DIM)
from config import (BASELINES_DIR, CHAT_MODEL, EMBED_MODEL_SERVED,
                    EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP)


# ══════════════════════════════════════════════════════════════════════════════
# ★ 并发旋钮总表（只在本适配器生效，不改 baselines/HippoRAG-main 源码）★
#   HIPPORAG_MAX_ASYNC  64  **OpenIE 抽取的并发上限**（一个数管两头）：
#                           (a) 官方 OpenIE 的 NER 池 / 三元组池（openie_openai.py:183/:221）。
#                               官方这版是 `OpenIE(max_workers=global_config.openie_max_workers)`
#                               （HippoRAG.py:223-225），默认只有 8，池子自己就传了参数；
#                               所以适配器**把 `openie_max_workers` 写进 BaseConfig**
#                               （`_openie_cfg_kwargs()`，只在官方 BaseConfig 真有这个字段时给）。
#                               老版官方是裸池（不传 max_workers）→ 走 `_pool_factory` 补参数。
#                               两条路由 `_patch_official_openie()` 运行时探测后各自生效，
#                               patch 类对「池子自己已传参」的版本是空操作，不能当唯一手段；
#                           (b) 适配器侧信号量，兜住 retrieve 阶段 DSPyFilter 的 LLM。
#                           两个池仍是官方那样先后串行，所以峰值 = HIPPORAG_MAX_ASYNC。
#   HIPPORAG_BP          1   同步 KV 背压：每 HIPPORAG_BP_SEC 秒读一次 vLLM `/metrics`，
#                            KV ≥ HIPPORAG_BP_KV_HIGH 或 waiting>0 就退让再抢槽。
#                            （HippoRAG 的 LLM 是同步调用，用不了 chat_retry 的 async 闸，
#                              所以这里写的是同步版背压。）
#   HIPPORAG_OPENIE_TRIES 3  ★ 官方更新后的硬失败的**单块**补救：`batch_openie` 只要有一个块
#                            带 `metadata['error']` 就 `raise RuntimeError`
#                            （openie_openai.py:214-216 NER / :252-254 三元组），而写 json 的
#                            `save_openie_results` 排在它**后面** → 一整轮几百万 token 全废
#                            （实测 3012 块里 2 块失败就把建图打断）。这里**只对失败的那个块**
#                            重试（粒度到 chunk，成功的块不受影响），最多 N 次尝试；
#                            仍失败就摘掉 error（官方旧版语义：这块没有实体/事实边、文本仍作
#                            passage 节点），build 末尾照旧汇报失败块数量。
#                            设 1 = 不重试（但仍会摘 error，不会打断整轮）；设 4 = 3 次重试。
#   HIPPORAG_RETRY_ROUNDS 0  建图**之后**再整轮补抽失败块（删条目 → 重跑官方增量）。
#                            **默认 0 = 不补抽**：temperature=0 贪心解码下同一 prompt 基本
#                            复现同一份畸形输出，补抽多半白花 LLM。设 >0 开启（如 2）。
#   HIPPORAG_FIX_JSON    1   OpenIE 输出形状修复（默认开，关掉 = HIPPORAG_FIX_JSON=0）：
#                            NER 里混进 int/bool 实体 → 官方 save_openie_results 的
#                            `sum(len(e) ...)`（HippoRAG.py:1124，**无 try/except**）直接把整轮
#                            index() 打断，而 json 要到 :1141 才写 → 几千块的抽取全废。
#                            修法：int/float → str，dict/None/bool 丢掉，三元组只留 3 元。
#                            **形状本来就合规时字节级原样返回**，对正常抽取零影响。
#   HIPPORAG_RESUME_COVERAGE 0.99
#                            上一轮没跑完（无 marker）时，已落盘 OpenIE 结果覆盖当前块达到这个
#                            比例 → 续跑增量不删目录（官方 openie json 就是内容 hash 索引的
#                            检查点，HippoRAG.py:1021-1065）；低于它才按「状态变化」重建。
#   HIPPORAG_KNN_CPU     0   建图算同义边时是否把官方 `retrieve_knn` 强制到 CPU。
#                            **默认 0 = 不改，照官方走 GPU**（`embed_utils.py:21`
#                            `cuda if torch.cuda.is_available()`）。走 GPU 需要**空出显存**：
#                            同义边 KNN 峰值几百 MB（key 向量 n×1024 + 相似度 1000×10000），
#                            而 vLLM 按 0.85 占满时只剩 6 MiB → 必 CUDA OOM。
#                            两条路二选一：① 把 vLLM 的 --gpu-memory-utilization 降到 ~0.80
#                            （留 ~2 GB，代价是 KV 池小 ~15k token）；② 设成 1 让这步走 CPU
#                            （只影响建图，查询路径不用这个函数）。
# ══════════════════════════════════════════════════════════════════════════════
HIPPORAG_MAX_ASYNC = int(os.environ.get("HIPPORAG_MAX_ASYNC", "64"))
HIPPORAG_BP_ENABLE = os.environ.get("HIPPORAG_BP", "1") == "1"
HIPPORAG_BP_SEC = float(os.environ.get("HIPPORAG_BP_SEC", "3"))
HIPPORAG_BP_KV_HIGH = float(os.environ.get("HIPPORAG_BP_KV_HIGH", "0.85"))
HIPPORAG_RETRY_ROUNDS = int(os.environ.get("HIPPORAG_RETRY_ROUNDS", "0"))
# ★ 单个块抽取失败的**总尝试次数**（首次 + 重试），默认 3 —— 与 HGR/LightRAG 的 3 轮兜底同口径。
#   官方更新后 `batch_openie` 只要有一个块带 `metadata['error']` 就 `raise RuntimeError`
#   （openie_openai.py:214-216 / :252-254），而写 json 的 `save_openie_results` 排在它后面
#   → 一整轮几百万 token 全废（实测 3012 块里 2 块失败就把建图打断）。
#   这里**只对失败的那个块**重试（包装在 OpenIE.ner / triple_extraction 上，粒度到 chunk），
#   成功的块一次都不会多跑；3 次都用完仍失败 → 摘掉 error（旧版语义：这块没有实体/事实边、
#   文本仍作 passage 节点）并登记，build 末尾汇报数量。
#   注：NER 一直失败的块，官方仍会对它跑一次 triple（实体列表为空），那一层同样最多 3 次。
#   要"3 次重试（共 4 次）"就设 4；要严格一次尝试就设 1。
HIPPORAG_OPENIE_TRIES = int(os.environ.get("HIPPORAG_OPENIE_TRIES", "3"))
# 单次 `_infer_vllm` 里**传输失败**（vLLM 无返回/超时）时的最大请求次数（默认 5）。
# 解析失败不走这里（有内容就直接返回，由上面那个单块重试管）。
# 要严格「一次尝试」：HIPPORAG_OPENIE_TRIES=1 + HIPPORAG_LLM_ATTEMPTS=1。
HIPPORAG_LLM_ATTEMPTS = int(os.environ.get("HIPPORAG_LLM_ATTEMPTS", "5"))
# OpenIE 输出的形状修复（默认开）：官方解析器对「实体里混进 int」「三元组里混进 int」会直接崩，
# 其中 save_openie_results 的 `len(e)`（HippoRAG.py:1124）没有 try/except → 整轮建图全废
# 且 json 一个字都没落盘。形状本来就合规时**原样返回**，对正常抽取零影响。
HIPPORAG_FIX_JSON = os.environ.get("HIPPORAG_FIX_JSON", "1") == "1"
# 上一轮没跑完（无 marker）时，已有 OpenIE 结果覆盖当前块的这个比例以上 → 续跑增量，不删目录。
# 官方 openie json 是按内容 hash 索引的增量检查点（HippoRAG.py:1021-1065），块没变就能续。
HIPPORAG_RESUME_COVERAGE = float(os.environ.get("HIPPORAG_RESUME_COVERAGE", "0.99"))
# 同义边 KNN 是否强制走 CPU（**默认关 = 照官方走 GPU**；建图最后一步 OOM 时设成 1）：
# 官方 `retrieve_knn` 在**调用时**选设备（`device = 'cuda' if torch.cuda.is_available()
# else 'cpu'`，`utils/embed_utils.py:20`），由建图的 `add_synonymy_edges` 调用
# （a5:HippoRAG.py:1306，tqdm 描述就是 "KNN for Queries"）。vLLM 按
# --gpu-memory-utilization 0.85 占满时几乎没余量（实测 free 6 MiB）→ 这一步一申请就
# CUDA OOM。它只在建图时用，查询路径不用，所以放 CPU 只影响建图那一次。
HIPPORAG_KNN_CPU = os.environ.get("HIPPORAG_KNN_CPU", "0") == "1"

# ══════════════════════════════════════════════════════════════════════════════
# ★ 官方长度 / 预算限制（默认 = 官方 BaseConfig 原值，见
#   baselines/HippoRAG-main/src/hipporag/utils/config_utils.py）
#   对齐实验直接用这些键；改完换 --graph-version 重建。
# ══════════════════════════════════════════════════════════════════════════════
# ⚠️ retrieval_top_k / qa_top_k 的作用（官方 HippoRAG.py 为证）：
#   linking_top_k   链接几个图节点当 PPR 种子            (:2128 link_top_k=)
#   retrieval_top_k PPR 算出/返回 top N 篇候选（检索深度）(:729/804/990 num_to_retrieve=)
#   qa_top_k        只把前 N 篇喂给答题模型（证据预算）    (:1130 docs[:qa_top_k])
# 注意：**官方 rag_qa() 我们没走**（query() 自己调 retrieve + 截断），所以这两个值
# 真正生效的位置是下面的 query()，BaseConfig 里也同步传一份以免"死配置"误导。
HIPPORAG_LINKING_TOP_K = int(os.environ.get("HIPPORAG_LINKING_TOP_K", "5"))       # 官方 5
HIPPORAG_RETRIEVAL_TOP_K = int(os.environ.get("HIPPORAG_RETRIEVAL_TOP_K", "200"))  # 官方 200
HIPPORAG_QA_TOP_K = int(os.environ.get("HIPPORAG_QA_TOP_K", "5"))                 # 官方 5
HIPPORAG_MAX_QA_STEPS = int(os.environ.get("HIPPORAG_MAX_QA_STEPS", "1"))         # 官方 1
HIPPORAG_DAMPING = float(os.environ.get("HIPPORAG_DAMPING", "0.5"))               # 官方 0.5（PPR 阻尼）
HIPPORAG_SYN_THRESHOLD = float(os.environ.get("HIPPORAG_SYN_THRESHOLD", "0.8"))   # 官方 0.8（同义边阈值）
# OpenIE 两个上限是**输出**上限（官方 metadata: "Maximum OUTPUT tokens for each online
# OpenIE NER/triple request"），不是输入截断：
HIPPORAG_NER_MAX_TOKENS = int(os.environ.get("HIPPORAG_NER_MAX_TOKENS", "512"))    # 官方 512
HIPPORAG_TRIPLE_MAX_TOKENS = int(os.environ.get("HIPPORAG_TRIPLE_MAX_TOKENS", "2048"))  # 官方 2048
# embedding_max_seq_len = **embedder 单条输入的声明上限**（官方默认 2048，因为官方用
# Contriever/NVEmbedV2 能吃长输入；见 embedding_model/Contriever.py:63 的 tokenizer
# max_length）。我们注入自己的 _BgeEmbedding，真实截断由全局 fit_embed() 按
# EMBED_MAX_TOKENS 执行。
# ★ 已按决定对齐官方 2048：不切块后本项目单块 = 一个整段 passage，triviaqa 的整篇维基文章
# 会被全局 fit_embed() 按 EMBED_MAX_TOKENS 截断，所以 2048 仍只是个
# 一个「与官方同口径」的声明值（本项目 embedder Qwen3-Embedding-0.6B 真实窗宽远大于此，
# 填 2048 不会虚假声明一个模型没有的能力）。
# ⚠️ 该值会进索引签名（utils/state_utils.py:45 "max_sequence_length"）→ 改它必须重建。
HIPPORAG_EMBED_MAX_SEQ_LEN = int(os.environ.get("HIPPORAG_EMBED_MAX_SEQ_LEN", "2048"))
_SLOTS = threading.Semaphore(HIPPORAG_MAX_ASYNC)    # 适配器侧在途上限
_BP = {"t": 0.0, "kv": 0.0, "waiting": 0}
_BP_LOCK = threading.Lock()
_POOL_PATCHED = False
# 官方 OpenIE 池子是否自己显式传 max_workers（_patch_official_openie 运行时探测）：
# True = 新版，必须把 openie_max_workers 写进 BaseConfig（patch 类是空操作）；
# False = 老版裸池，_pool_factory 补参数即生效。
_OPENIE_EXPLICIT_WORKERS = False
# 抽取失败的块（官方解析抛异常时 metadata['error'] 被填上，见 openie_openai.py:66/:114）
_OPENIE_FAILS = {"ner": set(), "triple": set()}
_FAIL_LOCK = threading.Lock()




def _kv_backpressure():
    """KV/排队过高就 sleep 退让（同步路径的「自适应并发」）。"""
    if not HIPPORAG_BP_ENABLE:
        return
    now = time.time()
    with _BP_LOCK:
        if now - _BP["t"] < HIPPORAG_BP_SEC:
            kv, waiting = _BP["kv"], _BP["waiting"]
        else:
            kv = waiting = None
    if kv is None:
        try:
            from common.llm import vllm_engine_stats
            st = vllm_engine_stats()
        except Exception:                                   # noqa: BLE001
            st = None
        kv, _running, waiting = st if st else (0.0, 0, 0)
        with _BP_LOCK:
            _BP.update(t=now, kv=kv, waiting=waiting)
    if waiting and waiting > 0:
        time.sleep(min(2.0, 0.2 * waiting))                 # 排队越长退让越久（上限 2s）
    elif kv and kv >= HIPPORAG_BP_KV_HIGH:
        time.sleep(0.5)


def _pool_factory(factory, workers: int = HIPPORAG_MAX_ASYNC):
    """包一层线程池工厂：官方裸 `ThreadPoolExecutor()` 调用点自动补 max_workers。"""
    def _tp(*args, **kwargs):
        if not args and kwargs.get("max_workers") is None:
            kwargs["max_workers"] = workers
        return factory(*args, **kwargs)
    return _tp




def _patch_official_openie() -> None:
    """给官方 OpenIE 打三个只在适配器里的补丁（不改 baselines/ 源码）：

    1. **并发**：官方两处池子的 max_workers 来源在不同版本不一样，运行时探测：
       · 裸池版：`ThreadPoolExecutor()`（openie_openai.py NER / 三元组），
         靠替换模块里的 `ThreadPoolExecutor` 名字补参数（函数体在**调用时**才查模块全局）。
       · 显式版（新版）：`ThreadPoolExecutor(max_workers=self.max_workers)`，
         来自 `config.openie_max_workers`（默认 8）→ 上面的替换是**空操作**，
         必须由 `_openie_cfg_kwargs()` 把字段写进 BaseConfig。两版都保证 64。
    2. **失败登记**：包一层 `OpenIE.ner` / `OpenIE.triple_extraction`，把
       `metadata['error']` 非空的块 id 记到 `_OPENIE_FAILS`——官方在解析抛异常时填这个字段
       （openie_openai.py:87-95 / :136-138），是「这块崩了」唯一精确的信号
       （落盘的 json 里只有空列表，分不出「崩了」和「这段真没三元组」）。
    3. **★ 官方更新带来的硬失败：单块重试 + 摘掉 error**。新版 `batch_openie` 在
       **任意一个块**有 error 时直接 `raise RuntimeError`（`:214-216` NER / `:252-254`
       三元组），而 `save_openie_results` 排在它**后面**才写 json → 一整轮抽取（实测
       3012 块里 2 块失败）直接打断、几百万 token 一个字都不落盘。这里**只对失败的那个块**
       最多尝试 `HIPPORAG_OPENIE_TRIES` 次（默认 3，粒度到 chunk），仍失败就摘掉
       `metadata['error']`——回到官方旧版
       语义（这块没有实体/事实边，文本仍留作 passage 节点），数量照旧汇报。这样
       `index()` 里 chunk/NER/triples 的数量校验仍然成立，json 也照常落盘。
    """
    global _POOL_PATCHED, _OPENIE_EXPLICIT_WORKERS
    if _POOL_PATCHED:
        return
    try:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        mod = importlib.import_module("hipporag.information_extraction.openie_openai")
    except Exception as e:                                   # noqa: BLE001
        print(f"[hipporag] 官方 OpenIE 补丁没打上（{type(e).__name__}: {e}）"
              f"→ 抽取并发仍是官方默认 min(32, cpu+4)，失败块也不会自动补抽", flush=True)
        return
    if getattr(mod, "_HIPPORAG_POOL_PATCHED", False):
        _POOL_PATCHED = True
        return
    # 探测：官方池子自己传不传 max_workers（决定 64 该从哪条路生效）
    try:
        _OPENIE_EXPLICIT_WORKERS = bool(
            re.search(r"ThreadPoolExecutor\(\s*max_workers", inspect.getsource(mod)))
    except Exception:                                        # noqa: BLE001
        _OPENIE_EXPLICIT_WORKERS = False

    mod.ThreadPoolExecutor = _pool_factory(_TPE)

    for meth, kind in (("ner", "ner"), ("triple_extraction", "triple")):
        orig = getattr(mod.OpenIE, meth)

        def _rec(self, chunk_key, *a, _orig=orig, _kind=kind, **kw):
            out = _orig(self, chunk_key, *a, **kw)
            if not _openie_has_error(out):
                return out
            # ★ 守卫：**输入为空的三元组抽取不值得重抽**。
            #   官方 `triple_extraction(chunk_key, passage, named_entities)` 的实体列表
            #   来自 NER 的输出（`openie_openai.py:156`）。NER 多次失败时我们返回空实体，
            #   官方照样拿 `{"named_entities": []}` 发一次 —— 那次**没有成功可能**，
            #   再重发只是把同一份注定失败的输出又拿两次（temperature=0 下更是一模一样）。
            #   首抽仍照官方发出（**不改官方调用次数**），只省掉后面那几次。
            if (_kind == "triple" and HIPPORAG_OPENIE_TRIES > 1
                    and _named_entities_empty(a, kw)):
                with _FAIL_LOCK:
                    _OPENIE_FAILS[_kind].add(chunk_key)
                _clear_openie_error(out)
                print(f"[hipporag] triple 块 {chunk_key} 输入实体为空 → 不重抽"
                      f"（首抽已按官方发出；空实体抽三元组无成功可能，"
                      f"省 {HIPPORAG_OPENIE_TRIES - 1} 次）", flush=True)
                return out
            # 只对**这个失败的块**重试（粒度到 chunk，成功的块不受影响）：
            # 官方更新后任一块失败会 raise 掉整个 batch（见本函数 docstring 第 3 条）
            for try_no in range(2, HIPPORAG_OPENIE_TRIES + 1):
                out2 = _orig(self, chunk_key, *a, **kw)
                if not _openie_has_error(out2):
                    with _FAIL_LOCK:
                        _OPENIE_FAILS[_kind].discard(chunk_key)
                    print(f"[hipporag] {_kind} 第 {try_no}/{HIPPORAG_OPENIE_TRIES} 次尝试成功："
                          f"{chunk_key}", flush=True)
                    return out2
                out = out2
            with _FAIL_LOCK:
                _OPENIE_FAILS[_kind].add(chunk_key)
            _clear_openie_error(out)      # ← 摘掉 error，官方的 batch 级 raise 就不会触发
            # ★ 必须打这一行：官方失败时只 `logger.warning(e)`（**不带块号**），
            #   成功路径才打「第 N 次尝试成功」。若重试用尽后这里不打，
            #   日志里就只剩几条裸的 "OpenIE response does not contain..."，
            #   看日志的人无法判断到底重抽了没有、是哪个块。
            print(f"[hipporag] {_kind} 块 {chunk_key} 抽取 {HIPPORAG_OPENIE_TRIES} 次全失败"
                  f" → 转空结构（该块无实体/事实边，文本仍留作 passage 节点，不打断整轮建图）",
                  flush=True)
            return out

        setattr(mod.OpenIE, meth, _rec)

    mod._HIPPORAG_POOL_PATCHED = True                        # 幂等标记（防重复包装）
    _POOL_PATCHED = True
    if _OPENIE_EXPLICIT_WORKERS:
        where = (f"新版官方池子自己传 max_workers → 由 BaseConfig.openie_max_workers 生效"
                 if "openie_max_workers" in _baseconfig_fields()
                 else "[warn] 新版官方池子自己传 max_workers，但 BaseConfig 没有该字段 → 仍是官方默认 8")
    else:
        where = "老版官方裸池 → 由 _pool_factory 补 max_workers 生效"
    print(f"[hipporag] OpenIE 抽取并发 = {HIPPORAG_MAX_ASYNC}（{where}）"
          f"；单个块失败最多尝试 {HIPPORAG_OPENIE_TRIES} 次"
          f"（首次 + {HIPPORAG_OPENIE_TRIES - 1} 次重试，只重试失败的块，"
          f"用尽后转「空结构」不打断整轮建图）"
          f"；建图后补抽 "
          f"{'关闭（HIPPORAG_RETRY_ROUNDS=0）' if HIPPORAG_RETRY_ROUNDS <= 0 else f'最多 {HIPPORAG_RETRY_ROUNDS} 轮'}"
          f"；同步 KV 背压 {'on' if HIPPORAG_BP_ENABLE else 'off'}",
          flush=True)


def _openie_has_error(out) -> bool:
    """这块的抽取是否失败（官方解析异常时把 metadata['error'] 填上）。"""
    meta = getattr(out, "metadata", None)
    return bool(isinstance(meta, dict) and meta.get("error"))


def _named_entities_empty(a: tuple, kw: dict) -> bool:
    """这次 `triple_extraction` 调用拿到的 `named_entities` 是否为空。

    官方签名 `triple_extraction(self, chunk_key, passage, named_entities)`
    （`openie_openai.py:105`）→ 位置参数里是 `a[1]`，也可能以关键字传。
    拿不到就不猜（返回 False）→ 维持原来的重试行为，绝不因为解析参数
    出意外而**少**重试。
    """
    ne = kw.get("named_entities")
    if ne is None and len(a) >= 2:
        ne = a[1]
    if ne is None:
        return False
    try:
        return len(ne) == 0
    except TypeError:                                          # 不可测量 → 不猜
        return False


def _clear_openie_error(out) -> None:
    """摘掉 `metadata['error']`，但留个我们自己的标记 —— 官方 batch 级校验只看 error 字段。

    语义回到官方**旧版**：这块没有实体/事实边，文本仍然留在图里当 passage 节点，
    整轮抽取不会被一块畸形输出废掉（并照常在 build 末尾汇报有多少块失败）。
    """
    meta = getattr(out, "metadata", None)
    if isinstance(meta, dict):
        meta.pop("error", None)
        meta["factalign_openie_failed"] = True


def _baseconfig_fields() -> set:
    """官方 BaseConfig 的字段名（拿不到就空集，调用方按「字段不存在」处理）。"""
    try:
        import dataclasses
        from hipporag.utils.config_utils import BaseConfig
        return {f.name for f in dataclasses.fields(BaseConfig)}
    except Exception:                                        # noqa: BLE001
        return set()


def _openie_cfg_kwargs() -> dict:
    """官方池子自己传 max_workers 时，唯一有效的并发旋钮是 config.openie_max_workers。

    老版没有这个字段 → 传了就是 `unexpected keyword argument` 崩，所以按字段存在与否给。
    """
    if not _OPENIE_EXPLICIT_WORKERS:
        return {}
    return ({"openie_max_workers": HIPPORAG_MAX_ASYNC}
            if "openie_max_workers" in _baseconfig_fields() else {})


def _official_cfg_kwargs() -> dict:
    """官方长度/预算限制写进 BaseConfig（按字段存在与否过滤，跨版本不崩）。

    与 `_openie_cfg_kwargs()` 同一思路：老版官方没有的字段，传了就是
    `unexpected keyword argument` 崩，所以逐个按 `_baseconfig_fields()` 过滤。
    """
    want = {
        "linking_top_k": HIPPORAG_LINKING_TOP_K,
        "retrieval_top_k": HIPPORAG_RETRIEVAL_TOP_K,
        "qa_top_k": HIPPORAG_QA_TOP_K,
        "max_qa_steps": HIPPORAG_MAX_QA_STEPS,
        "damping": HIPPORAG_DAMPING,
        "synonymy_edge_sim_threshold": HIPPORAG_SYN_THRESHOLD,
        "openie_ner_max_tokens": HIPPORAG_NER_MAX_TOKENS,
        "openie_triple_max_tokens": HIPPORAG_TRIPLE_MAX_TOKENS,
        "embedding_max_seq_len": HIPPORAG_EMBED_MAX_SEQ_LEN,
    }
    fields = _baseconfig_fields()
    if not fields:                       # 拿不到字段表 → 一个都不给，保持官方默认
        return {}
    return {k: v for k, v in want.items() if k in fields}


def _openie_failed_keys(path) -> list:
    """扫 openie 结果 json：三元组为空的块 id（只用于最后的兜底汇报）。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception:                                        # noqa: BLE001
        return []
    return [d.get("idx") for d in data.get("docs", []) if not d.get("extracted_triples")]


def _remove_openie_keys(path, keys) -> int:
    """把指定块从 openie 结果里删掉 → 官方增量逻辑下次会重抽它们。

    必须**删条目**而不是清空：清空仍被 `load_existing_openie`（HippoRAG.py:1021-1065）
    当成「已处理」，`HippoRAG.py:306` 的 assert 也要靠同一轮补回来才成立。
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    drop = set(keys)
    docs = data.get("docs", [])
    kept = [d for d in docs if d.get("idx") not in drop]
    data["docs"] = kept
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return len(docs) - len(kept)


def _repair_failed_openie(rag, docs, rounds: int | None = None) -> list:
    """建图后补抽抽取失败的块（默认关闭，见 HIPPORAG_RETRY_ROUNDS），返回仍失败的块 id。

    官方把解析失败的块也落盘成空条目（HippoRAG.py:1097-1108），下一轮
    `load_existing_openie` 就当成「已处理」→ 永远不会重抽。这里把失败块从 openie 结果里
    删掉再走一次官方增量：只有它们会重新发 LLM。

    必须同时 `force_index_from_scratch = True`：官方加边是
    `if chunk_key not in current_graph_nodes`（HippoRAG.py:905 / :948），上一轮已经把
    passage 节点建好了——不重建图的话，新抽出来的三元组**不会进图**。
    """
    rounds = HIPPORAG_RETRY_ROUNDS if rounds is None else rounds
    with _FAIL_LOCK:                       # 不补抽时直接汇报，别去碰版本相关的属性
        if rounds <= 0:
            return sorted(_OPENIE_FAILS["ner"] | _OPENIE_FAILS["triple"])
    openie_json = rag.openie_results_path
    for rnd in range(1, rounds + 1):
        with _FAIL_LOCK:
            bad = sorted(_OPENIE_FAILS["ner"] | _OPENIE_FAILS["triple"])
            _OPENIE_FAILS["ner"].clear()
            _OPENIE_FAILS["triple"].clear()
        if not bad:
            break
        n_del = _remove_openie_keys(openie_json, bad)
        print(f"[hipporag] 补抽第 {rnd}/{rounds} 轮：{len(bad)} 块抽取失败"
              f"（已从 openie 结果删 {n_del} 条）→ 重跑官方增量", flush=True)
        rag.global_config.force_index_from_scratch = True
        rag.index(docs)
        with _FAIL_LOCK:
            left = sorted(_OPENIE_FAILS["ner"] | _OPENIE_FAILS["triple"])
        print(f"[hipporag] 补抽第 {rnd} 轮结束："
              f"{'全部补回' if not left else f'仍有 {len(left)} 块失败'}", flush=True)
        if not left:
            break
    with _FAIL_LOCK:
        return sorted(_OPENIE_FAILS["ner"] | _OPENIE_FAILS["triple"])


def _json_mode(messages):
    text = " ".join((m.get("content", "") or "")
                    for m in messages if isinstance(m, dict)).lower()
    return "json" in text or "named_entities" in text or '"triples"' in text


def _openie_kind(messages) -> str | None:
    """这次调用是 NER 还是三元组抽取（看 prompt 里要哪个字段）。

    为什么要判：官方解析器只认 `{"named_entities": [...]}` / `{"triples": [...]}`
    （`_extract_json_list_field`，openie_openai.py:29-44）。模型若返回裸数组或
    `{"entities": [...]}` 这种换个名字的，官方直接报
    `OpenIE response does not contain a valid JSON object with 'named_entities'`
    （就是线上那两条）。要修形状就得先知道该修成哪个字段。

    ★★ **两个 if 的顺序不能反**（线上事故，5012 块全中）★★
    官方的三元组 prompt 把**实体列表当输入**塞进 user 帧
    （`openie_openai.py:113` 的 `named_entity_json=json.dumps({"named_entities": ...})`，
    模板 `templates/triple_extraction.py:14-21` 的 `{named_entity_json}`），
    所以**每一次三元组调用里都同时出现 `named_entities` 和 `triples`**。
    先判 named_entities → 三元组调用一律被判成 ner → `_normalize_openie_payload`
    把合规的 `{"triples": [...]}` 按实体列表修（`_coerce_entity` 把 list 元素全丢掉）
    → 改写成 `{"named_entities": []}` → 官方解析抛
    `OpenIE response does not contain a valid JSON object with 'triples'`，
    单块 3 次重试全废（temperature=0 必然复现）→ 图里一条事实边都没有。
    反向判不会误伤：NER 模板（`templates/ner.py`）里**没有** "triple" 字样；
    DSPy filter 模板（`filter_default_prompt.py`）两个词都没有（kind 仍是 None）。
    """
    text = " ".join((m.get("content", "") or "")
                    for m in messages if isinstance(m, dict)).lower()
    if "triple" in text:
        return "triple"
    if "named_entities" in text or "named entities" in text:
        return "ner"
    return None


# ── OpenIE 输出形状修复（HIPPORAG_FIX_JSON）────────────────────────────────────
_NER_JSON_RE = re.compile(r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}', re.DOTALL)
_TRIPLE_JSON_RE = re.compile(r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}', re.DOTALL)
# 字段别名：模型爱换名字（官方只认一个）
_FIELD_ALIASES = {
    "named_entities": ("named_entities", "named entities", "namedentities", "entities",
                       "entity_list", "entities_list", "named_entity_list"),
    "triples": ("triples", "triple", "relations", "relation_list", "facts", "triple_list"),
}


def _first_json_value(text: str, opener: str):
    """扫出第一个能解析的 JSON 值（dict 或 list），返回 (值, 起, 止)；找不到返回 None。

    官方解析器就是这种"从任意位置 raw_decode"的写法（`_extract_json_list_field`），
    所以模型在 JSON 前后加解释文字不影响——这里照抄同一套扫描口径。
    """
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != opener:
            continue
        try:
            val, end = dec.raw_decode(text[i:])
        except Exception:                                    # noqa: BLE001
            continue
        return val, i, i + end
    return None


def _parse_json_lenient(raw: str):
    """先 json，再退到 ast.literal_eval（官方就是用 eval 解单引号"JSON"的）。"""
    for fn in (json.loads, ast.literal_eval):
        try:
            return fn(raw)
        except Exception:                                    # noqa: BLE001
            continue
    return None


def _coerce_entity(e):
    """实体统一成 str；None/dict/list 丢掉（官方遇到 dict 是整块丢弃，我们只是不崩）。"""
    if isinstance(e, str):
        return e
    if e is None or isinstance(e, (dict, list, tuple, bool)):
        return None
    return str(e)                                            # int / float


def _normalize_openie_payload(content, kind: str | None = None):
    """把 OpenIE 的 JSON 输出修成官方解析器吃得下的形状；合规就**原样返回**。

    官方有两处会被畸形元素干掉：
      - NER：`dict.fromkeys(extracted_entities)`（openie_openai.py:61）元素是 dict → unhashable
        → 被 :63-72 吞掉，这块实体丢光（不致命）；
      - 实体统计：`sum(len(e) ...)`（HippoRAG.py:1124，save_openie_results）元素是 int
        → `object of type 'int' has no len()`，**没有 try/except** → 整个 index() 被打断，
        而 json 要到 :1141 才写 → 几千块的抽取全废，一个字都没落盘。
    三元组同理：`filter_invalid_triples` 的 `len(triple)`（llm_utils.py:247）遇到 int 元素崩。

    ★ 还有一类是**形状不对**（线上实际撞到的）：官方解析器只认 `{"named_entities": [...]}`，
    模型给**裸数组** `["A","B"]` 或**换名字** `{"entities": [...]}` 时直接抛
    `OpenIE response does not contain a valid JSON object with 'named_entities'`。
    temperature=0 下重试同一 prompt 多半复现同一份 → **能修形状就别靠重试**：
    这里把「别名字段」和「裸数组」都归一成官方要的那个字段（`kind` 指明是 ner 还是 triple；
    不传时退回老的字面量启发式）。

    ★ 两条保命规则（都是线上事故换来的，别删）：
      ① JSON 对象里装的其实是**另一个字段**时，按它实际是什么来修——kind 判错时绝不把
         内容改写成目标字段的空列表（`{"triples": [...]}` → `{"named_entities": []}`
         就是这条；紧接着官方就抛 `... valid JSON object with 'triples'`）；
      ② 只有整段里**根本没有 JSON 对象**时，才把裸数组包成对象——DSPy filter 的
         `{"fact": [[...]]}` 里也含裸三元组数组，但那是另一个字段、归 rerank.py 自己解析，
         一旦被这里重包成 `{"triples": [...]}` 就整段筛选结果丢失。
    """
    if not HIPPORAG_FIX_JSON or not content:
        return content

    # 需要修的目标字段：优先用调用方给的 kind，其次看字面量
    fields = []
    if kind == "ner":
        fields = ["named_entities"]
    elif kind == "triple":
        fields = ["triples"]
    else:
        if '"named_entities"' in content:
            fields.append("named_entities")
        if '"triples"' in content:
            fields.append("triples")

    for field in fields:
        aliases = _FIELD_ALIASES[field]
        found = _first_json_value(content, "{")
        obj = found[0] if found else None

        # ── A. 找到了 JSON 对象：字段名可能是别名，值可能是非列表 ──
        if isinstance(obj, dict):
            key = next((k for k in obj if str(k).strip().lower() in aliases), None)
            if key is None:
                # ★ 保命规则①：对象里装的其实是**另一个字段**（kind 判错时就是这种，见
                # `_openie_kind`）→ 按它实际是什么来修，绝不改写成目标字段的空列表。
                other = "triples" if field == "named_entities" else "named_entities"
                key = next((k for k in obj
                            if str(k).strip().lower() in _FIELD_ALIASES[other]), None)
                if key is not None:
                    field, aliases = other, _FIELD_ALIASES[other]
            if key is not None:
                val = obj[key]
                if isinstance(val, dict):                     # {"named_entities": {"a": 1}}
                    val = list(val.keys())
                if not isinstance(val, list):
                    val = [val]
                fixed = ([c for c in (_coerce_entity(e) for e in val) if c]
                         if field == "named_entities" else _fix_triples(val))
                # 只在"真的需要改"时才重写（别名/形状/元素都合规 → 原样返回）
                if key != field or fixed != val or list(obj.keys()) != [field]:
                    return json.dumps({field: fixed}, ensure_ascii=False)
                return content

        # ── B. 连 JSON 对象都没有，只有**裸数组** → 按元素形状判该包成哪个字段 ──
        #    ★ 保命规则②：形状自证优先于 kind（`[[...]]` 只可能是三元组，标量数组只可能是
        #    实体），并且**只在整段没有 JSON 对象时**才动手（见 docstring 第 ② 条）。
        if obj is None:
            arr = _first_json_value(content, "[")
            val = arr[0] if arr else None
            if isinstance(val, list) and val and all(isinstance(e, (list, tuple)) for e in val):
                fixed = _fix_triples(val)
                if fixed:                                    # 修完是空的 → 不猜，原样交回官方
                    return json.dumps({"triples": fixed}, ensure_ascii=False)
            elif isinstance(val, list) and field == "named_entities":
                fixed = [c for c in (_coerce_entity(e) for e in val) if c]
                if fixed:
                    return json.dumps({"named_entities": fixed}, ensure_ascii=False)

    # ── C. 老的字面量路径（保证原有行为不变：只在元素形状不对时改）──
    if '"named_entities"' in content:
        m = _NER_JSON_RE.search(content)
        obj = _parse_json_lenient(m.group()) if m else None
        if isinstance(obj, dict) and isinstance(obj.get("named_entities"), list):
            ents = obj["named_entities"]
            fixed = [c for c in (_coerce_entity(e) for e in ents) if c]
            if fixed != ents:
                obj["named_entities"] = fixed
                return content[:m.start()] + json.dumps(obj) + content[m.end():]
    if '"triples"' in content:
        m = _TRIPLE_JSON_RE.search(content)
        obj = _parse_json_lenient(m.group()) if m else None
        if isinstance(obj, dict) and isinstance(obj.get("triples"), list):
            fixed = _fix_triples(obj["triples"])
            if fixed != obj["triples"]:
                obj["triples"] = fixed
                return content[:m.start()] + json.dumps(obj) + content[m.end():]
    return content


def _fix_triples(triples: list) -> list:
    """三元组只留 3 元的字符串项（官方本意是「跳过不合法项」，这里替它跳）。"""
    out = []
    for t in triples:
        if not isinstance(t, (list, tuple)) or len(t) != 3:
            continue
        if any(x is None or isinstance(x, (dict, list, tuple)) for x in t):
            continue
        out.append([x if isinstance(x, str) else str(x) for x in t])
    return out


def _openie_coverage(working_dir: Path, docs: list) -> float:
    """已落盘 openie 结果覆盖当前块的比例（按 passage 原文比对）。

    只在「当前 CHAT_MODEL 对应的那份 json」上算；换模型时文件名就不同 → 返回 0（照旧重建）。
    """
    p = working_dir / f"openie_results_ner_{CHAT_MODEL.replace('/', '_')}.json"
    if not p.exists() or not docs:
        return 0.0
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:                                        # noqa: BLE001
        return 0.0
    have = {d.get("passage") for d in data.get("docs", [])}
    return sum(1 for t in docs if t in have) / len(docs)


# ── 同义边 KNN 强制 CPU（HIPPORAG_KNN_CPU）─────────────────────────────────────
_ORIG_KNN = None
_KNN_LOCK = threading.Lock()


def _force_cpu_knn(*args, **kwargs):
    """临时让 torch「看不见 GPU」，于是官方那句 device 选择落到 cpu（不复制官方算法）。"""
    import torch
    with _KNN_LOCK:
        orig = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        try:
            return _ORIG_KNN(*args, **kwargs)
        finally:
            torch.cuda.is_available = orig


def _patch_knn_cpu() -> None:
    """把官方 `retrieve_knn` 换成 CPU 版。**默认关闭**（`HIPPORAG_KNN_CPU=0`，照官方走 GPU）。

    开启后：官方 `utils/embed_utils.py:21` 的 `torch.device('cuda' if torch.cuda.is_available()
    else 'cpu')` 会落到 cpu。GPU 被 vLLM 占着 `--gpu-memory-utilization 0.85`（≈29/31.4 GB）
    时，这一步一申请就 `CUDA out of memory`（实测 free 6.25 MiB）→ 建图在 `KNN for Queries`
    挂掉；走 CPU 则只多花建图那一次的时间（它只在建图算同义边，`HippoRAG.py:985`，
    逐题检索不用）。

    必须打**两个命名空间**：`HippoRAG.py:34` 是 `from .utils.embed_utils import retrieve_knn`，
    名字被绑进模块全局 → 只改 `embed_utils` 会静默失效（同 HGR 那个坑）。
    """
    global _ORIG_KNN
    if _ORIG_KNN is not None or not HIPPORAG_KNN_CPU:
        return
    try:
        from hipporag.utils import embed_utils
        mod = importlib.import_module("hipporag.HippoRAG")
    except Exception as e:                                   # noqa: BLE001
        print(f"[hipporag] KNN 强制 CPU 的补丁没打上（{type(e).__name__}: {e}）"
              f"→ 官方会用 GPU，vLLM 占满显存时这块会 CUDA OOM", flush=True)
        return
    _ORIG_KNN = embed_utils.retrieve_knn
    embed_utils.retrieve_knn = _force_cpu_knn
    hit = []
    if hasattr(mod, "retrieve_knn"):
        mod.retrieve_knn = _force_cpu_knn
        hit.append("hipporag.HippoRAG")
    else:
        print("[hipporag] 注意：hipporag.HippoRAG 里没找到 retrieve_knn，"
              "调用点可能换了名字 → 只 patch 了 embed_utils", flush=True)
    print(f"[hipporag] 同义边 KNN 走 CPU（HIPPORAG_KNN_CPU=1）"
          f"；补丁命名空间：embed_utils + {', '.join(hit) or '（无）'}", flush=True)


def _infer_vllm(messages):
    """HippoRAG 的 LLM 总入口：KV 背压 → 抢槽 → chat_messages → 指数退避重试。

    NER / 三元组抽取 / retrieve 阶段的 DSPyFilter 全走这里，所以并发只由
    HIPPORAG_MAX_ASYNC（槽）+ 引擎指标（背压）决定，与官方池大小解耦。
    """
    json_mode = _json_mode(messages)
    kind = _openie_kind(messages)
    for attempt in range(HIPPORAG_LLM_ATTEMPTS):
        _kv_backpressure()
        with _SLOTS:
            content, pt, ct = chat_messages(messages, json_mode=json_mode,
                                            timeout=300)
        if content is not None:
            content = _normalize_openie_payload(content, kind)   # 形状修复（HIPPORAG_FIX_JSON）
            metadata = {"prompt_tokens": pt, "completion_tokens": ct,
                        "finish_reason": "stop"}
            return (content or ""), metadata, False      # (message, metadata, cache_hit)
        time.sleep(2 ** attempt)                         # KV 满排队超时是暂时的
    raise RuntimeError(f"vLLM 调用 {HIPPORAG_LLM_ATTEMPTS} 次均失败"
                       f"（HIPPORAG_MAX_ASYNC={HIPPORAG_MAX_ASYNC}）")


def _require_injection_api(ctor, repo_name: str) -> None:
    """确认这份官方仓库支持注入 `extraction_llm` / `embedding_model`（HippoRAG 2 现行版支持）。

    早先还兼容过 2.0.0-alpha.2（没有注入口，只能换 `_get_llm_class` 工厂名），那份已经不用了；
    现在只留一条明确的自检，免得指错仓库时报一句看不懂的 TypeError。
    """
    try:
        params = set(inspect.signature(ctor.__init__).parameters)
    except (TypeError, ValueError):                          # 拿不到签名就不拦
        return
    if not ({"extraction_llm", "embedding_model"} & params):
        raise RuntimeError(
            f"[hipporag] {repo_name} 的 HippoRAG.__init__ 没有注入口"
            f"（extraction_llm/embedding_model）——这个适配器只支持 HippoRAG 2 现行版"
            f"（2.0.0a5 及以后）。请用 HIPPORAG_REPO 指到 baselines/HippoRAG-main。")


def _graph_file_of(rag) -> Path:
    """官方图文件路径（HippoRAG 2 现行版是 `graph.pickle`，在 initialize_graph 里设好）。"""
    p = getattr(rag, "_graph_pickle_filename", None)
    if p:
        return Path(p)
    return Path(getattr(rag, "working_dir", ".")) / "graph.pickle"


def _check_graph_present(rag, working_dir, build: bool) -> int:
    """核对图真的落盘且非空，返回节点数（不对就抛错）。

    为什么必须核对：官方**只在有新增块时才存图**
    （`HippoRAG.py:574-580`：`if num_new_chunks > 0: ... save_igraph()`）。
    一旦这步被跳过（目录里有旧向量库/OpenIE 结果 → 官方认为"没有新增块"），
    建图就"成功"返回、marker 照写，而图是空的 → 检索时
    `Graph/store node count mismatch: expected N, got 0`（`HippoRAG.py:1758`）→ 每题都 unknown。
    **只核对，不改官方行为。**
    """
    gpath = _graph_file_of(rag)
    if not gpath.exists() or gpath.stat().st_size == 0:
        what = "建图结束但图没落盘" if build else "索引里的图文件缺失"
        raise RuntimeError(
            f"[hipporag] {what}：{gpath}\n"
            f"  官方只在「有新增块」时保存图（HippoRAG.py:574-580）；这个目录里可能已有"
            f"旧的向量库/OpenIE 结果，于是它认为没有新增块、直接跳过存图。\n"
            f"  处理：换一个 --graph-version 重跑，或删掉 rebuild 残骸后重建：\n"
            f"    rm -f {working_dir}/graph.* {working_dir}/chunk_metadata* "
            f"{working_dir}/build_complete.marker\n"
            f"    rm -rf {working_dir}/*embedding* {working_dir}/chunk_embeddings "
            f"{working_dir}/entity_embeddings {working_dir}/fact_embeddings\n"
            f"    （llm_response_cache / openie_results_*.json 可留着省 LLM）")
    graph = getattr(rag, "graph", None)
    n = graph.vcount() if graph is not None and hasattr(graph, "vcount") else -1
    if n == 0:
        raise RuntimeError(
            f"[hipporag] 图是空的（0 节点）：{gpath}\n"
            f"  说明 OpenIE 结果里没有任何实体/三元组 → 看上面抽取失败块的数量，"
            f"或确认 CHAT_MODEL 与 vLLM 的 --served-model-name 一致")
    print(f"[hipporag] 图落盘核对通过：{gpath.name}（{n} 节点）", flush=True)
    return n


def _index_identity() -> str:
    """索引身份：**纯由 config 决定**（不含时间/随机/id），所以 build 与 load 必然同值。

    官方新版（schema_version 4）注入 embedding/LLM 时**必须**给这个值
    （`HippoRAG.py:190`：`index_identity is required when injecting ...`），
    它进索引清单（`:317` components.explicit_identity）与 OpenIE 状态指纹（`:350`），
    两处都要在 build / load 之间**逐字相同**，否则 init 就报 manifest mismatch。
    """
    return (f"factalign-rag|chunk={EXTRACT_CHUNK_TOKENS}/{EXTRACT_CHUNK_OVERLAP}"
            f"|chat={CHAT_MODEL}|embed={EMBED_MODEL_SERVED}|inject=v1")


def _make_rag(ctor, config, llm_cls, emb_cls, repo_name: str = ""):
    """造官方 HippoRAG（注入我们自己的 LLM / embedding 实现）。"""
    _require_injection_api(ctor, repo_name)
    return _build_rag(ctor, config, llm_cls(config),
                      emb_cls(config, f"VLLM/{EMBED_MODEL_SERVED}"))


def _build_rag(ctor, config, llm, emb):
    """构造官方 HippoRAG，兼容「要 / 不要 index_identity」两代版本。

    · 新版（2.0.0a5 起，服务器上就是这版）：注入 embedding/LLM 时不给就 `ValueError`
      ——报错原文 `index_identity is required when injecting an embedding model, ...`。
    · 老版 alpha：构造函数没有这个 kwarg，传了是 `TypeError: unexpected keyword argument`。
    先带、`TypeError` 再退回不带，并打印实际走的那条（换版本时一眼能看出来）。
    """
    kw = dict(global_config=config, extraction_llm=llm, embedding_model=emb)
    identity = _index_identity()
    try:
        rag = ctor(index_identity=identity, **kw)
        print(f"[hipporag] index_identity={identity}（新版官方要求，进索引清单）", flush=True)
        return rag
    except TypeError as e:
        if "index_identity" not in str(e):
            raise                      # 别的 TypeError（真 bug）别吞
        rag = ctor(**kw)
        print("[hipporag] 官方构造函数没有 index_identity 参数（老版 alpha）→ 不传", flush=True)
        return rag


def _component_classes(BaseLLM, BaseEmbeddingModel, LLMConfig):
    """在**同一个函数里**定义注入用的那两个类，返回 (LLM 类, embedding 类)。

    为什么必须集中在这里：官方更新后把「索引生产者」的类身份写进索引清单
    （`component_class_identity()` = `module + __qualname__`，state_utils.py:54-57），
    每条构造路径上算一遍、和清单里的旧值比对。原来 build/load 各自 `def` 一遍，
    qualname 就成了 `HippoRAG.build.<locals>._VLLMLLM` vs `HippoRAG.load.<locals>._VLLMLLM`
    → 建图轮和查询轮算出来不是同一个字符串 → init 直接
    `StateConsistencyError: Index schema/config mismatch ... Use a fresh save_dir`。
    放进同一个模块级函数后，两条路径的 `__module__ + __qualname__` 完全一致。
    """
    class _VLLMLLM(BaseLLM):
        def __init__(self, global_config):
            super().__init__(global_config)
            self.llm_name = global_config.llm_name
            self._init_llm_config()

        def _init_llm_config(self):
            self.llm_config = LLMConfig.from_dict({
                "llm_name": self.llm_name,
                "generate_params": {"model": self.llm_name, "temperature": 0.0},
            })

        def infer(self, messages, **kwargs):
            # OpenIE（NER/triple）需要 JSON；DSPyFilter 的 rerank 走自由文本；
            # 并发闸 + KV 背压 + 重试都在模块级 _infer_vllm 里（build/load 共用）
            return _infer_vllm(messages)

    class _BgeEmbedding(BaseEmbeddingModel):
        def __init__(self, global_config, embedding_model_name):
            super().__init__(global_config=global_config)
            self.embedding_model_name = embedding_model_name
            self.embedding_dim = EMBED_DIM
            self.embedding_type = 'float'

        def batch_encode(self, texts, instruction=None, norm=True, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            if not texts:
                return np.zeros((0, self.embedding_dim), dtype=np.float32)
            if instruction:
                # query 侧：common.llm.embed 已加 bge 查询指令 + 归一化
                vecs = np.vstack([embed(t) for t in texts])
            else:
                # passage/entity/fact 侧：批量 embed（无指令 + 归一化）
                vecs = embed_batch(texts)
            return vecs

    return _VLLMLLM, _BgeEmbedding


class HippoRAG(BaseMethod):
    name = "hipporag"

    def __init__(self):
        self._rag = None

    def _find_repo(self) -> Path:
        """选官方仓库目录：`HIPPORAG_REPO` 优先，否则 `HippoRAG` / `HippoRAG-main`。

        多候选时把每个候选和最终选择都打出来（踩过的坑：baselines/ 里多出一个目录后，
        适配器在用户不知情的情况下换了实现，"昨天还能跑今天就崩"）。
        """
        env = os.environ.get("HIPPORAG_REPO", "").strip()
        if env:
            p = Path(env)
            cands = [p if p.is_absolute() else BASELINES_DIR / p]
        else:
            cands = [BASELINES_DIR / n for n in ("HippoRAG", "HippoRAG-main", "hipporag")]
        found = [p for p in cands if p.exists()]
        if not found:
            raise RuntimeError(
                f"未找到 HippoRAG 官方仓库（找过：{', '.join(str(p) for p in cands)}）。\n"
                f"  git clone https://github.com/OSU-NLP-Group/HippoRAG.git baselines/HippoRAG-main\n"
                f"  或用 HIPPORAG_REPO=<目录名/绝对路径> 指定")
        pick = found[0]
        if len(found) > 1:
            print(f"[hipporag] baselines/ 下有多个 HippoRAG 仓库："
                  f"{', '.join(p.name for p in found)} → 用 {pick.name}"
                  f"（要换：export HIPPORAG_REPO=<目录名>）", flush=True)
        print(f"[hipporag] 官方仓库 = {pick}", flush=True)
        return pick

    def load(self, working_dir: str = "data/hipporag_corpus"):
        """只加载已有索引，直接进查询。不做任何建图/断点/重建判断。

        官方构造时自动读图文件、embedding store 自动读 parquet，
        retrieve() 内部懒触发 prepare_retrieval_objects——构造即加载。
        """
        working_dir = Path(working_dir)
        if not (working_dir / "build_complete.marker").exists():
            raise RuntimeError(
                f"[hipporag] 索引不存在：{working_dir}\n"
                f"请先跑：python run.py --method hipporag --dataset ... 建索引")
        repo = self._find_repo()
        src = repo / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        try:
            from hipporag import HippoRAG as _HippoRAG
            from hipporag.utils.config_utils import BaseConfig
            from hipporag.llm.base import BaseLLM, LLMConfig
            from hipporag.embedding_model.base import BaseEmbeddingModel
        except ImportError as e:
            raise RuntimeError(
                "HippoRAG 依赖未装：只装轻依赖即可，例如 "
                "`pip install python-igraph openai filelock tenacity`（别装它的 torch/vllm）——"
                f"{e}"
            ) from e

        # 官方两处 OpenIE 裸线程池 → 顶到 HIPPORAG_MAX_ASYNC（见文件头旋钮总表）
        _patch_official_openie()

        # 注入用的两个组件类：集中在 _component_classes() 里定义，
        # 这样 build / load 两条路径算出的类身份（进官方索引清单）完全一致
        _VLLMLLM, _BgeEmbedding = _component_classes(BaseLLM, BaseEmbeddingModel, LLMConfig)

        config = BaseConfig(
            llm_name=CHAT_MODEL,
            embedding_model_name=f"VLLM/{EMBED_MODEL_SERVED}",
            openie_mode="online",
            save_dir=str(working_dir),
            # 官方长度/预算限制（linking/retrieval/qa top_k、damping、同义边阈值、
            # OpenIE 输出上限、embedding_max_seq_len）——按官方字段表过滤后传入。
            # ⚠️ 注意 retrieval_top_k/qa_top_k 在这里是**惰性的**：query() 自己调
            # retrieve(num_to_retrieve=) 并截断，不走官方 rag_qa()。真正生效在 query()。
            **_official_cfg_kwargs(),
            **_openie_cfg_kwargs(),          # 抽取并发（老版官方没有这个字段 → 空）
        )

        self._rag = _make_rag(_HippoRAG, config, _VLLMLLM, _BgeEmbedding, repo.name)
        _check_graph_present(self._rag, working_dir, build=False)
        return self

    def build(self, corpus_contexts: list[dict], working_dir: str = "data/hipporag_corpus"):
        repo = self._find_repo()
        src = repo / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))

        try:
            from hipporag import HippoRAG as _HippoRAG
            from hipporag.utils.config_utils import BaseConfig
            from hipporag.llm.base import BaseLLM, LLMConfig
            from hipporag.embedding_model.base import BaseEmbeddingModel
        except ImportError as e:
            raise RuntimeError(
                "HippoRAG 依赖未装：只装轻依赖即可，例如 "
                "`pip install python-igraph openai filelock tenacity`（别装它的 torch/vllm）——"
                f"{e}"
            ) from e

        # 官方两处 OpenIE 裸线程池 → 顶到 HIPPORAG_MAX_ASYNC（见文件头旋钮总表）
        _patch_official_openie()
        _patch_knn_cpu()          # 同义边 KNN 走 CPU（GPU 被 vLLM 占满 → 官方会 OOM）

        # 注入用的两个组件类：集中在 _component_classes() 里定义，
        # 这样 build / load 两条路径算出的类身份（进官方索引清单）完全一致
        _VLLMLLM, _BgeEmbedding = _component_classes(BaseLLM, BaseEmbeddingModel, LLMConfig)

        working_dir = Path(working_dir)
        marker = working_dir / "build_complete.marker"
        # ★ 签名里带上「用的是哪份官方仓库」：baselines/ 里换目录就必须重建，
        #   否则会拿别的仓库/参数留下的目录凑合（这正是上一轮翻车的原因）。
        marker_sig = (f"chunk={EXTRACT_CHUNK_TOKENS}/{EXTRACT_CHUNK_OVERLAP}"
                      f"|{CHAT_MODEL}|{EMBED_MODEL_SERVED}|inject"
                      f"|repo={repo.name}")
        working_dir.mkdir(parents=True, exist_ok=True)

        # 语料不切块（一个 context = 一个整段块）：与自带方法逐块一致
        docs = []
        for c in corpus_contexts:
            full = f"{c.get('title', '')}\n{c.get('text', '')}".strip()
            for chunk_text, _cs, _ce in split_by_tokens(full, EXTRACT_CHUNK_TOKENS,
                                                        EXTRACT_CHUNK_OVERLAP):
                if chunk_text.strip():
                    docs.append(chunk_text)

        # 「状态变化 → 删目录重建」：但上一轮**没跑完**（无 marker）时，已落盘的 openie 结果是
        # 官方的增量检查点（HippoRAG.py:1021-1065），块没变就能接着跑——别白删几百万 token。
        reuse = False
        if marker.exists() or any(working_dir.glob("openie_results_ner_*.json")):
            old_sig = marker.read_text(encoding="utf-8") if marker.exists() else ""
            if old_sig != marker_sig:
                cov = _openie_coverage(working_dir, docs)
                if cov >= HIPPORAG_RESUME_COVERAGE:
                    reuse = True
                    print(f"[hipporag] 上一轮没跑完（marker: {old_sig or '无'}），但已有 OpenIE 结果"
                          f"覆盖当前 {cov:.1%} 的块 → 续跑增量，不删目录"
                          f"（已抽过的块零 LLM；图会按 json 全量重建）", flush=True)
                else:
                    print(f"[hipporag] 索引状态变化（{old_sig or '未完成'} → {marker_sig}），"
                          f"已有 OpenIE 覆盖仅 {cov:.1%} → 清除 {working_dir} 重建", flush=True)
                    shutil.rmtree(working_dir, ignore_errors=True)
                    working_dir.mkdir(parents=True, exist_ok=True)

        config = BaseConfig(
            llm_name=CHAT_MODEL,
            embedding_model_name=f"VLLM/{EMBED_MODEL_SERVED}",
            openie_mode="online",               # 用注入的 _VLLMLLM 做 OpenIE
            save_dir=str(working_dir),
            # 同 load()：官方长度/预算限制按字段表过滤传入（retrieval/qa top_k 在这
            # 里惰性，真正生效在 query()）
            **_official_cfg_kwargs(),
            **_openie_cfg_kwargs(),             # 抽取并发（老版官方没有这个字段 → 空）
        )

        self._rag = _make_rag(_HippoRAG, config, _VLLMLLM, _BgeEmbedding, repo.name)
        if reuse:
            # 续跑：图必须从（可能变多了的）openie json 全量重建——官方加边是
            # `if chunk_key not in current_graph_nodes`（a5:HippoRAG.py:905/:948），
            # 沿用旧图文件会让「上一轮没进图」的块永远补不上边。
            self._rag.global_config.force_index_from_scratch = True

        print(f"[hipporag] 建知识图（OpenIE 三元组 + PPR）-> {working_dir}（{len(docs)} 块）",
              flush=True)
        # OpenIE 抽取在 HippoRAG 自己的线程池里，没有 phase 上下文 → 强制归到 build 桶
        with _FAIL_LOCK:
            _OPENIE_FAILS["ner"].clear()
            _OPENIE_FAILS["triple"].clear()
        set_default_phase("build")
        try:
            self._rag.index(docs)
            left_keys = _repair_failed_openie(self._rag, docs)   # 失败块自动补抽
        finally:
            set_default_phase(None)
        if left_keys:
            print(f"[hipporag] 建图完成，{len(left_keys)} 块抽取失败"
                  f"（官方解析失败 → 这些块在图里没有实体/事实边，官方口径就是丢掉结构、"
                  f"文本仍留作 passage 节点）。\n"
                  f"  清单：python tools/hipporag_reextract.py --dataset <ds> --report\n"
                  f"  原因：grep 'Exception for chunk' logs/hipporag/*.log\n"
                  f"  想自动补抽：把适配器里的 HIPPORAG_RETRY_ROUNDS 改成 >0", flush=True)
        # ★ 建完必须核对「图真的落盘了」——理由见 _check_graph_present 的 docstring。
        _check_graph_present(self._rag, working_dir, build=True)
        marker.write_text(marker_sig, encoding="utf-8")
        return self

    def query(self, question: str) -> dict:
        """在已建知识图上 PPR 检索 → 答案生成统一走我们的 generate_answer。"""
        if self._rag is None:
            raise RuntimeError("请先 build() 建索引")

        # ★ 图检索耗时拆解：官方把三个**累计**计时器挂在 HippoRAG 实例上
        #   （`HippoRAG.py:770-773` 打的就是它们）→ 取本题前后的差值 = 该题的耗时构成。
        #   `rerank_time` = Recognition Memory（事实打分 + 重排）,
        #   `ppr_time` = Personalized PageRank，`all_retrieval_time` = 检索总时。
        #   用来把"图检索耗时"从总时间里拆出来（GPU 时间归因用）。
        def _timers() -> tuple:
            r = self._rag
            return (float(getattr(r, "rerank_time", 0.0) or 0.0),
                    float(getattr(r, "ppr_time", 0.0) or 0.0),
                    float(getattr(r, "all_retrieval_time", 0.0) or 0.0))

        _t0 = _timers()
        try:
            # ★ 检索深度（官方 retrieval_top_k=200）：num_to_retrieve 显式传参**覆盖**
            #   官方 config 里的 retrieval_top_k（HippoRAG.py:728-729 只在 None 时才读
            #   config），所以这里才是真正生效的位置。
            solutions = self._rag.retrieve([question],
                                           num_to_retrieve=HIPPORAG_RETRIEVAL_TOP_K)
        except Exception as e:
            # 索引状态不一致（图没落盘 / 图与向量库不匹配）**不会**逐题自愈：
            # 官方在 prepare_retrieval_objects（HippoRAG.py:1756-1759）抛
            # StateConsistencyError。老写法把它吞掉 → 300 题全 unknown、白烧一整轮，
            # 还不报错（上次就是这么亏的）。这里第一次就直接中止。
            if type(e).__name__ == "StateConsistencyError":
                raise RuntimeError(
                    f"[hipporag] 索引状态不一致，检索无法进行：{e}\n"
                    f"  图与向量库对不上（多半是建图那次图没落盘）→ 本轮每道题都会是 "
                    f"unknown，已中止，不写假结果。\n"
                    f"  处理：重跑建图（换 --graph-version，或删掉该索引目录后重建）。"
                ) from e
            solutions = None
        _t1 = _timers()
        timing = {"rerank_seconds": round(_t1[0] - _t0[0], 3),
                  "ppr_seconds": round(_t1[1] - _t0[1], 3),
                  "retrieval_seconds": round(_t1[2] - _t0[2], 3)}
        if not solutions:
            return {"answer": "unknown", "evidence": [], "evidence_locs": [],
                    "entities": [], "paths": [], **timing}

        sol = solutions[0]
        # ★ 证据预算（官方 qa_top_k=5）：只有这前 N 篇进 generate_answer。
        #   与官方 rag_qa() 的 docs[:qa_top_k]（HippoRAG.py:1130）同口径。
        docs = list(sol.docs or [])[:HIPPORAG_QA_TOP_K]
        if not docs:
            return {"answer": "unknown", "evidence": [], "evidence_locs": [],
                    "entities": [], "paths": [], **timing}

        # ★ 每条证据带上分数/排名/来源元数据。`common/trace.py:184-185` 只剔除
        #   text / window_text / summary，**其余键原样保留**进每题 trace 的 chunks
        #   → 分数和来源块就能事后做"检索质量 vs 预算 / top-k 曲线"。
        #   分数来自官方 `sol.doc_scores`，与 `sol.docs` 同序（HippoRAG.py:790）。
        #   ⚠️ `doc_metadata` 里可能带 `content`（整段正文）—— 直接塞会**在 trace 里
        #      把同一段正文存两遍**（实测能多 30~40% 体积），所以只留「短字段」。
        # ★ `doc_scores` 是 **numpy 数组**（官方 `HippoRAG.py:790` 的
        #   `scores=np.asarray(sorted_doc_scores[...])`）→ **绝不能写 `x or []`**：
        #   `or` 会对数组求布尔值，元素数 >1 时抛
        #   `ValueError: The truth value of an array with more than one element is ambiguous`。
        #   实测：n=1 侥幸通过、n≥2 必崩 → 表现为"部分题崩、部分题正常"，500 题全废。
        #   （曾是一个真 bug：官方检索明明成功了，崩在**我们取分数这一行**，堆栈看起来像官方的问题。）
        _sc = getattr(sol, "doc_scores", None)
        _scores = [float(x) for x in _sc] if _sc is not None else []
        _mt = getattr(sol, "doc_metadata", None)
        _metas = list(_mt) if _mt is not None else []
        ev = []
        for _i, _d in enumerate(docs):
            _item = {"text": _d, "rank": _i}
            if _i < len(_scores):
                try:
                    _item["score"] = round(float(_scores[_i]), 6)
                except Exception:                              # noqa: BLE001
                    pass
            if _i < len(_metas) and isinstance(_metas[_i], dict):
                for _k, _v in _metas[_i].items():
                    if isinstance(_v, (int, float, bool)) or _v is None:
                        _item.setdefault(_k, _v)
                    elif isinstance(_v, str) and len(_v) <= 200:
                        _item.setdefault(_k, _v)
            ev.append(_item)

        # ★ 实体口径：官方 `sol.graph_seeds` = 重排后**当作 PPR 种子的那批事实**
        #   （`HippoRAG.py:762` 把 `top_k_facts` 传进 `_build_retrieval_result` 的
        #   `graph_seeds` 形参），每个元素是 `Triple = Tuple[str, str, str]`
        #   （`utils/typing.py:3`）= (主语, 关系, 宾语)。
        #   取「主语 + 宾语」去重 = 这题实际引用到的实体，与 proprag 的 entities 同义
        #   （`run.py:552` 注释写的是"题目中提取的实体"）。
        #   ⚠️ 此前这里硬编码 `[]` → `run.py:520` 把它交给 trace，每题实体 JSON 全空，
        #      看起来像"检索没命中实体"，其实只是没往外拿。
        ent_names, paths = [], []
        for f in (getattr(sol, "graph_seeds", None) or []):
            if isinstance(f, (list, tuple)) and len(f) >= 3:
                ent_names += [str(f[0]).strip(), str(f[2]).strip()]
                paths.append([str(f[0]).strip(), str(f[1]).strip(), str(f[2]).strip()])
            else:
                ent_names.append(str(f).strip())
        entities = list(dict.fromkeys(e for e in ent_names if e))

        # ★ 生成走**项目统一生成口径**（`generate_answer` → `prompts/rag_prompt.txt`），
        #   与 standardrag / factalign / proprag / s2grag / … 同口径：生成侧固定，
        #   方法之间的差异只来自**检索**（本项目审的就是 factalign 的检索）。
        #   ⚠️ 官方 `qa()`（HippoRAG.py:1106）**有意不走**：它套 `rag_qa_{dataset}`
        #      模板，而官方**只带 `rag_qa_musique.py` 一个**，对 hotpotqa / 2wiki /
        #      popqa / triviaqa 全部回退到 musique 模板（官方自己打 warning，
        #      `HippoRAG.py:1140-1144`）—— 那是"官方只调过 musique"，不是"每个数据集
        #      都有对应的生成器"，套上去等于给 4/5 个数据集用错的一 shot demo。
        #      该偏差记在 PARAMS_ALIGNED.md（生成侧统一是有意选择）。
        answer = generate_answer(question, [{"text": d} for d in docs], dataset=getattr(self, "dataset", ""))
        return {"answer": answer, "evidence": ev,
                "evidence_locs": [], "entities": entities, "paths": paths, **timing}

    def run(self, question, contexts):
        """per-question 路径不建议（建图太重），统一走语料级 build/query。"""
        raise NotImplementedError(
            "HippoRAG 是语料级方法：先 build(全部上下文) 一次，再逐题 query。"
        )

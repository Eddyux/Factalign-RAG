#!/usr/bin/env python
"""换/调 embedding 模型时的自检：**配置说的** vs **服务端实际给的**。

为什么需要它：换 embedder 时最容易出错的三处都**不会报错**——
  ① `EMBED_DIM` 与实际返回维度不一致 → 向量以错误维度进 numpy/faiss，算出来是垃圾；
  ② `EMBED_MODEL_SERVED` 与 vLLM `--served-model-name` 不一致 → 有时 404，有时悄悄用了别的模型；
  ③ 查询前缀格式不对（bge 的指令句给了 Qwen3）→ 检索质量掉但日志一片正常。
本脚本把这三处一次性打出来，并在能探到服务端时直接对比。

用法：
    python tools/check_embedder.py                          # 用代码默认值
    python tools/check_embedder.py --config configs/default.json
    python tools/check_embedder.py --config configs/embedding-qwen3-8b.json
    python tools/check_embedder.py --embed                # 真的发一次 embedding（3 条文本）
    python tools/check_embedder.py --embed --json         # 机器可读
退出码：0=配置自洽；1=服务端维度与配置不一致；2=服务端不可达（配置本身仍打印）。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _prescan_config(argv):
    """在 import config 之前读出 --config（与 run.py 同一套做法）。"""
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--config="):
            return a.split("=", 1)[1]
    return None


CFG = _prescan_config(sys.argv[1:])
if CFG:
    from common.config_file import apply_config_file
    apply_config_file(CFG, verbose=False)

import config as C                                              # noqa: E402
from common.llm import tokenizer_kind                           # noqa: E402


def _probe_dim(timeout=20):
    """调一次 embeddings 端点看真实维度；失败返回 (None, 原因)。"""
    import requests
    url = f"{C.VLLM_EMBED_BASE_URL}/embeddings"
    try:
        r = requests.post(url, json={"model": C.EMBED_MODEL_SERVED, "input": ["dim probe"]},
                          timeout=timeout)
    except Exception as e:                                      # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    try:
        return len(r.json()["data"][0]["embedding"]), None
    except Exception as e:                                      # noqa: BLE001
        return None, f"响应结构异常（{type(e).__name__}: {e}）"


def _probe_models(timeout=10):
    """读 /v1/models：served 名 + max_model_len（embedding 实例通常也报 max_model_len）。"""
    import requests
    base = C.VLLM_EMBED_BASE_URL.rstrip("/")
    try:
        r = requests.get(f"{base}/models", timeout=timeout)
        r.raise_for_status()
        d = r.json().get("data") or []
    except Exception as e:                                      # noqa: BLE001
        return {"available": False, "error": f"{type(e).__name__}: {e}"}
    ids = [x.get("id") for x in d]
    out = {"available": True, "ids": ids}
    if d:
        out["max_model_len"] = d[0].get("max_model_len")
    return out


def _warnings(dim_actual, dim_err, models):
    w = []
    if dim_err:
        w.append(f"服务端 embedding 探测失败：{dim_err}\n"
                 f"   → 确认 {C.VLLM_EMBED_BASE_URL} 上有 `--task embed` 的实例，"
                 f"且 served-model-name = {C.EMBED_MODEL_SERVED!r}")
    elif dim_actual != C.EMBED_DIM:
        w.append(f"⚠️ 维度不一致：服务端 {dim_actual} 维 vs 配置 EMBED_DIM={C.EMBED_DIM}\n"
                 f"   → 把 embed_dim 改成 {dim_actual}，然后**重建所有向量**（新 --graph-version）")
    if models.get("available") and C.EMBED_MODEL_SERVED not in (models.get("ids") or []):
        w.append(f"⚠️ served-model-name 不匹配：配置 {C.EMBED_MODEL_SERVED!r}，"
                 f"服务端提供 {models.get('ids')}\n"
                 f"   → 改 embed_model_served，或核对 vLLM 的 --served-model-name")
    if C.EMBED_MAX_TOKENS > C.EMBED_REQUEST_TOKENS:
        w.append(f"⚠️ embed_max_tokens({C.EMBED_MAX_TOKENS}) > "
                 f"embed_request_tokens({C.EMBED_REQUEST_TOKENS})：单条就超预算，"
                 f"请求会被拆成一条一组（能跑但慢）")
    if C.EXTRACT_CHUNK_TOKENS > C.EMBED_MAX_TOKENS:
        w.append(f"⚠️ 切块 {C.EXTRACT_CHUNK_TOKENS} token > 单条嵌入上限 "
                 f"{C.EMBED_MAX_TOKENS}：语料块会被截断后再嵌入（口径不一致）")
    if C.EMBED_QUERY_FORMAT == "bge" and "qwen" in str(C.EMBED_MODEL).lower():
        w.append("⚠️ 模型名像 Qwen3-Embedding，但查询前缀还是 bge 的英文指令句"
                 "（应为 embed_query_format=\"qwen3\"，否则检索质量下降且无报错）")
    if C.EMBED_QUERY_FORMAT == "qwen3" and "qwen" not in str(C.EMBED_MODEL).lower():
        w.append(f"⚠️ 查询前缀设成 qwen3，但模型是 {C.EMBED_MODEL}"
                 f"（确认这是你想要的非对称格式）")
    if C.CHUNK_TOKENIZER != C.EMBED_TOKENIZER:
        w.append(f"i 切块尺子({C.CHUNK_TOKENIZER}) ≠ embedding 尺子({C.EMBED_TOKENIZER})："
                 f"这是有意的官方对账模式；两种口径的建图结果不可直接比较")

    # 查询前缀与官方模板逐字符比对（离线可查，不需要服务端）。
    # 这里故意用一个**独立的**官方模板字面量当基准：将来谁改了 llm.py 的 f-string，
    # 这条会失败，而不是静默变成"和官方差一个空格"。
    from common.llm import query_prefix
    if not C.EMBED_QUERY_TEMPLATE:
        fmt = (C.EMBED_QUERY_FORMAT or "").lower()
        probe = "probe"
        got = query_prefix(probe)
        if fmt == "qwen3":
            want = f"Instruct: {C.EMBED_QUERY_TASK}\nQuery:{probe}"   # 官方：Query: 后无空格
        elif fmt == "bge":
            want = C.EMBED_QUERY_INSTRUCTION + probe                  # 官方 bge：指令句直接前置
        else:
            want = got                                                # none：不加前缀，无从比对
        if got != want:
            w.append(f"✗ {fmt} 查询前缀与官方模板不一致：\n"
                     f"      实际 {got!r}\n      应为 {want!r}\n"
                     f"      前缀错一个字符 → tokenizer 切出不同 token，与官方查询口径不符")
    return w


def _embed_smoke():
    """真的嵌入 3 条文本：验证端点可用 + 维度 + 前缀确实生效（query 与 doc 向量应不同）。"""
    import numpy as np
    from common.llm import embed, embed_batch
    q = "Who wrote the novel that inspired the film Blade Runner?"
    try:
        vq = np.asarray(embed(q), dtype=np.float32)
        vd = np.asarray(embed_batch([q])[0], dtype=np.float32)
    except Exception as e:                                      # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    cos = float(vq @ vd)
    return {"ok": True, "dim": int(vq.shape[0]), "norm": float(np.linalg.norm(vq)),
            "cos_query_vs_doc": round(cos, 6),
            "prefix_effective": (C.EMBED_QUERY_FORMAT != "none"
                                 and abs(cos - 1.0) > 1e-6)}


def main() -> int:
    ap = argparse.ArgumentParser(description="embedding 配置自检（配置 vs 服务端）")
    ap.add_argument("--config", default=CFG, help="configs/*.json（只看它注入了什么）")
    ap.add_argument("--embed", action="store_true", help="真的发一次 embedding 请求验证")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    args = ap.parse_args()

    dim_actual, dim_err = _probe_dim()
    models = _probe_models()
    warns = _warnings(dim_actual, dim_err, models)
    smoke = _embed_smoke() if args.embed else None

    cfg = {
        "config_file": args.config,
        "embed_model": str(C.EMBED_MODEL),
        "embed_model_served": C.EMBED_MODEL_SERVED,
        "embed_dim": C.EMBED_DIM,
        "embed_max_tokens": C.EMBED_MAX_TOKENS,
        "embed_tokenizer": C.EMBED_TOKENIZER,
        "chunk_tokenizer": C.CHUNK_TOKENIZER,
        "embed_request_tokens": C.EMBED_REQUEST_TOKENS,
        "embed_query_format": C.EMBED_QUERY_FORMAT,
        "embed_query_task": C.EMBED_QUERY_TASK,
        "embed_query_template": C.EMBED_QUERY_TEMPLATE,
        "embed_doc_prefix": C.EMBED_DOC_PREFIX,
        "extract_chunk_tokens": C.EXTRACT_CHUNK_TOKENS,
        "extract_chunk_overlap": C.EXTRACT_CHUNK_OVERLAP,
        "vllm_embed_base_url": C.VLLM_EMBED_BASE_URL,
    }
    server = {"embedding_dim_actual": dim_actual, "embedding_probe_error": dim_err,
              "models": models}
    out = {"config": cfg, "server": server, "warnings": warns, "smoke": smoke,
           "tokenizer_in_use": tokenizer_kind()}

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("=" * 72)
        print(f"配置文件        : {args.config or '（未指定，用代码默认值）'}")
        print(f"embedding 模型  : {C.EMBED_MODEL}")
        print(f"served 名       : {C.EMBED_MODEL_SERVED}  @ {C.VLLM_EMBED_BASE_URL}")
        print(f"维度            : 配置 {C.EMBED_DIM}   服务端 "
              + (f"{dim_actual}" if dim_actual else f"未知（{dim_err}）"))
        print(f"单条截断        : {C.EMBED_MAX_TOKENS} token（尺子 {C.EMBED_TOKENIZER}，"
              f"实际加载 {tokenizer_kind()}）")
        print(f"切块            : {C.EXTRACT_CHUNK_TOKENS}/{C.EXTRACT_CHUNK_OVERLAP} token"
              f"（尺子 {C.CHUNK_TOKENIZER}）")
        print(f"单请求预算      : {C.EMBED_REQUEST_TOKENS} token"
              + (f"   引擎 max_model_len = {models.get('max_model_len')}"
                 if models.get("max_model_len") else ""))
        print(f"查询前缀        : {C.EMBED_QUERY_FORMAT}"
              + (f"（任务：{C.EMBED_QUERY_TASK}）" if C.EMBED_QUERY_FORMAT == "qwen3" else "")
              + (f"   模板：{C.EMBED_QUERY_TEMPLATE}" if C.EMBED_QUERY_TEMPLATE else ""))
        if C.EMBED_DOC_PREFIX:
            print(f"文档前缀        : {C.EMBED_DOC_PREFIX!r}")
        if smoke:
            print(f"实测嵌入        : {smoke}")
        print("-" * 72)
        if warns:
            for w in warns:
                print(w)
        else:
            print("✅ 配置自洽，且与服务端一致")
        print("=" * 72)

    if dim_err:
        return 2
    if dim_actual is not None and dim_actual != C.EMBED_DIM:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

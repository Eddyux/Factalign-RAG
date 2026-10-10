"""探测 vLLM 服务端配置并落盘（SIGMOD 成本口径的关键证据）。

为什么要单独做（P0-1）：
  成本/时间对比里最容易被审稿人质疑的一点是 **prefix caching**。vLLM 默认开启它，
  而 IRCoT 这类方法每题 10 次调用、前 8k token 完全相同 —— 命中前缀缓存后 prefill
  几乎不花时间。**开与不开，prefill token 和 GPU 时间能差好几倍**，而"统一了语料/
  reader/预算"的对比必须说清这一点。原来日志只记 `max_model_len`（且常常取不到），
  没有任何一处记录服务端到底怎么起的。

本模块在 run 开始前探一次，落 `results/server_config.json`：
  · `GET /v1/models`  → 完整元数据（`max_model_len`、模型 id/root 等）
  · `GET /version`    → vLLM 版本（比 pip 包版本更可靠：可能是容器里的 nightly）
  · `GET /metrics`    → 引擎容量 + 计数器（KV 总块数、prefix cache 命中计数…）
  · 文本日志里的 `[cfg] vLLM max_model_len = …` 也来自这里的探测结果

同时给出**明确结论**（`prefix_caching` = on/off/unknown + 判据），因为审稿人要的是
"你到底开没开"，而不是一堆原始指标。探测失败不抛异常：只写 `available: false`，
并提示"成本结论需注明未验证服务端配置"。
"""

import json
import time
from pathlib import Path

import requests

from config import (VLLM_BASE_URL, VLLM_EMBED_BASE_URL, CHAT_MODEL, EMBED_MODEL_SERVED,
                    EMBED_MAX_TOKENS, EMBED_REQUEST_TOKENS, EMBED_SERVER_BATCH_TOKENS,
                    RESULTS_DIR)


def _root_of(base_url: str) -> str:
    """http://host:port/v1 → http://host:port（/version 与 /metrics 挂在根路径）。"""
    return base_url.rsplit("/v1", 1)[0] if "/v1" in base_url else base_url


def _get(url: str, timeout: float = 5.0):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        return r, None
    except Exception as e:                          # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


# ── 指标解析：只挑与「可比性/缓存/容量」相关的计数器 ──
_METRIC_PATTERNS = {
    "kv_cache_usage_perc": ("gpu_cache_usage_perc", "kv_cache_usage_perc"),
    "prefix_cache_hit_rate": ("gpu_prefix_cache_hit_rate", "prefix_cache_hit_rate"),
    "prefix_cache_queries": ("gpu_prefix_cache_queries", "prefix_cache_queries"),
    "prefix_cache_hits": ("gpu_prefix_cache_hits", "prefix_cache_hits"),
    "num_requests_running": ("num_requests_running",),
    "num_requests_waiting": ("num_requests_waiting",),
    "prompt_tokens_total": ("prompt_tokens_total",),
    "generation_tokens_total": ("generation_tokens_total",),
    "request_success_total": ("request_success_total",),
    "request_prefill_time_seconds": ("request_prefill_time_seconds",),
    "request_decode_time_seconds": ("request_decode_time_seconds",),
    "gpu_cache_total_blocks": ("gpu_cache_total_blocks", "cache_config_info"),
    "gpu_cache_usage_blocks": ("gpu_cache_usage_blocks",),
    "preemption_total": ("preemption",),
    "num_gpu_blocks": ("num_gpu_blocks",),
}


def parse_engine_metrics(text: str) -> dict:
    """从 /metrics 文本里抽出我们关心的计数器（同名多标签则求和）。"""
    out: dict = {}
    if not text:
        return out
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name_part, _, val_part = line.rpartition(" ")
        try:
            val = float(val_part)
        except ValueError:
            continue
        for key, pats in _METRIC_PATTERNS.items():
            if any(p in name_part for p in pats):
                out[key] = out.get(key, 0.0) + val
                break
    return out


def probe_server(base_url: str | None = None, timeout: float = 5.0) -> dict:
    """探一个 vLLM 实例（chat 或 embed 都适用）。失败不抛异常。"""
    base = base_url or VLLM_BASE_URL
    root = _root_of(base)
    info: dict = {"base_url": base}

    r, err = _get(f"{base}/models", timeout)
    if r is not None:
        try:
            data = r.json().get("data") or []
            info["models"] = data
            if data:
                m0 = data[0]
                info["served_model"] = m0.get("id")
                info["max_model_len"] = m0.get("max_model_len")
                info["model_root"] = m0.get("root")
                info["owned_by"] = m0.get("owned_by")
        except Exception as e:                      # noqa: BLE001
            info["models_error"] = f"{type(e).__name__}: {e}"
    else:
        info["models_error"] = err

    r, err = _get(f"{root}/version", timeout)
    info["vllm_version"] = (r.json() or {}).get("version") if r is not None else None
    if r is None:
        info["version_error"] = err

    r, err = _get(f"{root}/metrics", timeout)
    if r is not None:
        info["engine_metrics"] = parse_engine_metrics(r.text)
    else:
        info["metrics_error"] = err

    info["available"] = bool(info.get("models") or info.get("engine_metrics"))
    return info


def _cache_verdict(info: dict) -> tuple[str, str]:
    """判断 prefix caching 开没开：优先看命中计数器，其次看指标里有没有相关族。

    ⚠️ 计数器在 `info["engine_metrics"]` **里面**（不是顶层）——早期版本在顶层找，
    结果永远判成 unknown（等于没探）。
    """
    m = info.get("engine_metrics") or {}
    queries = m.get("prefix_cache_queries")
    hits = m.get("prefix_cache_hits")
    if queries or hits:
        q, h = float(queries or 0.0), float(hits or 0.0)
        rate = (h / q) if q else 0.0
        return "on", (f"引擎暴露 prefix cache 计数器：queries={q:.0f} hits={h:.0f} "
                      f"(采样时点命中率 {rate:.1%})；vLLM 默认开启 → 本次对比在该配置下成立")
    if m.get("prefix_cache_hit_rate") is not None:
        return "on", (f"引擎暴露 prefix cache 命中率指标 "
                      f"({m['prefix_cache_hit_rate']})；vLLM 默认开启")
    # 没有计数器族：拿不到就无法断定（不要瞎猜成 off）
    if not info.get("available"):
        return "unknown", "服务端探测失败，无法判断 prefix caching（成本结论需注明未验证）"
    return "unknown", ("引擎未暴露 prefix cache 计数器（可能是旧版 vLLM 或已关闭）；"
                       "无法从指标断定 → 请在论文/artifact 里注明实际启动参数")


def server_snapshot(chat_base: str | None = None, embed_base: str | None = None) -> dict:
    """chat + embed 两个实例的快照 + 结论，落盘用。"""
    chat = probe_server(chat_base or VLLM_BASE_URL)
    embed = probe_server(embed_base or VLLM_EMBED_BASE_URL)
    verdict, why = _cache_verdict(chat)
    # 交叉检查：服务端实际 served model 与配置里写的是不是同一个
    # （换过模型但忘了改 CHAT_MODEL 是常见坑，会让两次 run 的图/结果不可比）
    warnings: list[str] = []
    served = chat.get("served_model")
    if served and CHAT_MODEL and served != CHAT_MODEL:
        warnings.append(f"⚠ 服务端 chat 实际是 {served!r}，但 CHAT_MODEL={CHAT_MODEL!r}"
                        f" → 换模型请开新 --graph-version")
    served_emb = embed.get("served_model")
    if served_emb and EMBED_MODEL_SERVED and served_emb != EMBED_MODEL_SERVED:
        warnings.append(f"⚠ 服务端 embedding 实际是 {served_emb!r}，"
                        f"但 EMBED_MODEL_SERVED={EMBED_MODEL_SERVED!r}")
    # ★ embedding 侧两个数必须与服务端对得上。**两个是不同的引擎参数，别混**：
    #   ① EMBED_MAX_TOKENS      ←→ 引擎 `--max-model-len`（**单条**序列上限）
    #      超了 → vLLM 在 API 层直接 400。这个能从 /v1/models 的 max_model_len 探到，硬报。
    #   ② EMBED_REQUEST_TOKENS  ←→ 引擎 `--max-num-batched-tokens`（**一批**的 token 总量）
    #      超了 → 既不报错也不执行，**永久静默挂住**（vllm#41969）。
    #      ⚠️ 这个值 vLLM **不上报**，所以**不能**拿 max_model_len 当代理去猜
    #      （两者常差很多，如 batch=8192 vs model_len=2048）→ 只用显式声明的
    #      `EMBED_SERVER_BATCH_TOKENS` 比；没声明就只提示、不报警，避免误报。
    emb_len = embed.get("max_model_len")
    if emb_len and EMBED_MAX_TOKENS > emb_len:
        warnings.append(
            f"⚠ EMBED_MAX_TOKENS={EMBED_MAX_TOKENS} > 服务端 embedding "
            f"max_model_len={emb_len} → 单条超长文本会被 vLLM 直接 400。"
            f"调小 EMBED_MAX_TOKENS，或把服务端 --max-model-len 提到 ≥{EMBED_MAX_TOKENS}")
    if EMBED_SERVER_BATCH_TOKENS:
        if EMBED_REQUEST_TOKENS > EMBED_SERVER_BATCH_TOKENS:
            warnings.append(
                f"⚠ EMBED_REQUEST_TOKENS={EMBED_REQUEST_TOKENS} > 你声明的服务端 "
                f"--max-num-batched-tokens={EMBED_SERVER_BATCH_TOKENS} → 单请求会**静默挂住**"
                f"（不报错、不执行）。降到 ≤{EMBED_SERVER_BATCH_TOKENS}，或把引擎该参数调大")
    else:
        warnings.append(
            f"ℹ EMBED_REQUEST_TOKENS={EMBED_REQUEST_TOKENS} 无法核对：vLLM 不上报 "
            f"--max-num-batched-tokens。请自行确认它 ≥{EMBED_REQUEST_TOKENS}"
            f"（超了会静默挂住），确认后用 `EMBED_SERVER_BATCH_TOKENS=<值>` 声明以启用自动校验")
    if not chat.get("available"):
        warnings.append("⚠ chat 服务端探测失败 → 表格/图里必须注明服务端配置未验证")
    snap = {
        "probed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chat": {**chat, "configured_model": CHAT_MODEL},
        "embedding": {**embed, "configured_model": EMBED_MODEL_SERVED},
        "prefix_caching": verdict,           # on / off / unknown
        "prefix_caching_evidence": why,
        "warnings": warnings,
        # 客户端实际生效的口径（与上面服务端默认值分开记，避免混为一谈）
        "client_decoding": {
            "temperature": 0.0,
            "thinking": False,
            "note": "客户端只显式设 temperature/关思考；top_p/top_k/seed 等用服务端默认值"
                    "（见上方 engine_metrics 与启动参数）",
        },
        "tokenizer": _tokenizer_status(),
        "notes": [
            "max_model_len 用服务端返回值（不是写死常量）；超长请求会被客户端按它截断",
            "prefix caching 会显著改变 prefill 耗时与 GPU busy 秒 → 对比实验必须在同一设置下跑",
        ],
    }
    return snap


def _tokenizer_status() -> dict:
    """切块用的 bge tokenizer 是否真的可用（不可用会**退化到字符级切块**，改变语料单元）。"""
    try:
        from common.llm import _get_tokenizer, EMBED_MODEL, tokenizer_failure
        _tok_err = tokenizer_failure()
        tok = _get_tokenizer()
        if tok is None:
            return {"available": False, "model": str(EMBED_MODEL),
                    "error": _tok_err or "（未记录；见 common.llm._TOKENIZER_FAILURES）",
                    "impact": "⚠ 切块退化到字符级（1 token≈4 字符）→ chunk 数与别的 run 不可比"}
        return {"available": True, "model": str(EMBED_MODEL)}
    except Exception as e:                          # noqa: BLE001
        return {"available": False, "error": f"{type(e).__name__}: {e}",
                "impact": "⚠ 切块可能退化为字符级 → chunk 数与其他 run 不可比"}


def write_server_config(snapshot: dict | None = None, path: Path | None = None) -> Path | None:
    """落盘 results/server_config.json（失败只警告，不影响实验）。"""
    try:
        snap = snapshot or server_snapshot()
        p = Path(path or (RESULTS_DIR / "server_config.json"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        return p
    except Exception as e:                          # noqa: BLE001
        print(f"[server] ⚠ 服务端配置落盘失败（不影响实验）：{e}", flush=True)
        return None


def summary_lines(snap: dict) -> list[str]:
    """给日志头用的几行（数据来自探测结果，不再是 None）。"""
    chat, emb = snap.get("chat", {}), snap.get("embedding", {})
    L = []
    L.append(f"[cfg] vLLM chat = {chat.get('served_model') or '?'} "
             f"v{chat.get('vllm_version') or '?'} max_model_len={chat.get('max_model_len')}")
    L.append(f"[cfg] vLLM embed = {emb.get('served_model') or '?'} "
             f"max_model_len={emb.get('max_model_len')}")
    L.append(f"[cfg] prefix caching = {snap.get('prefix_caching')} "
             f"（{snap.get('prefix_caching_evidence')}）")
    tk = snap.get("tokenizer") or {}
    L.append(f"[cfg] 切块 tokenizer = {'可用' if tk.get('available') else '不可用⚠'} "
             f"{tk.get('model', '')}")
    if not tk.get("available"):
        L.append(f"[cfg] ⚠ {tk.get('impact', '切块可能退化，chunk 数与其他 run 不可比')}")
    for w in (snap.get("warnings") or []):
        L.append(f"[cfg] {w}")
    return L


def summary_block(snap: dict) -> dict:
    """给 results/summary.jsonl 用的精简块（审稿人从一行汇总就能看到成本口径）。"""
    if not snap:
        return {"available": False}
    chat = snap.get("chat") or {}
    return {
        "chat_served_model": chat.get("served_model"),
        "chat_max_model_len": chat.get("max_model_len"),
        "vllm_version": chat.get("vllm_version"),
        "prefix_caching": snap.get("prefix_caching"),
        "tokenizer_ok": (snap.get("tokenizer") or {}).get("available"),
        "warnings": snap.get("warnings") or [],
    }


if __name__ == "__main__":                          # 手动探一次
    import sys
    snap = server_snapshot()
    print(json.dumps(snap, ensure_ascii=False, indent=2))
    p = write_server_config(snap)
    if p:
        print(f"\n→ {p}", file=sys.stderr)

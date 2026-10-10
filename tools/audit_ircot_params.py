#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ircot 的「参数是不是过大」实测脚本（在**服务器**上跑，本地没 GPU/vLLM）。

回答只能用真实数据回答的三个问题：
  ① 每轮 thought 到底生成了多少 token？`IRCOT_MAX_NEW_TOKENS=300` 是"天花板"还是"实际用量"？
     → 看 `ct` 分布。**若大量贴着 300，说明 stop=["\\n"] 没生效**（模型不吐换行），
       那 300 就是白烧 decode；否则 300 只是个安全网，实际早被 stop 截断。
  ② 每轮 prompt 到底多大？8000 预算有没有把 few-shot 示例删光？
     → 看 `pt`；顺带看 user 消息里还有没有 `answer is`（官方 codex prompt 的 thought
       示范句式）。示例被删光 = 模型看不到格式示范，质量会掉，而日志不会报警。
  ③ 每轮耗时多少？`IRCOT_LLM_TIMEOUT=300` 是不是形同虚设？
     → 看 `seconds` 分布 / 最大值。

用法（服务器）：
    python tools/audit_ircot_params.py                       # 自动扫 logs/ircot/*.llm_calls.jsonl
    python tools/audit_ircot_params.py 某个.llm_calls.jsonl  # 指定文件（可多个）

字段口径（`common/llm.py:_call_log_out`）：顶层 `pt`/`ct`/`seconds`/`role`/`phase`，
以及 `req.{max_tokens,timeout,stop,flatten}` —— 后四个是**这次调用真正发出去的值**，
所以这个脚本同时也在验证"配置里的旋钮到底有没有走到线上"。
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = min(len(ys) - 1, max(0, int(round(p / 100.0 * (len(ys) - 1)))))
    return ys[k]


def _fmt(xs: list[float]) -> str:
    if not xs:
        return "(无)"
    return (f"mean={statistics.mean(xs):.1f}  median={statistics.median(xs):.0f}  "
            f"p95={_pct(xs, 95):.0f}  max={max(xs):.0f}")


def main(argv: list[str]) -> int:
    files = [Path(a) for a in argv[1:]]
    if not files:
        files = sorted((ROOT / "logs" / "ircot").glob("*.llm_calls.jsonl"))
    if not files:
        print("[x] 没找到 logs/ircot/*.llm_calls.jsonl —— 本机没跑过 ircot。"
              "请在服务器上跑，或把文件路径当参数传进来。")
        return 2

    ct: list[float] = []
    pt: list[float] = []
    sec: list[float] = []
    max_tok_seen: Counter = Counter()
    timeout_seen: Counter = Counter()
    stop_seen: Counter = Counter()
    n = 0
    n_err = 0
    n_ct0 = 0
    n_no_demo = 0
    n_chat = 0
    for f in files:
        print(f"── {f}  ({f.stat().st_size / 1e6:.1f} MB)")
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("kind") not in (None, "chat"):
                    continue
                n += 1
                c = float(rec.get("ct") or 0)
                p = float(rec.get("pt") or 0)
                s = float(rec.get("seconds") or 0)
                ct.append(c)
                pt.append(p)
                sec.append(s)
                if rec.get("error"):
                    n_err += 1
                if c == 0:
                    n_ct0 += 1
                req = rec.get("req") or {}
                max_tok_seen[str(req.get("max_tokens"))] += 1
                timeout_seen[str(req.get("timeout"))] += 1
                stop_seen[json.dumps(req.get("stop"), ensure_ascii=False)] += 1
                msgs = rec.get("messages") or []
                users = [m for m in msgs if m.get("role") == "user"]
                if users:
                    n_chat += 1
                    txt = users[0].get("content") or ""
                    if "answer is" not in txt.lower():
                        n_no_demo += 1

    print(f"\n聊天调用数 = {n}")
    if not n:
        return 1

    print("\n① 真正发出去的生成上限 / 超时 / stop（来自每条记录的 req）")
    print(f"   max_tokens : {dict(max_tok_seen)}   ← 期望 300（IRCOT_MAX_NEW_TOKENS）")
    print(f"   timeout    : {dict(timeout_seen)}   ← 期望 300（IRCOT_LLM_TIMEOUT）")
    print(f"   stop       : {dict(stop_seen)}   ← 期望 [\"\\n\"]（IRCOT_STOP=1）")

    print("\n② completion_tokens（每轮 thought 实际生成长度）")
    print(f"   {_fmt(ct)}")
    near = sum(1 for c in ct if c >= 290)
    print(f"   ≥290（贴着 300，疑似 stop 失效 / 模型不吐换行）= {near}/{n} = "
          f"{100.0 * near / n:.1f}%")
    print(f"   ==0（失败或空输出）= {n_ct0}/{n} = {100.0 * n_ct0 / n:.1f}%"
          f"；带 error 字段 = {n_err}/{n} = {100.0 * n_err / n:.1f}%")

    print("\n③ prompt_tokens（每轮上文 + few-shot 示例）")
    print(f"   {_fmt(pt)}")
    print("   预算：IRCOT_PROMPT_BUDGET=8000，另留 IRCOT_EST_GEN=300 给生成")
    if n_chat:
        print(f"   user 里**看不到** 'answer is' 示范句式 = {n_no_demo}/{n_chat} = "
              f"{100.0 * n_no_demo / n_chat:.1f}%  ← 高了说明 8000 预算把示例删光了"
              f"（`fit_ircot_prompt` 从头部删示例，日志不报警）")

    print("\n④ seconds（单次 LLM 调用墙钟）")
    print(f"   {_fmt(sec)}")
    slow = sum(1 for s in sec if s >= 60)
    print(f"   ≥60s = {slow}/{n} = {100.0 * slow / n:.1f}%（300s 超时只有在服务器卡死时"
          f"才会被摸到；真被摸到 = 该题这轮白等 5 分钟）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

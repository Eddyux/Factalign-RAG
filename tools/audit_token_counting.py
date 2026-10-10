#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""token 计数**交叉对账**：三份独立记录互相核，找出「记错记混」。

同一个 run 的 token 有三份独立来源，正常应当对得上：

  ① **逐题结果行** `results/{method}/{ds}/v{v}r{r}.jsonl` 的 `prompt_tokens` /
     `completion_tokens` —— 每题 `snapshot()` 差值累加（`run.py:558/593`）。
  ② **LLM 逐条审计** `logs/{method}/*.llm_calls.jsonl` 的 `pt` / `ct` —— `chat()` 出口
     每条记录一次（**含失败与重试**，`common/llm.py:308-311`）。
  ③ **进程内全局桶** `get_stats()`（只在该 run 的汇总打印里，不落盘）与
     **vLLM 引擎计数** `results/build_tokens.jsonl`（建图段差值，独立于本项目内存统计）。

对不上的地方**每一处都有明确含义**，本工具把它们分门别类：

  · ① < ② 的差 = **重试/失败的花费**。`run.py:574-576` 有意「前两次失败不计费、第 3 次
    记最后一次」，所以逐题累加**天然小于**逐条审计 —— 这是**声明过的口径**，不是 bug。
    差值若与「重试过的题数」量级一致就正常；差得离谱才要查。
  · ② 的 `chat_calls` 与 ① 的 `llm_calls` 差值 = 失败次数（失败也计 chat_calls，
    但 `pt=ct=0`，见 `common/llm.py:308-311`）。
  · ② 里 `ct >= max_tokens` 的条数 = **被 max_tokens 截断**的调用（截断会少算 completion）。
  · ② 里 `pt + ct == 0` 且带 `error` 的条数 = 真正的失败调用。
  · `phase` 字段：审计行自带 phase。**若建图期的调用被记成 `query`，说明该 adapter 的
    线程池没走 `set_default_phase("build")`** —— 这正是 `common/llm.py:109-128` 记的那个
    真实事故（proprag @2wiki 建图记成 `chat prompt=0`）。

用法（服务器，语料/日志在那边）：
    python tools/audit_token_counting.py --method proprag --dataset hotpotqa --sample 500
    python tools/audit_token_counting.py --method proprag --dataset hotpotqa --sample 500 --v 1 --r 1
    python tools/audit_token_counting.py --method proprag --dataset hotpotqa --sample 500 --phase-expect build
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _read_jsonl(p: Path):
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--v", type=int, default=1)
    ap.add_argument("--r", type=int, default=1)
    args = ap.parse_args(argv[1:])

    # ── ① 逐题结果行 ──
    res = ROOT / "results" / args.method / args.dataset / f"v{args.v}r{args.r}.jsonl"
    if not res.exists():                     # 兼容旧布局
        alt = ROOT / "results" / f"{args.method}_{args.dataset}_v{args.v}r{args.r}.jsonl"
        res = alt if alt.exists() else res
    rows = list(_read_jsonl(res)) if res.exists() else []
    r_pt = sum(int(r.get("prompt_tokens") or 0) for r in rows)
    r_ct = sum(int(r.get("completion_tokens") or 0) for r in rows)
    r_calls = sum(int(r.get("llm_calls") or 0) for r in rows)
    r_retried = sum(1 for r in rows if int(r.get("retries") or 0) >= 1)
    print(f"① 逐题结果行  {res}")
    print(f"   {len(rows)} 题 | prompt={r_pt:,} completion={r_ct:,} "
          f"| llm_calls={r_calls:,} | 重试过的题={r_retried}")
    if not rows:
        print(f"   ⚠ 没有结果行（路径对不对？）")

    # ── ② LLM 逐条审计 ──
    pats = [str(ROOT / "logs" / args.method / "*.llm_calls.jsonl"),
            str(ROOT / "logs" / args.method / f"*{args.dataset}*.llm_calls.jsonl")]
    files: list[Path] = []
    for p in pats:
        for f in sorted(glob.glob(p)):
            fp = Path(f)
            if fp not in files:
                files.append(fp)
    if not files:
        print(f"\n② ⚠ 找不到 logs/{args.method}/*.llm_calls.jsonl")
        return 2
    a_pt = a_ct = 0
    by_phase: Counter = Counter()
    by_phase_tok: Counter = Counter()
    n_chat = n_err = n_zero = n_trunc = 0
    trunc_by_role: Counter = Counter()
    roles: Counter = Counter()
    per_file: list[tuple[str, int, int, int]] = []
    print(f"\n② LLM 逐条审计  {len(files)} 个文件")
    for f in files:
        f_pt = f_ct = f_n = 0
        for rec in _read_jsonl(f):
            if rec.get("kind") not in (None, "chat", "chat_messages"):
                continue
            pt = int(rec.get("pt") or 0)
            ct = int(rec.get("ct") or 0)
            f_pt += pt
            f_ct += ct
            f_n += 1
            by_phase[rec.get("phase") or "?"] += 1
            by_phase_tok[rec.get("phase") or "?"] += pt + ct
            roles[rec.get("role") or "-"] += 1
            if rec.get("error"):
                n_err += 1
            if pt + ct == 0:
                n_zero += 1
            mt = (rec.get("req") or {}).get("max_tokens")
            if mt and ct >= int(mt):
                n_trunc += 1
                trunc_by_role[rec.get("role") or "-"] += 1
        a_pt += f_pt
        a_ct += f_ct
        n_chat += f_n
        per_file.append((f.name, f_n, f_pt, f_ct))
    print(f"   共 {n_chat:,} 条 | prompt={a_pt:,} completion={a_ct:,}")
    print(f"   phase 分布（条数）: {dict(by_phase)}")
    print(f"   phase 分布（token）: {dict(by_phase_tok)}")
    print(f"   role 分布（条数）: {dict(roles.most_common(12))}")
    print(f"   ==0 token 的行 = {n_zero:,}（其中带 error 的 {n_err:,}）")
    print(f"   ct >= max_tokens 被截断 = {n_trunc:,}  按 role: {dict(trunc_by_role)}")
    for name, n, p, c in per_file:
        print(f"     · {name}: {n:,} 条 prompt={p:,} completion={c:,}")

    # ── 对账 ──
    print("\n═══ 对账 ═══")
    d_pt, d_ct, d_calls = a_pt - r_pt, a_ct - r_ct, n_chat - r_calls
    print(f"② - ①   prompt={d_pt:+,}  completion={d_ct:+,}  calls={d_calls:+,}")
    print(f"   （正数 = 有花费没进逐题累加。`run.py:574-576` **有意**不记前两次失败的花费，"
          f"所以这里通常为正；重试过的题 {r_retried} 道，差 {d_calls} 次调用）")
    if n_trunc:
        print(f"   ⚠ 有 {n_trunc:,} 条 completion 贴到 max_tokens —— 这些调用的输出被截断，"
              f"审计里的 ct 是**实际生成量**，但内容不完整（可能被解析成空 → 计入失败）")

    # ── ③ build_tokens.jsonl（vLLM 引擎侧差值，独立于内存统计）──
    bt = ROOT / "results" / "build_tokens.jsonl"
    if bt.exists():
        hit = [r for r in _read_jsonl(bt)
               if r.get("method") == args.method and r.get("dataset") == args.dataset
               and int(r.get("sample") or 0) == args.sample]
        print(f"\n③ vLLM 引擎差值  {bt.name}  （{len(hit)} 条匹配 {args.method}/{args.dataset}"
              f" n={args.sample}）")
        for r in hit[-3:]:
            print(f"   {r.get('ts')} v{r.get('graph_version')}: "
                  f"chat prompt={r.get('chat_prompt')} completion={r.get('chat_completion')} "
                  f"| embed={r.get('embed_prompt')}")
        if hit:
            gp = hit[-1].get("chat_prompt")
            print(f"   与②的建图段对比：② build phase prompt = "
                  f"{by_phase_tok.get('build', 0):,}（chat+completion 合计）")
            if gp is not None:
                print(f"   ⚠ ③ chat_prompt={gp:,} 是**引擎侧真值**；它与②的差 = "
                      f"绕过本项目统计的调用（如官方库自己发的 HTTP）或区间外调用")
    else:
        print(f"\n③ 无 {bt}（没跑过建图？）")

    # ── phase 归属体检 ──
    print("\n═══ phase 归属体检（记混的高发区）═══")
    if by_phase.get("query") and by_phase.get("build"):
        print(f"   两个 phase 都有记录（build {by_phase['build']:,} / query {by_phase['query']:,}）")
    elif by_phase.get("query") and not by_phase.get("build"):
        print(f"   ⚠ 只有 query、没有 build —— 若这个 run 建过图，说明建图期的调用被记进了"
              f" query 桶（`common/llm.py:109-128` 记的真实事故）。检查该 adapter 是否在"
              f" build 期间 set_default_phase(\"build\")")
    else:
        print(f"   phase 分布：{dict(by_phase)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

"""run_rounds.py — 按「拆解轮数」跑 test_layer 的一层薄封装（默认 5 轮），log / trace 照常记录。

为什么要这个文件：只是想换轮数、又不想每次手写一长串参数，**同时不动 test_layer.py**。
它不复制 test_layer 的任何逻辑：只把 `--decomp_steps N`（和配套的 `--gen_top`）补进 argv，
再调 `test_layer.main()`。其余口径完全由你命令行里的参数决定（--rerank / --exp / --tag …）。
★ 它**不 patch 任何东西** → 过程日志、逐条调用审计、逐题 trace 全按 test_layer 的默认开着。

轮数语义（与 test_layer / factalign 一致）：`--decomp_steps N` = 检索轮数上限，
**第 1 轮的查询就是原问题**，所以：
    N=4 → 1 原问题 + 3 次「分解/改写」（= 现有默认）
    N=5 → 1 原问题 + 4 次「分解/改写」   ← 默认值
    N=6 → 1 原问题 + 5 次 …

生成预算配套：池子上限 = `--decomp_top_k`(默认 5) × N，所以这里默认把 `--gen_top` 设成
`5 × N`（= 池子**全部**送进 prompt，不被 20 截断）。想保持 20（让精排真的筛掉尾部）就显式
`--gen-top 20`。（若你改了 `--decomp_top_k`，请自己显式给 `--gen-top`。）

产物（与直接跑 test_layer 完全一样）：
  logs/ours/{exp}/{ds}_{sample}_{tag}_{stamp}.log   + 同名 .llm_calls.jsonl
  results/ours/{exp}/test_layer_{tag}_{ds}.jsonl    + ourssummary.jsonl 追加一行
  traces/ours/{exp}/{tag}/{ds}/v{n}r{m}_n{sample}/{qid}.json
★ trace 目录里带 tag：**换轮数请换 --tag**（r4 / r5 / r6…），否则同一 tag 的逐题 trace 会被覆盖。

用法:
  python run_rounds.py --rounds 5 --dataset 2wiki --sample 500 --workers 32 --rerank hybrid \\
      --exp rounds --tag r5 --decompose --decompose_stmt --decompose_recall proposition
  python run_rounds.py --rounds 6 --dataset 2wiki --sample 500 --workers 32 --rerank hybrid \\
      --exp rounds --tag r6 --decompose --decompose_stmt --decompose_recall proposition \\
      --gen-top 25                     # 池 ≤30 → 精排会真的筛掉 5 条
"""

import argparse
import sys

_WRAP = argparse.ArgumentParser(add_help=False)
_WRAP.add_argument("--rounds", type=int, default=5,
                   help="检索轮数上限（第 1 轮 = 原问题）；默认 5 = 1 原问题 + 4 次改写")
_WRAP.add_argument("--gen-top", type=int, default=0,
                   help="送生成的命题数；0 = 自动 = decomp_top_k(5) × rounds（池子全送）")
_w, rest = _WRAP.parse_known_args()

# 已显式给 `--decomp_steps` 就尊重它，并让 gen_top 的默认值跟着**实际轮数**算
# （两种写法都认：`--decomp_steps 3` / `--decomp_steps=3`）
_eff_rounds = _w.rounds
for _i, _a in enumerate(rest):
    if _a == "--decomp_steps" and _i + 1 < len(rest):
        try:
            _eff_rounds = int(rest[_i + 1])
        except ValueError:
            pass
    elif _a.startswith("--decomp_steps="):
        try:
            _eff_rounds = int(_a.split("=", 1)[1])
        except ValueError:
            pass

# 池子上限 = decomp_top_k(默认 5) × 实际轮数；非默认 decomp_top_k 请自己显式给 --gen-top
_gen_top = _w.gen_top or (5 * _eff_rounds)

_inject = []
if not any(a.split("=")[0] == "--decomp_steps" for a in rest):
    _inject += ["--decomp_steps", str(_w.rounds)]
if not any(a.split("=")[0] == "--gen_top" for a in rest):
    _inject += ["--gen_top", str(_gen_top)]

sys.argv = [sys.argv[0], *rest, *_inject]
print(f"[run_rounds] 轮数={_eff_rounds}（1 原问题 + {max(_eff_rounds - 1, 0)} 次分解/改写）"
      f"；追加参数: {' '.join(_inject) if _inject else '(无，已由你给全)'}", flush=True)
if any(_a in ("-h", "--help") for _a in rest):
    print("run_rounds.py 额外参数（test_layer 没有的）:\n"
          "  --rounds N    检索轮数上限（默认 5 = 1 原问题 + 4 次改写）→ 传给 --decomp_steps\n"
          "  --gen-top M   送生成的命题数（默认 = 5 × 轮数，池子全送）→ 传给 --gen_top\n"
          "以下为 test_layer 的参数:", flush=True)

import test_layer  # noqa: E402  （放在 argv 拼好之后导入，行为与直接跑 test_layer.py 一致）

if __name__ == "__main__":
    test_layer.main()

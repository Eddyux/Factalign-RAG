"""压力对账：多线程 + 反复 reset 下，**没有任何 token 能丢**。

判据最强的一条：**全局合计 == 所有线程所有调用的用量之和**（逐条审计的 `pt` 累加）。
只要有一个桶没登记，全局就会小于逐条之和。

覆盖：32 个 worker 线程、各自多轮调用、主线程在中途**反复** reset。
"""
import sys
import threading
import time

sys.path.insert(0, r"C:\pyproject\HyperfactRAG-main")
import common.llm as L

PT_PER_CALL = 100
CT_PER_CALL = 7


def fake_chat_vllm(system, user, json_mode, timeout, thinking=True,
                   max_tokens=None, flatten=True, stop=None):
    time.sleep(0.001)
    return "out", PT_PER_CALL, CT_PER_CALL, "out"


L._chat_vllm = fake_chat_vllm
L.LLM_CALL_LOG_ON = False          # 这轮只测记账，不写日志

WORKERS, ROUNDS, RESETS = 32, 8, 3


def worker(wid, barrier, round_one_ready, round_one_go):
    """每轮：先做**一次**调用并等门（让主线程能在确定时刻 reset），再做完剩下的轮次。

    ★ 为什么要等门：`reset_stats()` 的时序必须是确定的，否则"reset 时已经花了多少"
      每次不同，断言就只能写得松（我第一版写 `calls > 0` / `calls >= 线程数`，
      宽松到 bug 把 calls 从 480 砍到 256 都照样 PASS）。
    """
    barrier.wait()
    with L.phase("query"):
        L.chat("s", f"w{wid}-r0")
    round_one_ready.wait()                 # 32 个线程都完成第 1 次调用
    round_one_go.wait()                    # 等主线程 reset 完
    for r in range(1, ROUNDS):
        with L.phase("query"):
            L.chat("s", f"w{wid}-r{r}")
        time.sleep(0.001)


L.reset_stats()
results = []
for cycle in range(RESETS + 1):
    barrier = threading.Barrier(WORKERS + 1)
    ready = threading.Barrier(WORKERS + 1)     # 每线程完成第 1 次调用
    go = threading.Event()                     # 主线程放行
    ts = [threading.Thread(target=worker, args=(i, barrier, ready, go))
          for i in range(WORKERS)]
    for t in ts:
        t.start()
    barrier.wait()
    ready.wait()                               # ← 此时恰好 32 次调用已花掉
    L.reset_stats()                            # 确定时刻 reset（旧账清零）
    if cycle == RESETS:
        base = L.get_stats()["query"]
    go.set()                                   # 放行：每线程再做 ROUNDS-1 次
    for t in ts:
        t.join()
    s = L.get_stats()
    results.append((cycle, s["query"]["prompt_tokens"], s["query"]["completion_tokens"],
                    s["query"]["chat_calls"], s["query"]["chat_seconds"]))

# 最后一次 reset 之后，应恰好记录 WORKERS × (ROUNDS-1) 次调用
expect_calls = WORKERS * (ROUNDS - 1)
final = L.get_stats()["query"]
print(f"{WORKERS} 线程 × {ROUNDS} 轮；每轮在「全部线程各花 1 次」的确定时刻 reset")
print(f"{'轮次':>4} | {'全局 pt':>9} | {'全局 calls':>10}")
print("-" * 32)
for cycle, qpt, _qct, qc, _qs in results:
    print(f"{cycle:>4} | {qpt:>9,} | {qc:>10,}")

print(f"\n最后一次 reset 之后：应恰好 {expect_calls} 次调用")
print(f"实际：{final['chat_calls']} 次 | pt={final['prompt_tokens']:,} "
      f"ct={final['completion_tokens']:,}")
ok_calls = final["chat_calls"] == expect_calls
ok_tokens = (final["prompt_tokens"] == expect_calls * PT_PER_CALL
             and final["completion_tokens"] == expect_calls * CT_PER_CALL)
ok_seconds = final["chat_seconds"] > 0        # 时间也要跟着记上
print(f"判据 1：调用次数 == {expect_calls}（一个不丢）: {ok_calls}")
print(f"判据 2：pt/ct 恰好 = 次数 × ({PT_PER_CALL}/{CT_PER_CALL}): {ok_tokens}")
print(f"判据 3：chat_seconds 也被记上（不是只丢 token）: {ok_seconds}")
print(f"\n结论: {'PASS' if (ok_calls and ok_tokens and ok_seconds) else 'FAIL'}")

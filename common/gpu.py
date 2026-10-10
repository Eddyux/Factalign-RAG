"""GPU 时间/显存采样（SIGMOD 成本表要的是「GPU 花了多久」，不是「客户端等了多久」）。

为什么必须单独量：
  `common/llm.py` 记的 `chat_seconds` / `embed_seconds` 是**客户端挂钟时间**——
  在并发下包含 vLLM 排队 + 网络往返。把它当 "GPU 时间" 报给审稿人，在并发 8~16 时
  会严重失真（同一个模型、同样的 token，并发越高"GPU 时间"越大，物理上不成立）。
  真正可复现的量是 **GPU busy 秒**（Σ 时间片 × 利用率），外加峰值显存。

做法：后台线程每 `interval` 秒采一次 `nvidia-smi`（不需要 pynvml/pytorch 依赖），
     GPU busy 秒 = Σ_i util_i × Δt_i（对每张卡分别积分后求和）
  没有 nvidia-smi（开发机/CPU 机）时自动降级为 "unavailable"，不影响任何实验。

同时把 `vLLM /metrics` 里的 token 累计读出来做交叉校验（见 common/llm.vllm_token_count）：
  summary 里 client 侧 token 与 server 侧 token 应该对得上，对不上说明有截断/重试。

用法（run.py 已接）::

    g = GPUProbe(interval=1.0).start()
    ...
    stats = g.stop()      # {available, gpus, seconds_wall, gpu_busy_seconds, peak_mem_mb, energy_j}
"""

import re
import shutil
import subprocess
import threading
import time

# nvidia-smi 查询字段（老驱动可能没有 utilization.memory / power.draw → 单独降级）
_FIELDS_FULL = ("index,name,utilization.gpu,memory.used,memory.total,power.draw")
_FIELDS_MIN = ("index,name,utilization.gpu,memory.used,memory.total")
_NUM = re.compile(r"^\s*(\d+)\s*,\s*(.*?)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+|N/A)\s*)?$")


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _query(fields: str):
    """跑一次 nvidia-smi，返回 [(idx, name, util%, mem_used_mb, mem_total_mb, power_w|None)]。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
    except Exception:                       # noqa: BLE001
        return None
    if out.returncode != 0:
        return None
    rows = []
    for line in (out.stdout or "").strip().splitlines():
        m = _NUM.match(line)
        if not m:
            continue
        idx, name, util, used, total, power = m.groups()
        rows.append({"index": int(idx), "name": name,
                     "util": float(util), "mem_used_mb": float(used),
                     "mem_total_mb": float(total),
                     "power_w": (float(power) if power and power != "N/A" else None)})
    return rows or None


class GPUProbe:
    """后台采样 GPU 利用率/显存/功耗，给出 **GPU busy 秒** 与峰值显存。"""

    def __init__(self, interval: float = 1.0):
        self.interval = max(0.2, float(interval))
        self.available = False
        self.gpus: dict[int, dict] = {}
        self._samples: list[tuple[float, list]] = []
        self._stop = threading.Event()
        self._thr = None
        self.t0 = self.t1 = 0.0

    # ── 生命周期 ──
    def start(self) -> "GPUProbe":
        self.t0 = time.time()
        if not nvidia_smi_available():
            self.available = False
            return self
        first = _query(_FIELDS_FULL) or _query(_FIELDS_MIN)
        if not first:
            return self
        self.available = True
        for g in first:
            self.gpus[g["index"]] = {"name": g["name"], "mem_total_mb": g["mem_total_mb"],
                                     "peak_mem_used_mb": g["mem_used_mb"],
                                     "peak_util": g["util"], "power_w_peak": g["power_w"]}
        self._samples.append((time.time(), first))
        self._thr = threading.Thread(target=self._loop, name="gpu-probe", daemon=True)
        self._thr.start()
        return self

    def _loop(self):
        while not self._stop.wait(self.interval):
            rows = _query(_FIELDS_FULL) or _query(_FIELDS_MIN)
            if not rows:
                continue
            now = time.time()
            self._samples.append((now, rows))
            for g in rows:
                info = self.gpus.setdefault(g["index"], {
                    "name": g["name"], "mem_total_mb": g["mem_total_mb"],
                    "peak_mem_used_mb": 0.0, "peak_util": 0.0, "power_w_peak": None})
                info["peak_mem_used_mb"] = max(info["peak_mem_used_mb"], g["mem_used_mb"])
                info["peak_util"] = max(info["peak_util"], g["util"])
                if g["power_w"] is not None:
                    info["power_w_peak"] = max(info["power_w_peak"] or 0.0, g["power_w"])

    def stop(self) -> dict:
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=max(2.0, self.interval * 2))
        self.t1 = time.time()
        return self.summary()

    # ── 汇总 ──
    def summary(self) -> dict:
        wall = round((self.t1 or time.time()) - self.t0, 2) if self.t0 else 0.0
        if not self.available or not self._samples:
            return {"available": False, "seconds_wall": wall,
                    "note": "没有 nvidia-smi 或采样失败 → GPU 时间不可用（不影响实验）"}
        # 逐卡梯形积分：busy 秒 = Σ util% × Δt；同时按卡累计采样到的 Δt
        busy: dict[int, float] = {i: 0.0 for i in self.gpus}
        energy: dict[int, float] = {i: 0.0 for i in self.gpus}
        samples_n = len(self._samples)
        for (t_a, rows_a), (t_b, rows_b) in zip(self._samples, self._samples[1:]):
            dt = t_b - t_a
            if dt <= 0 or dt > max(10.0, self.interval * 10):     # 窗口异常（休眠）→ 丢弃
                continue
            by_idx = {r["index"]: r for r in rows_a}
            for r in rows_b:
                a = by_idx.get(r["index"], r)
                util = (a["util"] + r["util"]) / 2.0
                busy[r["index"]] = busy.get(r["index"], 0.0) + util / 100.0 * dt
                pa, pb = a.get("power_w"), r.get("power_w")
                if pa is not None and pb is not None:
                    energy[r["index"]] = energy.get(r["index"], 0.0) + (pa + pb) / 2.0 * dt
        gpus = []
        for idx, info in sorted(self.gpus.items()):
            gpus.append({**info, "gpu_busy_seconds": round(busy.get(idx, 0.0), 2),
                         "energy_j": round(energy.get(idx, 0.0), 1) if energy.get(idx) else None})
        n_gpu = max(1, len(self.gpus))
        return {
            "available": True,
            "seconds_wall": wall,                       # 这段 run 的挂钟时长
            "samples": samples_n,
            "interval_sec": self.interval,
            "n_gpus": len(self.gpus),
            "gpu_busy_seconds": round(sum(g["gpu_busy_seconds"] for g in gpus), 2),
            # 单卡视角的「等价占用时间」：多卡时便于和 wall 比较（>100% = 多卡并行）
            "gpu_busy_seconds_per_gpu": round(sum(g["gpu_busy_seconds"] for g in gpus) / n_gpu, 2),
            "gpu_utilization_avg": round(
                sum(g["gpu_busy_seconds"] for g in gpus) / max(1e-9, wall * n_gpu), 3),
            "peak_mem_used_mb": max((g["peak_mem_used_mb"] for g in gpus), default=None),
            "energy_j_total": round(sum(g["energy_j"] or 0.0 for g in gpus), 1) or None,
            "gpus": gpus,
        }


def probe_once() -> dict:
    """不需要后台线程的即时快照（供 check_env 之类的自检用）。"""
    if not nvidia_smi_available():
        return {"available": False}
    rows = _query(_FIELDS_FULL) or _query(_FIELDS_MIN)
    return {"available": bool(rows), "gpus": rows}

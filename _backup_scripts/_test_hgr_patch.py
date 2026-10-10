"""隔离验证 hypergraphrag_adapter 的 VDB patch（不依赖 vLLM / aioboto3）。

加载真实 adapter 文件 + 真实 fork storage.py，用假 client 验证：
1. patch 生效（NanoVectorDBStorage.query 被替换）
2. 同步扫描确实挪出了 loop 线程
3. 8 路并发真正并行（假 client sleep 0.2s，串行应 ~1.6s）
4. 返回格式与原版一致
"""
import importlib.util
import sys
import types
import asyncio
import threading
import time

FORK = r"D:\pyproject\HyperfactRAG\baselines\HyperGraphRAG-main"
sys.path.insert(0, FORK)

# ── stub hypergraphrag 包（跳过 __init__.py 的 aioboto3/ollama 重依赖）──
pkg = types.ModuleType("hypergraphrag")
pkg.__path__ = [FORK + r"\hypergraphrag"]
sys.modules["hypergraphrag"] = pkg

# ── stub 项目依赖（patch 只用 asyncio + fork 导入，其余 import 仅占位）──
m = types.ModuleType("methods"); m.__path__ = []
sys.modules["methods"] = m
mb = types.ModuleType("methods.base")
class BaseMethod: pass
mb.BaseMethod = BaseMethod
sys.modules["methods.base"] = mb

c = types.ModuleType("common"); c.__path__ = []
sys.modules["common"] = c
cl = types.ModuleType("common.llm")
for _n in ("chat_messages", "embed_batch", "bucket_ctx", "get_phase_bucket",
           "set_default_phase", "split_by_tokens"):
    setattr(cl, _n, object())
sys.modules["common.llm"] = cl

cfg = types.ModuleType("config")
cfg.BASELINES_DIR = None
cfg.CHAT_MODEL = "stub"
cfg.EXTRACT_CHUNK_TOKENS = 400
cfg.EXTRACT_CHUNK_OVERLAP = 40
sys.modules["config"] = cfg

# ── 加载真实 adapter 文件（非副本），执行真实 patch ──
spec = importlib.util.spec_from_file_location(
    "methods.hypergraphrag_adapter",
    r"D:\pyproject\HyperfactRAG\methods\hypergraphrag_adapter.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

adapter.HyperGraphRAG._patch_vdb_query_to_threadpool()

from hypergraphrag.storage import NanoVectorDBStorage  # noqa: E402

assert NanoVectorDBStorage.query.__name__ == "_query_off_loop", \
    f"patch 未生效: {NanoVectorDBStorage.query}"
print("1) patch 生效：NanoVectorDBStorage.query ->", NanoVectorDBStorage.query.__name__)

# ── 假实例（跳过 __post_init__，只填 query 用到的三个属性）──
import numpy as np  # noqa: E402

inst = NanoVectorDBStorage.__new__(NanoVectorDBStorage)

async def fake_embed(texts):
    assert isinstance(texts, list) and texts, "embedding_func 应收到 [query] 列表"
    return np.random.rand(len(texts), 1024).astype(np.float32)

inst.embedding_func = fake_embed
inst.cosine_better_than_threshold = 0.2

class _FakeClient:
    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = 0
    def query(self, *, query, top_k=5, better_than_threshold=None):
        self.calls += 1
        self.tid = threading.get_ident()
        if self.delay:
            time.sleep(self.delay)
        return [{"__id__": "ent-1", "__metrics__": 0.93, "entity_name": "X"}]

async def main():
    loop_tid = threading.get_ident()

    # 2) 扫描线程 ≠ loop 线程
    client = _FakeClient()
    inst._client = client
    out = await inst.query("test question", top_k=5)
    assert client.calls == 1 and client.tid != loop_tid, "扫描没挪出 loop 线程"
    assert out[0]["id"] == "ent-1" and out[0]["distance"] == 0.93 and \
        out[0]["entity_name"] == "X", f"返回格式不对: {out}"
    print("2) 扫描在线程池执行（tid != loop tid），返回格式与原版一致")

    # 3) 8 路并发并行（每路 client sleep 0.2s；串行 ~1.6s）
    client = _FakeClient(delay=0.2)
    inst._client = client
    t0 = time.time()
    results = await asyncio.gather(*[inst.query(f"q{i}", top_k=5) for i in range(8)])
    wall = time.time() - t0
    assert len(results) == 8 and all(r[0]["id"] == "ent-1" for r in results)
    assert wall < 1.2, f"8 路并发 wall={wall:.2f}s（串行应为 ~1.6s），并发未生效"
    print(f"3) 8 路并发 wall={wall:.2f}s（串行 ~1.6s）→ 真并行")

asyncio.run(main())
print("\nPATCH_OK：全部断言通过")

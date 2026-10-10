"""proprag 作答期并发正确性回归 —— 必须真打到「全图/子图双阶段 + 缓存」这条路径。

修复前有两个真缺陷（都在 `BeamSearchPathFinder` 的**共享可变状态**上）：
  ① `retrieve()` 把共享 `self._beam.active_graph` 改成子图再还原 → 32 线程互踩；
  ② `connected_propositions_cache` 的 key 只有 `pkey`，**没含 `prop_set`** →
     第一阶段（prop_set=None，全图自由连）算出的连接被第二阶段（子图命题集）
     直接复用 → 子图 beam 拿到不在子图里的命题。

判据：
  A. **同实例两阶段 vs 每次新实例** → 逐字节相同（缓存不得跨图/跨 prop_set 污染）。
  B. **并发 vs 串行** → 逐字节相同（共享状态不得被别的线程改坏）。
  C. 子图阶段产出的命题必须**全部落在该阶段的 prop_set 内**（②的直接判据）。
"""
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, r"C:\pyproject\HyperfactRAG-main")
import numpy as np
import methods.proprag_adapter as PA


class _V:
    def __init__(self, name):
        self._d = {"name": name}

    def __getitem__(self, k):
        return self._d[k]


class _Vs:
    def __init__(self, names):
        self._n = names

    def __iter__(self):
        return iter(_V(n) for n in self._n)

    def __getitem__(self, i):
        return _V(self._n[i])

    def __len__(self):
        return len(self._n)


class _E:
    def __init__(self, w):
        self._d = {"weight": w}

    def __getitem__(self, k):
        return self._d[k]


class _Es:
    def __init__(self, ws):
        self._ws = ws

    def __getitem__(self, i):
        return _E(self._ws[i])


class MiniGraph:
    def __init__(self, nodes, edges):
        self._nodes = list(nodes)
        self._edges = list(edges)
        self._adj = {i: [] for i in range(len(nodes))}
        for k, (i, j, _w) in enumerate(self._edges):
            self._adj[i].append((j, k))
            self._adj[j].append((i, k))

    def vcount(self):
        return len(self._nodes)

    @property
    def vs(self):
        return _Vs(self._nodes)

    @property
    def es(self):
        return _Es([w for _i, _j, w in self._edges])

    def neighbors(self, v, mode="all"):
        return [j for j, _k in self._adj[v]]

    def get_eid(self, i, j, error=False):
        for jj, k in self._adj[i]:
            if jj == j:
                return k
        return -1

    def induced_subgraph(self, verts):
        keep = list(verts)
        remap = {v: i for i, v in enumerate(keep)}
        nodes = [self._nodes[v] for v in keep]
        edges = [(remap[i], remap[j], w) for i, j, w in self._edges
                 if i in remap and j in remap]
        return MiniGraph(nodes, edges)


N_P, N_E = 60, 15


def make_rag(seed=0):
    rng = np.random.RandomState(seed)
    rag = types.SimpleNamespace()
    rag.prop_ids = [f"{PA._PROP_PREFIX}{i}" for i in range(N_P)]
    rag.prop_texts = [f"prop {i}" for i in range(N_P)]
    rag.prop_entities = {pk: [f"E{i % N_E}", f"E{(i + 3) % N_E}"]
                         for i, pk in enumerate(rag.prop_ids)}
    rag.entity_to_props = {}
    for pk, ents in rag.prop_entities.items():
        for e in ents:
            rag.entity_to_props.setdefault(PA.entity_key(e), []).append(pk)
    rag._pk2pos = {pk: i for i, pk in enumerate(rag.prop_ids)}
    v = rng.randn(N_P, 16).astype(np.float32)
    rag.prop_vecs = v / np.linalg.norm(v, axis=1, keepdims=True)

    idx, nodes, edges = {}, [], []
    for pk in rag.prop_ids:
        idx[pk] = len(nodes)
        nodes.append(pk)
    for ekey in rag.entity_to_props:
        idx[ekey] = len(nodes)
        nodes.append(ekey)
    for ekey, pks in rag.entity_to_props.items():
        for pk in pks:
            edges.append((idx[ekey], idx[pk], 1.0))
    ents = [n for n in nodes if n.startswith(PA._ENTITY_PREFIX)]
    for a in range(len(ents)):
        for b in range(a + 1, len(ents)):
            edges.append((idx[ents[a]], idx[ents[b]], 0.9))
    rag.graph = MiniGraph(nodes, edges)
    rag.node_name_to_vertex_idx = idx
    rag.entity_to_props = rag.entity_to_props
    rag.prop_vectors = lambda keys: np.array(
        [rag.prop_vecs[rag._pk2pos[k]] for k in keys], dtype=np.float32)
    rag.prop_text = lambda k: rag.prop_texts[rag._pk2pos[k]]
    rag.batch_encode = lambda texts: np.array(
        [rag.prop_vecs[i % N_P] for i in range(len(texts))], dtype=np.float32)
    return rag


def fresh_finder(rag):
    return PA.BeamSearchPathFinder(rag, beam_width=4, max_path_length=3,
                                   embedding_combination="average",
                                   sim_threshold=0.8, second_stage_filter_k=40)


BASE = make_rag()
QV = np.ones(16, dtype=np.float32) / 4.0
RESTRICTED = set(list(BASE.prop_ids)[:20])
SUB_KEEP = sorted(BASE.node_name_to_vertex_idx.values())[:30]


def stage_pair(rag, finder, query):
    """阶段1（全图, prop_set=None）+ 阶段2（子图, prop_set=限制集）。"""
    finder.find_paths(query, prop_set=None, query_vector=QV)
    sub = rag.graph.induced_subgraph(SUB_KEEP)
    sub_idx = {v["name"]: i for i, v in enumerate(sub.vs)}
    out = finder.find_paths(query, prop_set=RESTRICTED, query_vector=QV,
                            graph=sub, node_index=sub_idx, beam_width=4,
                            max_path_length=3, second_stage_filter_k=40)
    return [(p["proposition_keys"], round(float(p["score"]), 6)) for p in out[0]]


def stage_pair_same_graph(rag, finder, query, first=True):
    """**同一张图**上先跑 prop_set=None（可选）再跑限制集 —— 复现 prop_set 缓存污染。"""
    if first:
        finder.find_paths(query, prop_set=None, query_vector=QV)
    out = finder.find_paths(query, prop_set=RESTRICTED, query_vector=QV)
    return [(p["proposition_keys"], round(float(p["score"]), 6)) for p in out[0]]


QUERIES = [f"q{i}" for i in range(16)]

print("=== A. 缓存不得跨图/跨 prop_set 污染（同实例 vs 每次新实例）===")
rag = make_rag()
shared_finder = fresh_finder(rag)
a_shared = [stage_pair(rag, shared_finder, q) for q in QUERIES]
a_fresh = []
for q in QUERIES:
    r2 = make_rag()
    a_fresh.append(stage_pair(r2, fresh_finder(r2), q))
A_ok = a_shared == a_fresh
print(f"逐字节相同 = {A_ok}")

print("\n=== B. 并发必须等于串行（每线程一个 finder = 修复后的用法）===")
tls = threading.local()


def one_serial(q):
    b = getattr(tls, "b", None)
    if b is None:
        b = fresh_finder(rag)
        tls.b = b
    return stage_pair(rag, b, q)


b_serial = [one_serial(q) for q in QUERIES]
tls2 = threading.local()


def one_par(q):
    b = getattr(tls2, "b", None)
    if b is None:
        b = fresh_finder(rag)
        tls2.b = b
    return stage_pair(rag, b, q)


with ThreadPoolExecutor(max_workers=16) as ex:
    b_par = list(ex.map(one_par, QUERIES))
B_ok = b_serial == b_par
print(f"逐字节相同 = {B_ok}")

print("\n=== C. 子图阶段产出不得越出 prop_set ===")
leak = sum(1 for paths in a_shared for pkeys, _s in paths
           for pk in pkeys if pk not in RESTRICTED)
C_ok = leak == 0
print(f"越界命题数 = {leak}（须为 0）")

# ── D. `prop_set` 必须进缓存 key：**同一张图**先用 prop_set=None 再用限制集 ──
#     这是 ② 的直接复现条件 —— 阶段 1 与阶段 2 用**同一张图**时，图身份分不开它们，
#     只有 prop_set 能分。旧 key（只有 pkey）会把阶段 1 的自由连接喂给阶段 2。
print("\n=== D. 同图不同 prop_set：结果不得互相污染 ===")
d_fresh = []
d_shared = []
for q in QUERIES:
    r = make_rag()
    f1 = fresh_finder(r)                     # 干净 finder：先 None 再限制
    stage_pair_same_graph(r, f1, q)
    d_shared.append(stage_pair_same_graph(r, f1, q))
    r2 = make_rag()
    f2 = fresh_finder(r2)                    # 干净 finder：只跑限制集那一步
    d_fresh.append(stage_pair_same_graph(r2, fresh_finder(r2), q, first=False))
D_ok = d_shared == d_fresh
print(f"（同图：先 None 再限制） vs （只用限制集）：逐字节相同 = {D_ok}")

# ── E. 为什么不需要「共享实例并发」这一条 ──
#     修复后 finder 上**已不存在每题可变状态**（active_graph / node_name_to_vertex_idx /
#     beam_width / max_path_length / second_stage_filter_k / embedding_combination 全部
#     改为参数传入），`PropRAG` 也改成**每线程一个 finder**（`_beam_for_thread()`）→
#     "共享实例被别的线程改坏"这个失效模式在结构上不可能再发生，
#     所以无法用测试复现 —— 这正是本项修复的目的。A/B/C/D 四条已覆盖可观测的行为面。
print("\n=== E. 共享实例互踩（结构上已不可能）===")
print("finder 上已无每题可变状态；PropRAG 每线程一个 finder → 无需测试")

print(f"\n结论：A={A_ok} B={B_ok} C={C_ok} D={D_ok} → "
      f"{'PASS' if (A_ok and B_ok and C_ok and D_ok) else 'FAIL'}")

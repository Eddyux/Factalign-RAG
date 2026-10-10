"""HippoRAG 单块重抽工具：只动 openie 结果文件，不碰图/向量，不改 baselines/ 源码。

背景（为什么需要它）
--------------------
官方 HippoRAG 的 OpenIE 本来就是**增量**的：
  `HippoRAG.index()` → `load_existing_openie()`（HippoRAG.py:1021-1065）读出
  `openie_results_ner_<model>.json`，只为「**不在这份文件里**的 chunk key」重跑 LLM；
  `merge_openie_results()`（:1067-1110）再把新结果追加回去。
所以「单独重抽某几块」是官方原生支持的——**前提是那几块不在文件里**。

坑在于：抽取失败的块也会被写进文件。失败在官方内部被吞掉
（`openie_openai.py:112-120`：`except` → 记 warning → 返回 `triples=[]`；
NER 失败同理 `:63-72` → `unique_entities=[]`），而 `merge_openie_results` 照样把它落盘成
`{'extracted_entities': [], 'extracted_triples': []}`。于是下一轮 `load_existing_openie`
认为它「已处理」→ **永远不会被重抽**，这块语料单元的事实就永久缺失。

因此单独重抽 = 把那些 doc 条目**删掉**（不是清空）：
下一轮 index() 就会把它当待处理，只对它发 LLM（其余块零调用），
`HippoRAG.py:306` 的 `assert len(chunk_to_rows) == len(ner_results_dict) == ...`
也仍然成立（同一轮里会被补回来）。

用法
----
  # 1) 看有多少块是空的（顺带把清单写到 json 旁边 openie_failed_chunks.txt）
  python tools/hipporag_reextract.py --dataset hotpotqa --graph-version 1 --report

  # 2) 只删日志里报错的那几块（最精确：日志里就是 `Exception for chunk <id>: ...`）
  python tools/hipporag_reextract.py --dataset hotpotqa --graph-version 1 \
      --from-log logs/hipporag/hotpotqa_300_*.log --drop

  # 3) 用原命令重跑（增量：只重抽被删掉的块，其余块不再发 LLM）
  python run.py --method hipporag --dataset hotpotqa --sample 300 --workers 1

⚠️ 前提：`data/hipporag/<dataset>/v<n>/build_complete.marker` 必须已存在。marker 不存在
（上一轮没跑完/换过建图参数）时，适配器会 `shutil.rmtree` 整个目录重建
（`methods/hipporag_adapter.py:319-324`）——那会连这份 openie json 一起清掉，变成全量重抽。
本工具检测到 marker 缺失会直接拒绝执行（除非 `--force`）。
"""

import argparse
import glob
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 让 tools/ 下能 import config

from config import DATA_DIR, CHAT_MODEL  # noqa: E402

EXC_RE = re.compile(r"Exception for chunk (\S+?):")


def _find_openie_json(working_dir: Path) -> Path:
    """官方文件名：<save_dir>/openie_results_ner_<llm_name 的 / 换 _>.json（HippoRAG.py:177）。"""
    want = working_dir / f"openie_results_ner_{CHAT_MODEL.replace('/', '_')}.json"
    if want.exists():
        return want
    found = sorted(working_dir.glob("openie_results_ner_*.json"))
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(
            f"{working_dir} 里没有 openie_results_ner_*.json —— 建图还没跑到落盘那一步？")
    raise FileNotFoundError(
        f"{working_dir} 里有多个 openie json（换过 CHAT_MODEL 会留下多个），请手动指定：\n"
        + "\n".join(f"  {p.name}" for p in found))


def _ids_from_logs(paths: list[str]) -> set[str]:
    ids: set[str] = set()
    for p in paths:
        for name in (sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]):
            path = Path(name)
            if not path.exists():
                print(f"[log] 跳过（不存在）：{path}")
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            hits = EXC_RE.findall(text)
            ids.update(hits)
            print(f"[log] {path}: 抓到 {len(hits)} 条 Exception")
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="数据集名（用 data/hipporag/<dataset>/v<n>/）")
    ap.add_argument("--graph-version", type=int, default=1)
    ap.add_argument("--working-dir", help="直接指定索引目录（覆盖 --dataset/--graph-version）")
    ap.add_argument("--report", action="store_true", help="只报告（默认动作）")
    ap.add_argument("--drop", action="store_true",
                    help="执行删除（先自动备份 json）；不带 --from-log 时删所有空三元组的块")
    ap.add_argument("--from-log", nargs="+", default=[], metavar="LOG",
                    help="从日志抓 `Exception for chunk <id>:` 的 id（支持通配符），只处理这些块")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--force", action="store_true", help="marker 缺失也照做（危险：重跑会全量重建）")
    args = ap.parse_args()

    if args.working_dir:
        working_dir = Path(args.working_dir)
    elif args.dataset:
        working_dir = DATA_DIR / "hipporag" / args.dataset / f"v{args.graph_version}"
    else:
        ap.error("需要 --dataset 或 --working-dir")

    if not working_dir.exists():
        print(f"✗ 目录不存在：{working_dir}")
        return 1
    marker = working_dir / "build_complete.marker"
    if not marker.exists() and not args.force:
        print(f"✗ {marker} 不存在：上一轮没跑完（或换过建图参数）。\n"
              f"  此时重跑会被适配器判定「状态变化」而整目录重建"
              f"（methods/hipporag_adapter.py:319-324），本工具改的 json 会被一起删掉。\n"
              f"  先让当前这轮跑完（会写 marker），或用 --force 明确表示你已知情。")
        return 2

    js = _find_openie_json(working_dir)
    data = json.loads(js.read_text(encoding="utf-8-sig"))     # 容忍 BOM
    docs = data.get("docs", [])
    by_id = {d.get("idx"): d for d in docs}

    empty_tri = [d for d in docs if not d.get("extracted_triples")]
    empty_ent = [d for d in docs if not d.get("extracted_entities")]
    log_ids = _ids_from_logs(args.from_log) if args.from_log else set()
    if log_ids:
        hit = sorted(i for i in log_ids if i in by_id)
        miss = sorted(i for i in log_ids if i not in by_id)
        print(f"[log] 日志里的块 {len(log_ids)} 个：json 里命中 {len(hit)}、未命中 {len(miss)}")
        if miss:
            print("      未命中（可能属于别的 run / 已被重建）："
                  + ", ".join(m[:26] for m in miss[:5]) + (" ..." if len(miss) > 5 else ""))

    print(f"\n[openie] {js}")
    print(f"  文档块总数            : {len(docs)}")
    print(f"  抽不出三元组的块      : {len(empty_tri)}"
          f"（{len(empty_tri) / max(1, len(docs)) * 100:.2f}%）")
    print(f"  抽不出实体的块        : {len(empty_ent)}"
          f"（{len(empty_ent) / max(1, len(docs)) * 100:.2f}%）")
    print(f"  两者都空的块          : "
          f"{sum(1 for d in docs if not d.get('extracted_triples') and not d.get('extracted_entities'))}")
    print("  注：json 里不含 metadata，无法区分「抽取崩了」和「这段真没三元组」；"
          "要精确就到日志里 grep 'Exception for chunk'（见 --from-log）")

    failed_list = working_dir / "openie_failed_chunks.txt"
    if log_ids:
        targets = sorted(i for i in log_ids if i in by_id)
    else:
        targets = [d["idx"] for d in empty_tri]
    if not args.drop:
        failed_list.write_text("\n".join(targets) + ("\n" if targets else ""), encoding="utf-8")
        print(f"\n[report] 待重抽清单（{len(targets)} 个）已写：{failed_list}")
        if targets:
            for i in targets[:5]:
                print(f"  - {i}  {by_id[i].get('passage', '')[:60]!r}")
            if len(targets) > 5:
                print(f"  ... 其余 {len(targets) - 5} 个见清单文件")
        print("\n下一步：--drop 删掉这些条目 → 用原命令重跑（增量，只重抽它们）")
        return 0

    if not targets:
        print("\n[drop] 没有要删的块，什么都不做")
        return 0
    if not args.no_backup:
        bak = js.with_suffix(js.suffix + f".bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(js, bak)
        print(f"\n[backup] {bak}")
    drop = set(targets)
    data["docs"] = [d for d in docs if d.get("idx") not in drop]
    js.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"[drop] 删掉 {len(docs) - len(data['docs'])} 个 doc 条目，"
          f"剩 {len(data['docs'])} 个 → {js}")
    print("下一步：python run.py --method hipporag --dataset "
          f"{args.dataset or '<dataset>'} --sample 300 --workers 1"
          "（增量：只有被删的块会再发 LLM）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

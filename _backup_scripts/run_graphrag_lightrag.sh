#!/usr/bin/env bash
# ============================================================
# 批量跑：graphrag × 4 数据集（hotpotqa/musique/2wiki/popqa）
#         + lightrag 补齐 popqa / 2wiki（hotpotqa/musique 已跑过）
#
# 每个 `python run.py` 调用结束时会自动：
#   1. 写每题结果   -> results/{method}_{dataset}.jsonl（断点续，重跑跳过已完成题）
#   2. 追加汇总行   -> results/summary.jsonl（EM/F1 + build/query 分桶 token + 耗时）
#
# ⚠️ graphrag 建索引很慢（300 题语料可能几小时），建议挂后台：
#     nohup bash run_graphrag_lightrag.sh > graphrag_lightrag_run.log 2>&1 &
#     然后 tail -f graphrag_lightrag_run.log 看进度
#
# 每个「方法 × 数据集」单独一条命令、单独一个 log；某个失败不中断后续。
# ============================================================
set -u
set -o pipefail   # 让 python 的退出码穿透 | tee，失败能被抓到

SAMPLE=300
WORKERS=16

LOGDIR=logs
mkdir -p "$LOGDIR"
FAILED=()

# 终端直接全量输出，不过滤：graphrag/lightrag 的进度条是 \r 覆盖式单行刷新
# （不带 \n），grep --line-buffered 只按 \n 刷新会把进度条整个吞掉；
# 两个 adapter 自己的终端 handler 本来就只打「进度条 + 报错」（INFO 写 LOG），
# 不需要再过滤。完整输出仍由 tee 全量存 LOG。

echo "== 数据预检（缺 300 题样本的数据集会跳过）=="
for DS in hotpotqa musique 2wiki popqa; do
  if ls "data/samples/${DS}_${SAMPLE}.json" >/dev/null 2>&1; then
    echo "  [OK] ${DS}_${SAMPLE}.json"
  else
    echo "  [缺] ${DS}_${SAMPLE}.json  -> 先跑: python sample_datasets.py --datasets ${DS} --sample ${SAMPLE}"
  fi
done
echo ""

run_one() {
  local METHOD=$1
  local DS=$2
  if ! ls "data/samples/${DS}_${SAMPLE}.json" >/dev/null 2>&1; then
    echo "[跳过] ${METHOD} @ ${DS}：数据缺失，先下数据"
    return 0
  fi
  local LOG="$LOGDIR/${METHOD}_${DS}.log"
  echo "=============================================================="
  echo "[$(date '+%F %T')] 开始 ${METHOD} @ ${DS} (n=${SAMPLE})"
  echo "  日志: ${LOG}"
  echo "=============================================================="
  python run.py --method "$METHOD" --dataset "$DS" --sample "$SAMPLE" --workers "$WORKERS" 2>&1 \
    | tee "$LOG"
  local status
  status=${PIPESTATUS[0]}
  if [ "$status" -eq 0 ]; then
    echo "[完成] ${METHOD} @ ${DS}"
  else
    echo "[失败] ${METHOD} @ ${DS}（详见 ${LOG}），继续下一个"
    FAILED+=("${METHOD} @ ${DS}")
  fi
  echo ""
}

# ── 1) graphrag × 4 数据集 ──
for DS in hotpotqa musique 2wiki popqa; do
  run_one graphrag "$DS"
done

# ── 2) lightrag 补齐剩余两个数据集（hotpotqa/musique 已跑过）──
for DS in popqa 2wiki; do
  run_one lightrag "$DS"
done

echo "=============================================================="
echo "全部跑完。"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "失败项："
  for f in "${FAILED[@]}"; do echo "  - $f"; done
else
  echo "无失败项。"
fi
echo ""
echo "token/耗时汇总: results/summary.jsonl"
echo "出对比表:       python report_tokens.py"

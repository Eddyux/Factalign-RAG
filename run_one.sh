#!/usr/bin/env bash
# ============================================================================
# run_one.sh —— 一行一个「数据集」＋ 一行一个「方法」；**要跑哪些就放开哪些行**。
#
#   全都不注释 → 五个数据集 × 全部方法 依次跑完
#   注释掉 1 行数据集 → 只跑剩下 4 个数据集（× 全部方法）
#   注释掉 1 行方法   → 所有选中的数据集中都不跑这个方法
#
# 用法:
#   bash run_one.sh                 # 会先打印「将要跑 N 个数据集 × M 个方法 = K 次」，要确认
#   bash run_one.sh --yes           # 不问，直接跑（推荐用在 nohup/后台）
#   SAMPLE=10 bash run_one.sh --yes # 冒烟：每题集只抽 10 题
#   WORKERS=8 bash run_one.sh --yes # 覆盖并发上限
#
# 想临时「全放开」而不改文件（比如从全部注释的版本直接开跑）：
#   sed 's/^#[[:space:]]*//' run_one.sh | bash -s -- --yes
#
# 注意:
#   · 逐行**从上到下**跑；一次跑一个（建图和检索别混跑，否则 KV cache 打满 → 超时）
#   · 断点续：同 v+r 重跑自动跳过已完成题（想重跑先删 results/{method}/{dataset}/v{n}r{m}.jsonl）
#   · 换模型/换切块/换题数 → **必须开新 V**；换查询侧参数 → 开新 R
#   · 单个方法不可用（官方源码未接入/缺依赖）不会中断整批：run.py 自己打「不可用」后正常退出
# ============================================================================
set -u

# ── tokenizer 离线（★ 服务器没有外网时必须）────────────────────────────────
# 每个 python 进程第一次切块/embed 都会 from_pretrained(EMBED_MODEL)（common/llm.py:765-782），
# 而 EMBED_MODEL 是 **HF 名** → 先去 HF_ENDPOINT（:777 默认 hf-mirror.com）做一次 HEAD，
# 超时则重试 5 次 × 10s。10 方法 × 5 数据集 = 50 个进程 ≈ 45 分钟白等；更糟的是
# split_by_tokens 会**静默退化成字符级切块**（common/llm.py:932 把异常吞掉）→
# 「哪些进程按 token 切、哪些按字符切」取决于当时网络通不通，同一份语料口径就已经不同。
# 设了这两个：不联网、直读 ~/.cache/huggingface/hub/；缓存里若真缺文件会**立刻**报
# LocalEntryNotFoundError（而不是慢慢重试），一眼可辨。想临时联网：HF_HUB_OFFLINE=0 bash run_one.sh
# 前置确认：ls ~/.cache/huggingface/hub/ | grep -i qwen3  → models--Qwen--Qwen3-Embedding-0.6B
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

cd "$(dirname "$0")" || exit 1        # 脚本在仓库根；放别处请改成仓库路径

# ── 配置（改这里）────────────────────────────────────────────────────────────
CFG=configs/baselines.json
PY=python                   # 虚拟环境写全路径，如 PY=/root/miniconda3/envs/rag/bin/python
SAMPLE=${SAMPLE:-500}       # 每题集抽多少题：10 = 冒烟；500 = 全量
WORKERS=${WORKERS:-32}      # 并发**上限**（实际并发由自适应闸动态调节，见 summary.concurrency）
V=${V:-1}                   # --graph-version：换模型/切块/题数 → 开新 v
R=${R:-1}                   # --retrieval：换查询参数 → 开新 r
ONLY_COLD=0                 # 1 = 每次切换数据集前做一次冷缓存检查（见文件末尾）
YES=0                       # --yes 时跳过确认
for a in "$@"; do [ "$a" = "--yes" ] && YES=1; done


# ════════════════════════════════════════════════════════════════════════════
# 数据集（放开=跑；注释掉=不跑）
#   hotpotqa / 2wiki / musique = 主实验（多跳）
#   popqa（单跳长尾）/ triviaqa（开放域，语料最大 → 建图类最贵）= 对照
# ════════════════════════════════════════════════════════════════════════════
DATASETS=$(grep -vE '^[[:space:]]*#' <<'LIST' | sed '/^[[:space:]]*$/d'
hotpotqa
# 2wiki
# musique
# popqa
# triviaqa
LIST
)

# ════════════════════════════════════════════════════════════════════════════
# 方法（放开=跑；注释掉=所有数据集里都不跑它）
# ════════════════════════════════════════════════════════════════════════════
METHODS=$(grep -vE '^[[:space:]]*#' <<'LIST' | sed '/^[[:space:]]*$/d'
# zeroshot
# standardrag
# factalign
# hypergraphrag
# lightrag
hipporag
# proprag
# ircot
# selfask
# s2grag
LIST
)


# ── 准备：把选中数据集的样本补齐（缺哪个补哪个；已有的不动）───────────────
prepare() {
  local ds
  echo "── 准备样本（--sample $SAMPLE）──"
  for ds in $DATASETS; do
    # ★ 守卫必须**同时**看「样本」与「合并语料指纹」，不能只看样本：
    #   `--steps corpus` 才是建 data/merged/ 的那一步（build_all.py:161-169），
    #   而它跟 samples 挤在**同一条命令**里 —— 只看样本文件就把整条命令跳过，
    #   merged 就**永远建不出来**（症状：验收里「chunk 数与 merged 一致」
    #   和「语料指纹无 problems」两条 WARN 永久挂着、消不掉）。
    #   三个步骤各自幂等（raw/samples/corpus 已存在即跳过，见 build_all.py:66/123/164），
    #   所以样本已存在时多跑一次这条命令**没有代价**，只会补上缺的 merged。
    if [ -f "data/samples/${ds}_${SAMPLE}.json" ] \
       && [ -f "data/merged/${ds}_${SAMPLE}/checksums.json" ]; then
      echo "  有 data/samples/${ds}_${SAMPLE}.json + data/merged/${ds}_${SAMPLE}/"
    else
      echo "  建 $ds ..."
      "$PY" tools/build_all.py --config "$CFG" --datasets "$ds" --sample "$SAMPLE" \
          --steps raw,samples,corpus || echo "  ⚠ $ds 准备失败，继续"
    fi
  done
}

# ── 跑一个「方法 × 数据集」──────────────────────────────────────────────────
run_cell() {
  local m="$1" ds="$2"
  if [ "$ONLY_COLD" = "1" ]; then
    "$PY" tools/clean_run.py --check-cold || true
  fi
  echo ""
  echo "════════ [$m @ $ds]  sample=$SAMPLE workers=$WORKERS v$V r$R ════════"
  "$PY" run.py --config "$CFG" --method "$m" --dataset "$ds" --sample "$SAMPLE" \
      --workers "$WORKERS" --graph-version "$V" --retrieval "$R"
  local rc=$?
  if [ "$rc" != "0" ]; then
    echo "⚠ [$m @ $ds] 退出码 $rc（继续跑下一个）"
  else
    # 跑完顺手验收（有 FAIL 只提示，不中断整批）
    "$PY" tools/verify_run.py --method "$m" --dataset "$ds" --sample "$SAMPLE" \
        --v "$V" --r "$R" || echo "⚠ [$m @ $ds] 验收未全过（见上面 FAIL 项）"
  fi
}

# ── 主流程 ──────────────────────────────────────────────────────────────────
NDS=$(echo "$DATASETS" | grep -c . )
NMT=$(echo "$METHODS" | grep -c . )
echo "选中数据集（$NDS）: $(echo $DATASETS | tr '\n' ' ')"
echo "选中方法（$NMT）  : $(echo $METHODS | tr '\n' ' ')"
echo "合计 $((NDS * NMT)) 次运行；逐行跑（一次一个）"
if [ "$YES" != "1" ]; then
  printf "继续？（输入 yes 回车）："
  ans=""
  read -r ans || true          # EOF/无 stdin（nohup、管道）时 read 失败 → 别让 set -u 炸
  [ "$ans" = "yes" ] || { echo "已取消（想无人值守请加 --yes）"; exit 0; }
fi

prepare

i=0
for ds in $DATASETS; do
  for m in $METHODS; do
    i=$((i + 1))
    echo ""
    echo "########## [$i/$((NDS * NMT))] $m @ $ds ##########"
    run_cell "$m" "$ds"
  done
done

echo ""
echo "全部完成：$i 次运行。"
echo "汇总：results/summary.jsonl（含 cost / gpu / server / concurrency 四块）"
echo "图与表：$PY tools/make_figures.py --out results/figures --format png,pdf"


# ════════════════════════════════════════════════════════════════════════════
# 备忘
# ----------------------------------------------------------------------------
# · 想只跑一个数据集：把其余 4 行数据集注释掉（方法那 10 行不用动）
# · 想只跑一个方法：  把其余 9 行方法注释掉（数据集那 5 行不用动）
# · 想跑「一个数据集的一个方法」：两边都只留 1 行；这种方式更适合直接写
#     $PY run.py --config $CFG --method proprag --dataset hotpotqa --sample $SAMPLE \
#         --workers $WORKERS --graph-version $V --retrieval $R
# · 冷缓存：prefix cache 是进程内的，跨 run 复用会让 prefill 几乎不耗时、成本不可比。
#     每次换数据集前想确认一下：ONLY_COLD=1 bash run_one.sh --yes
#     最彻底：重启 vLLM（见 sources/models.json 的 serve_cmd）
# · 建图凭证：data/{method}/{dataset}/v$V/build/{config,env,stats,checksums}.json
# · 逐题 prompt/原始输出/每轮检索：traces/{method}/{dataset}/v${V}r${R}_n${SAMPLE}/{qid}.json
# · 并发建议：standardrag 可 16+；建图类（hipporag/proprag/
#     s2grag）8；triviaqa 语料最大，建图类 4~8。跑完看 results/failures.jsonl 的重试率，
#     >10% 就该把 WORKERS 降下来（对 GPU-busy 口径影响不大，但会拖长墙钟并推高失败率）。
# · s2grag 官方只支持 hotpotqa / 2wiki；源码未就位时会打「不可用」直接退出。
# ============================================================================

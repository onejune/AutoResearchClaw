#!/bin/bash
# run_experiments.sh
# 批量并行运行实验，支持配置最大并行数。
#
# 用法：
#   bash scripts/run_experiments.sh                    # 运行所有实验，默认并行 2
#   bash scripts/run_experiments.sh --parallel 3       # 最大并行 3
#   bash scripts/run_experiments.sh --exps "001 003"   # 只运行指定实验
#   bash scripts/run_experiments.sh --dry-run          # 只打印命令，不执行

set -euo pipefail

# ── 默认参数 ──────────────────────────────────────────────────────────────
MAX_PARALLEL=2
DRY_RUN=false
SELECTED_EXPS=""
LOG_DIR="./logs"
CONF_DIR="./conf/experiments"

# ── 解析参数 ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --exps)     SELECTED_EXPS="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

# ── 收集实验配置 ──────────────────────────────────────────────────────────
if [[ -n "$SELECTED_EXPS" ]]; then
    CONF_FILES=()
    for exp_id in $SELECTED_EXPS; do
        f=$(ls "$CONF_DIR"/exp_*"${exp_id}"*.yaml 2>/dev/null | head -1)
        [[ -n "$f" ]] && CONF_FILES+=("$f") || echo "[WARN] 未找到实验 $exp_id"
    done
else
    mapfile -t CONF_FILES < <(ls "$CONF_DIR"/exp_*.yaml 2>/dev/null | sort)
fi

if [[ ${#CONF_FILES[@]} -eq 0 ]]; then
    echo "[ERROR] 没有找到任何实验配置文件"
    exit 1
fi

echo "=========================================="
echo "实验列表（共 ${#CONF_FILES[@]} 个，最大并行 ${MAX_PARALLEL}）："
for f in "${CONF_FILES[@]}"; do echo "  - $f"; done
echo "=========================================="

# ── 并行执行 ──────────────────────────────────────────────────────────────
running=0
pids=()
conf_map=()  # pid → conf_file 映射

run_exp() {
    local conf="$1"
    local exp_name
    exp_name=$(basename "$conf" .yaml)
    local log_file="$LOG_DIR/${exp_name}.log"

    if $DRY_RUN; then
        echo "[DRY-RUN] python scripts/train.py --conf $conf > $log_file 2>&1"
        return 0
    fi

    echo "[START] $exp_name → $log_file"
    python scripts/train.py --conf "$conf" > "$log_file" 2>&1 &
    echo $!
}

wait_one() {
    # 等待任意一个子进程结束
    local idx=0
    while true; do
        for i in "${!pids[@]}"; do
            local pid="${pids[$i]}"
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" && status=0 || status=$?
                local conf="${conf_map[$i]}"
                local exp_name
                exp_name=$(basename "$conf" .yaml)
                if [[ $status -eq 0 ]]; then
                    echo "[DONE ✓] $exp_name (pid=$pid)"
                else
                    echo "[FAIL ✗] $exp_name (pid=$pid, exit=$status)"
                fi
                unset 'pids[$i]'
                unset 'conf_map[$i]'
                pids=("${pids[@]}")
                conf_map=("${conf_map[@]}")
                return
            fi
        done
        sleep 2
    done
}

for conf in "${CONF_FILES[@]}"; do
    # 等待直到有空位
    while [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; do
        wait_one
    done

    pid=$(run_exp "$conf")
    if [[ -n "$pid" ]]; then
        pids+=("$pid")
        conf_map+=("$conf")
    fi
done

# 等待所有剩余进程
while [[ ${#pids[@]} -gt 0 ]]; do
    wait_one
done

echo "=========================================="
echo "所有实验完成！查看结果："
echo "  cat experiments/leaderboard.json"
echo "=========================================="

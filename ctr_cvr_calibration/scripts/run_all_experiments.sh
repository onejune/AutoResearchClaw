#!/bin/bash
# 运行所有校准实验

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "CTR/CVR 预估校准研究 - 实验运行"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. 数据质量检查
echo -e "${BLUE}[1/6] 数据质量检查${NC}"
python scripts/check_data_quality.py
echo ""

# 2. 创建输出目录
echo -e "${BLUE}[2/6] 创建输出目录${NC}"
mkdir -p results logs checkpoints results/figures
echo ""

# 3. 运行实验
echo -e "${BLUE}[3/6] 运行实验${NC}"
echo ""

experiments=(
    "exp01_baseline"
    "exp02_temperature"
    "exp03_isotonic"
    "exp04_focal_loss"
    "exp05_multitask"
)

for exp in "${experiments[@]}"; do
    exp_file="experiments/${exp}.py"
    
    if [ -f "$exp_file" ]; then
        echo -e "${GREEN}运行 $exp ...${NC}"
        python "$exp_file" 2>&1 | tee "logs/${exp}.log"
        echo ""
    else
        echo -e "⚠️  $exp_file 不存在，跳过"
        echo ""
    fi
done

# 4. 汇总结果
echo -e "${BLUE}[4/6] 汇总结果${NC}"
if [ -f "scripts/analyze_results.py" ]; then
    python scripts/analyze_results.py
else
    echo "⚠️  analyze_results.py 不存在，跳过"
fi
echo ""

# 5. 生成报告
echo -e "${BLUE}[5/6] 生成报告${NC}"
echo "结果保存在 results/ 目录"
echo ""

# 6. 完成
echo "=========================================="
echo -e "${GREEN}✅ 所有实验完成！${NC}"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  - 实验报告: results/summary.md"
echo "  - 训练日志: logs/*.log"
echo "  - 可视化:   results/figures/"

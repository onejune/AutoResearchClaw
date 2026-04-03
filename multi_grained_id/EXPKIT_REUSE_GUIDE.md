# expkit 复用指南

> 基于改造后的实验管理系统（v2.0），支持多项目并行、零硬编码、集中配置。

---

## 一、复用到新项目（4步）

**第1步：复制核心文件**

```bash
NEW_PROJECT=/mnt/workspace/open_research/autoresearch/your_project

# 复制 expkit 包
cp -r multi_grained_id/src/expkit  $NEW_PROJECT/src/
cp multi_grained_id/experiment_manager.py  $NEW_PROJECT/
cp multi_grained_id/scripts/expctl  $NEW_PROJECT/scripts/
cp multi_grained_id/scripts/manage_experiments.sh  $NEW_PROJECT/scripts/
cp multi_grained_id/scripts/experiment_queue_manager.py  $NEW_PROJECT/scripts/
chmod +x $NEW_PROJECT/scripts/expctl $NEW_PROJECT/scripts/manage_experiments.sh
```

**第2步：复制并修改配置文件**

```bash
cp multi_grained_id/expkit.conf  $NEW_PROJECT/expkit.conf
```

打开 `expkit.conf`，通常只需要改一行（其余保持默认即可）：

```ini
# 留空自动用目录名，或手动指定
PROJECT_NAME=my_project_name
```

**第3步：创建空的实验配置**

```bash
cat > $NEW_PROJECT/experiments_config.yaml << 'EOF'
version: "1.0"
experiments: {}
EOF
```

**第4步：导出并启动**

```bash
cd $NEW_PROJECT
./scripts/expctl export   # 生成 experiments_config.json
./scripts/expctl start    # 启动守护进程
```

---

## 二、添加实验

**方式A：命令行添加（推荐，单个实验）**

```bash
./scripts/expctl add \
    --id exp001 \
    --name "DeepFM Baseline" \
    --script "scripts/run_exp001_deepfm.py" \
    --result-dir "exp001_deepfm" \
    --priority 1 \
    --description "DeepFM baseline 对比" \
    --use-gpu

# 添加后导出并重启守护进程生效
./scripts/expctl export
./scripts/expctl restart
```

**方式B：直接编辑 YAML（批量添加时更方便）**

```yaml
# experiments_config.yaml
version: "1.0"
experiments:
  exp001:
    name: "DeepFM Baseline"
    script: "scripts/run_exp001_deepfm.py"
    result_dir: "exp001_deepfm"
    priority: 1
    description: "DeepFM baseline 对比"
    use_gpu: true
    completed: false
    max_restarts: 3
  exp002:
    name: "WideDeep"
    script: "scripts/run_exp002_widedeep.py"
    result_dir: "exp002_widedeep"
    priority: 2
    description: "WideDeep 对比"
    use_gpu: true
    completed: false
    max_restarts: 3
```

然后：

```bash
./scripts/expctl export && ./scripts/expctl restart
```

---

## 三、实验脚本的唯一约定

脚本结束时必须把结果写到指定位置，守护进程通过检测该文件判断实验是否完成：

```python
import json
from pathlib import Path

result_dir = Path("results/exp001_deepfm")
result_dir.mkdir(parents=True, exist_ok=True)

(result_dir / "results.json").write_text(json.dumps({
    "auc": 0.8512,       # ← 有此字段，守护进程才会标记完成
    "logloss": 0.3821,
    "training_time": 120,
}))
```

> **如果你的指标不是 AUC**，在 `expkit.conf` 里改一行即可：
> ```ini
> COMPLETION_KEYS=auc,AUC,accuracy,loss_final,rmse
> ```
> 逗号分隔，任意一个字段存在即视为完成。

---

## 四、expkit.conf 完整说明

```ini
# 项目标识（留空自动用目录名）
PROJECT_NAME=

# 配置文件路径（相对项目根目录）
EXPERIMENTS_YAML=experiments_config.yaml
EXPERIMENTS_JSON=experiments_config.json

# 运行时文件路径
DAEMON_LOG=logs/daemon.log
DAEMON_PID=logs/daemon.pid
EXP_LOGS_DIR=logs/experiments
STATE_FILE=experiment_manager_state.json

# 结果目录
RESULTS_DIR=results

# 守护进程参数
DAEMON_INTERVAL=30      # 检查间隔（秒）
MAX_RESTARTS=3          # 实验最大重启次数

# 资源调度阈值
GPU_IDLE_UTIL=10        # GPU 空闲判定利用率 (%)
GPU_IDLE_MEM_MB=500     # GPU 空闲判定显存占用 (MB)
CPU_IDLE_UTIL=70        # CPU 空闲判定利用率 (%)

# 完成检测字段（results.json 中任一存在即视为完成）
COMPLETION_KEYS=auc,AUC,accuracy,loss_final
```

---

## 五、日常操作速查

```bash
./scripts/expctl status          # 查看所有实验状态（含项目名）
./scripts/expctl list            # 列出实验队列
./scripts/expctl list --status pending    # 只看待运行
./scripts/expctl list --status completed  # 只看已完成
./scripts/expctl log             # 查看守护进程日志
./scripts/expctl dashboard       # 实时仪表盘（5秒刷新）

./scripts/expctl complete exp001 # 手动标记完成
./scripts/expctl rerun exp002    # 重跑某个实验
./scripts/expctl kill exp003     # 强制停止某个实验

./scripts/expctl start           # 启动守护进程
./scripts/expctl stop            # 停止守护进程
./scripts/expctl restart         # 重启守护进程
```

---

## 六、多项目并行说明

改造后每个项目完全隔离，互不干扰：

```
project_a/
├── expkit.conf                  ← 各自独立配置
├── logs/
│   ├── daemon.log               ← 各自独立日志
│   ├── daemon.pid               ← 各自独立 PID
│   └── experiments/expXXX.log
└── experiment_manager_state.json

project_b/
├── expkit.conf
├── logs/
│   ├── daemon.log
│   ├── daemon.pid
│   └── experiments/expXXX.log
└── experiment_manager_state.json
```

`expctl status` 输出头部会显示当前项目名，方便区分：

```
================================================================================
📊 实验管理器状态  [project_a]
================================================================================
```

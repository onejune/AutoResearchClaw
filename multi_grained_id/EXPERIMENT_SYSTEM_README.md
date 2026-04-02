# 实验管理系统 v2.0 - 完整指南

## 🎯 系统概述

这是一个**全自动化的实验管理系统**，提供：

- ✅ **自动队列管理** - 动态增删改查实验
- 🔄 **实时监控** - 检测运行状态和进度
- 🛡️ **自动恢复** - 失败实验自动重启
- 📊 **智能调度** - GPU/CPU 资源优化分配
- 🚀 **一键控制** - 统一命令行接口

---

## 📁 系统架构

```
/mnt/workspace/open_research/autoresearch/multi_grained_id/
├── experiment_manager.py              # 核心管理器（监控 + 调度）
├── experiments_config.json            # 实验配置（JSON 格式）
├── experiments_config.yaml            # 实验配置（YAML 格式）
├── experiment_manager_state.json      # 运行时状态
├── EXPERIMENT_MANAGER.md              # 本文件
├── EXPERIMENT_SYSTEM_README.md        # 快速入门
└── scripts/
    ├── expctl                         # 🔥 统一入口（推荐使用）
    ├── manage_experiments.sh          # 守护进程控制
    ├── experiment_queue_manager.py    # 队列管理工具
    └── run_expXXX_*.py                # 实验脚本
```

---

## 🚀 快速开始

### 1. 查看当前状态

```bash
cd /mnt/workspace/open_research/autoresearch/multi_grained_id
./scripts/expctl status
```

### 2. 启动守护进程（推荐）

```bash
./scripts/expctl start
```

守护进程会：
- 每 30 秒检查一次实验状态
- 自动启动待运行的实验
- 自动重启失败的实验
- 智能分配 GPU/CPU 资源

### 3. 实时仪表盘

```bash
./scripts/expctl dashboard
# 按 Ctrl+C 退出
```

---

## 📋 队列管理

### 列出所有实验

```bash
# 查看所有实验
./scripts/expctl list

# 只看待运行的
./scripts/expctl list --status pending

# 只看已完成的
./scripts/expctl list --status completed
```

### 添加新实验

```bash
./scripts/expctl add \
    --id exp009 \
    --name "DCN" \
    --script "scripts/run_exp009_dcn.py" \
    --result-dir "exp009_dcn" \
    --priority 9 \
    --description "DCN 模型对比" \
    --no-gpu  # 或使用 --use-gpu
```

### 更新实验配置

```bash
# 调整优先级
./scripts/expctl update --id exp009 --priority 5

# 标记为完成
./scripts/expctl complete exp003

# 或标记为未完成（重新运行）
python scripts/experiment_queue_manager.py update --id exp003 --not-completed
```

### 移除实验

```bash
./scripts/expctl remove --id exp009
```

### 导出配置

```bash
# 将 YAML 配置导出为 JSON（供 experiment_manager.py 使用）
./scripts/expctl export
```

---

## 🎮 实验控制

### 手动启动实验

```bash
# 自动选择空闲资源
./scripts/expctl run exp001
```

### 停止实验

```bash
./scripts/expctl kill exp001
```

### 重启实验

```bash
./scripts/expctl rerun exp001
```

### 标记完成

```bash
# 当实验提前完成或不需要继续时
./scripts/expctl complete exp001
```

---

## 🛡️ 守护进程管理

### 启动/停止/重启

```bash
./scripts/expctl start     # 启动
./scripts/expctl stop      # 停止
./scripts/expctl restart   # 重启
```

### 查看日志

```bash
# 最近 50 行
./scripts/expctl log

# 实时跟踪
tail -f /tmp/experiment_manager.log
```

### 检查状态

```bash
./scripts/manage_experiments.sh status
```

---

## 📊 监控面板示例

```
================================================================================
📊 实验管理器状态
================================================================================

🖥️  GPU 状态:
   GPU 0: 🔴 忙碌 (利用率:85.3%, 显存:4096MB)
   GPU 1: ✅ 空闲 (利用率:5.2%, 显存:128MB)

💻  CPU 状态：✅ 空闲 (利用率:25.4%)

🧪 实验队列:
--------------------------------------------------------------------------------
ID         名称                 状态         进度                 备注
--------------------------------------------------------------------------------
exp001     AutoEmb            🔄 运行中    45.2% (2882/6378)    自动化维度分配
exp002     DDS                🔄 运行中    loss=0.4408          数据分布搜索
exp003     Hierarchical v2    ✅ 已完成    -                    层次化 Embedding
exp004     MetaEmb            ⏸️  待运行    -                    元学习冷启动
exp005     Contrastive        ⏸️  待运行    -                    对比学习增强
exp006     FiBiNET            ⏸️  待运行    -                    特征交互
exp007     Combined           ⏸️  待运行    -                    综合方案
exp008     DeepFM             ⏸️  待运行    -                    DeepFM 模型对比
================================================================================
```

---

## ⚙️ 配置说明

### experiments_config.yaml

```yaml
version: "1.0"
experiments:
  exp001:
    name: "AutoEmb"
    script: "scripts/run_exp001_autoemb_v3.py"
    result_dir: "exp001_autoemb"
    priority: 1              # 数字越小优先级越高
    description: "自动化维度分配"
    completed: false         # 是否已完成
    max_restarts: 3          # 最大重启次数
    use_gpu: true            # 是否使用 GPU
    min_memory_mb: 4096      # 最小显存要求
    tags: ["embedding", "auto"]
```

### 优先级规则

- **数字越小，优先级越高**
- 高优先级实验优先获得资源
- 建议：重要实验设为 1-3，普通实验 4-10

---

## 🔄 自动化流程

### 守护进程工作流

```
启动守护进程
    ↓
每 30 秒循环:
    1. 检查 GPU/CPU 状态
    2. 监控运行中的实验
       ├─ 发现崩溃 → 自动重启（最多 3 次）
       ├─ 发现完成 → 标记完成，释放资源
       └─ 正常运行 → 继续监控
    3. 获取待运行实验列表（按优先级排序）
    4. 分配空闲资源
       ├─ 有 GPU → 分配到 GPU
       └─ 无 GPU 但 CPU 空闲 → 分配到 CPU
    5. 启动新实验
    ↓
重复...
```

### 失败恢复机制

```
实验意外停止
    ↓
检测到进程不存在
    ↓
restart_count < max_restarts?
    ├─ 是 → 等待 5 秒 → 重启实验 → restart_count++
    └─ 否 → 标记为 failed → 记录错误原因
```

---

## 📝 最佳实践

### 1. 始终使用守护进程

不要手动启动实验，让守护进程统一管理：

```bash
# ✅ 推荐
./scripts/expctl start

# ❌ 不推荐
nohup python scripts/run_exp001.py &
```

### 2. 合理设置优先级

```
优先级 1-3: 核心实验，必须跑
优先级 4-7: 重要实验，尽快跑
优先级 8+: 探索性实验，有空再跑
```

### 3. 定期检查状态

```bash
# 每小时查看一次
watch -n 3600 './scripts/expctl status'

# 或设置 cron
0 * * * * cd /path/to/project && ./scripts/expctl status >> log.txt
```

### 4. 及时标记完成

实验完成后及时标记，避免重复运行：

```bash
./scripts/expctl complete exp003
```

### 5. 备份配置文件

系统会自动备份，但重要修改前建议手动备份：

```bash
cp experiments_config.yaml experiments_config.yaml.bak
```

---

## 🔧 故障排查

### 实验无法启动

```bash
# 1. 检查脚本是否存在
ls scripts/run_expXXX.py

# 2. 检查依赖
python -c "import torch; import transformers"

# 3. 查看详细日志
cat /tmp/exp_logs/expXXX.log

# 4. 手动测试
python scripts/run_expXXX.py
```

### 守护进程未运行

```bash
# 检查 PID
cat /tmp/experiment_manager.pid

# 查看进程
ps aux | grep experiment_manager

# 重新启动
./scripts/expctl restart
```

### GPU 未被识别

```bash
# 检查 nvidia-smi
nvidia-smi

# 检查环境变量
echo $CUDA_VISIBLE_DEVICES

# 重启守护进程
./scripts/expctl restart
```

### 实验卡住

```bash
# 强制杀死
./scripts/expctl kill expXXX

# 重启
./scripts/expctl rerun expXXX

# 如果反复失败，检查日志
tail -100 /tmp/exp_logs/expXXX.log
```

---

## 📈 扩展功能

### 添加自定义监控指标

编辑 `experiment_manager.py` 的 `_parse_progress_from_log` 方法：

```python
def _parse_progress_from_log(self, log_file: Path) -> str:
    # 添加自定义解析逻辑
    if 'auc=' in line:
        match = re.search(r'auc=([0-9.]+)', line)
        if match:
            return f"AUC={match.group(1)}"
```

### 集成到 CI/CD

```bash
# GitHub Actions 示例
- name: Run Experiments
  run: |
    cd /mnt/workspace/open_research/autoresearch/multi_grained_id
    ./scripts/expctl start
    sleep 3600  # 运行 1 小时
    ./scripts/expctl status
```

### 发送通知

编辑 `experiment_manager.py`，在实验完成时添加通知：

```python
if self.check_experiment_completed(exp_id):
    print(f"✅ {exp_id} 已完成!")
    # 发送邮件/钉钉/企业微信通知
    self._send_notification(exp_id, "completed")
```

---

## 📞 支持

遇到问题请：

1. 查看日志：`./scripts/expctl log`
2. 检查状态：`./scripts/expctl status`
3. 重启守护进程：`./scripts/expctl restart`
4. 联系管理员

---

**版本**: v2.0  
**最后更新**: 2026-04-02  
**维护者**: 牛顿 🍎

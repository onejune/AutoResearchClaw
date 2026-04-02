# 实验管理系统使用指南

## 概述

实验管理系统提供自动化的实验队列管理功能，包括：
- ✅ 自动启动待运行的实验
- 🔄 实时监控实验状态
- 🔄 自动重启失败的实验
- 📊 资源智能调度（GPU/CPU）
- 📝 完整的状态记录和日志

## 快速开始

### 1. 查看当前状态

```bash
cd /mnt/workspace/open_research/autoresearch/multi_grained_id
python experiment_manager.py --status
```

### 2. 启动守护进程（推荐）

```bash
# 使用脚本启动
./scripts/manage_experiments.sh start

# 或直接运行
nohup python experiment_manager.py --daemon --interval 30 > /tmp/exp_manager.log 2>&1 &
```

守护进程会：
- 每 30 秒检查一次实验状态
- 自动启动待运行的实验
- 自动重启意外停止的实验（最多 3 次）
- 智能分配 GPU/CPU 资源

### 3. 管理守护进程

```bash
# 查看状态
./scripts/manage_experiments.sh status

# 查看日志
./scripts/manage_experiments.sh log

# 停止
./scripts/manage_experiments.sh stop

# 重启
./scripts/manage_experiments.sh restart
```

### 4. 手动控制单个实验

```bash
# 启动指定实验（自动选择空闲资源）
python experiment_manager.py --start exp001

# 停止指定实验
python experiment_manager.py --stop exp001

# 重启指定实验
python experiment_manager.py --restart exp001
```

## 实验队列配置

当前配置的实验：

| ID | 名称 | 描述 | 优先级 |
|----|------|------|--------|
| exp001 | AutoEmb | 自动化维度分配 | 1 |
| exp002 | DDS | 数据分布搜索 | 2 |
| exp003 | Hierarchical v2 | 层次化 Embedding (已完成) | 3 |
| exp004 | MetaEmb | 元学习冷启动 | 4 |
| exp005 | Contrastive | 对比学习增强 | 5 |
| exp006 | FiBiNET | 特征交互 | 6 |
| exp007 | Combined | 综合方案 | 7 |

**添加新实验**: 编辑 `experiment_manager.py` 中的 `self.experiments` 字典。

## 资源调度策略

### GPU 优先
- 空闲 GPU 判断标准：利用率 < 10% 且 显存 < 500MB
- 优先将实验分配到空闲 GPU
- 多 GPU 环境会自动利用所有空闲卡

### CPU 回退
- 当所有 GPU 都忙碌时，自动降级到 CPU
- CPU 空闲判断标准：利用率 < 70%

### 优先级队列
- 按 `priority` 字段排序，数字越小优先级越高
- 高优先级实验会先获得资源

## 失败恢复机制

### 自动重启
- 检测到运行中的实验意外停止
- 自动重启，最多 3 次
- 每次重启前等待 5 秒

### 错误记录
- 失败原因记录在状态文件中
- 达到最大重启次数后标记为"failed"

## 状态文件

### 位置
`/mnt/workspace/open_research/autoresearch/multi_grained_id/experiment_manager_state.json`

### 内容示例
```json
{
  "started_at": "2026-04-02T06:40:00",
  "updated_at": "2026-04-02T06:45:00",
  "experiments": {
    "exp001": {
      "status": "running",
      "pid": 12345,
      "start_time": "2026-04-02T06:40:00",
      "restart_count": 0
    },
    "exp003": {
      "status": "completed",
      "end_time": "2026-04-02T06:30:00"
    }
  }
}
```

## 日志文件

### 位置
- 实验日志：`/tmp/exp_logs/{exp_id}.log`
- 管理器日志：`/tmp/experiment_manager.log`

### 查看日志
```bash
# 查看某个实验的日志
tail -f /tmp/exp_logs/exp001.log

# 查看管理器日志
tail -f /tmp/experiment_manager.log
```

## 监控面板

运行以下命令查看实时状态：

```bash
python experiment_manager.py --status
```

输出示例：
```
================================================================================
📊 实验管理器状态
================================================================================

🖥️  GPU 状态:
   GPU 0: ✅ 空闲 (利用率:5.2%, 显存:128MB)
   GPU 1: 🔴 忙碌 (利用率:85.3%, 显存:4096MB)

💻  CPU 状态：✅ 空闲 (利用率:25.4%)

🧪 实验队列:
--------------------------------------------------------------------------------
ID         名称                 状态         进度                 备注
--------------------------------------------------------------------------------
exp001     AutoEmb            🔄 运行中    45.2% (2882/6378)    自动化维度分配
exp002     DDS                🔄 运行中    loss=0.4408          数据分布搜索
exp003     Hierarchical v2    ✅ 已完成    -                    层次化 Embedding
exp004     MetaEmb            ⏸️  待运行    -                    元学习冷启动
...
```

## 高级用法

### 自定义检查间隔
```bash
# 每 60 秒检查一次
python experiment_manager.py --daemon --interval 60
```

### 修改最大重启次数
编辑 `experiment_manager.py` 中的 `monitor_and_restart` 方法：
```python
self.monitor_and_restart(max_restarts=5)  # 改为 5 次
```

### 集成到 crontab
```bash
# 每天早上 8 点检查实验状态
0 8 * * * cd /mnt/workspace/open_research/autoresearch/multi_grained_id && \
          python experiment_manager.py --status >> /var/log/exp_manager.log 2>&1
```

## 故障排查

### 实验无法启动
1. 检查脚本是否存在：`ls scripts/run_expXXX.py`
2. 检查依赖：`python -c "import torch; import transformers"`
3. 查看详细日志：`cat /tmp/exp_logs/expXXX.log`

### 守护进程未运行
```bash
# 检查 PID 文件
cat /tmp/experiment_manager.pid

# 手动启动
./scripts/manage_experiments.sh start
```

### GPU 未被识别
```bash
# 检查 nvidia-smi
nvidia-smi

# 检查 CUDA 环境变量
echo $CUDA_VISIBLE_DEVICES
```

## 最佳实践

1. **始终使用守护进程模式** - 避免手动管理的复杂性
2. **定期检查日志** - 及时发现异常
3. **合理设置优先级** - 重要实验优先运行
4. **监控资源使用** - 避免资源争用
5. **备份状态文件** - 防止数据丢失

## 联系与支持

遇到问题请：
1. 查看日志文件
2. 检查状态文件
3. 重启守护进程
4. 联系管理员

---

最后更新：2026-04-02

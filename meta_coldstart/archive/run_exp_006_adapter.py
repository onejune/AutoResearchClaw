#!/usr/bin/env python3
"""
Adapter Tuning 实验
"""
import os, sys, logging, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.config       import Config
from src.data         import TaskBuilder
from src.baseline_fixed import train_global_model
from src.adapter      import create_adapted_model, train_adapter
from src.evaluate     import evaluate_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

K_SHOT       = 100
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "results", "exp_006_adapter.md")


def eval_adapted(adapter_func, test_tasks, features, device):
    """评估 adapter 方法在多个任务上的表现"""
    aucs, pcocs = [], []
    for task in test_tasks:
        try:
            # 获取 support set 进行适配
            support_loader = task.support_loader(features, batch_size=min(64, len(task.support_df)))
            query_loader   = task.query_loader(features)
            if query_loader is None or support_loader is None:
                continue
            
            # 创建适配后的模型
            adapted_model = adapter_func(task, support_loader)
            
            # 在 query set 上评估
            r = evaluate_model(adapted_model, query_loader, device)
            if not np.isnan(r["auc"]):
                aucs.append(r["auc"])
                pcocs.append(r["pcoc"])
        except Exception as e:
            logger.warning(f"Failed to evaluate task {task.campaignset_id}: {e}")
            continue
    return {"mean_auc": float(np.mean(aucs)) if aucs else 0.5,
            "mean_pcoc": float(np.mean(pcocs)) if pcocs else 1.0,
            "n_tasks": len(aucs)}


def main():
    cfg    = Config(k_shot=K_SHOT)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    builder = TaskBuilder(cfg)
    train_tasks, test_tasks = builder.build_tasks(k_shot=K_SHOT)
    features    = builder.features
    vocab_sizes = builder.vocab_sizes

    # 首先训练一个预训练模型
    logger.info("Training pretrained model...")
    t0 = time.time()
    pretrained_model = train_global_model(builder, cfg, device)
    pretrain_time = time.time() - t0
    logger.info(f"Pretrained model ready (time: {pretrain_time:.1f}s)")

    # Adapter 方法
    def adapt_with_adapter(task, support_loader):
        """使用 adapter 进行任务适配"""
        # 创建带 adapter 的模型
        adapter_model = create_adapted_model(pretrained_model, k_shot=K_SHOT, adapter_bottleneck=32)
        
        # 训练 adapter 参数
        trained_adapter = train_adapter(adapter_model, support_loader, epochs=5, lr=1e-2)
        return trained_adapter

    logger.info("Evaluating Adapter method...")
    t0 = time.time()
    r = eval_adapted(adapt_with_adapter, test_tasks, features, device)
    adapter_time = time.time() - t0
    r["train_time"] = pretrain_time + adapter_time
    results = {"Adapter": r}
    
    logger.info(f"Adapter: AUC={r['mean_auc']:.4f}, PCOC={r['mean_pcoc']:.3f}")

    write_results(results, K_SHOT)
    logger.info(f"Results saved to {RESULTS_FILE}")


def write_results(results: dict, k_shot: int):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    lines = [
        "# exp_006: Adapter Tuning 实验\n\n",
        f"K-shot = {k_shot}\n\n",
        "## 汇总\n\n",
        "| 方法 | Mean AUC | Mean PCOC | 训练时间 | 说明 |\n",
        "|------|----------|-----------|---------|------|\n",
    ]
    descs = {
        "Adapter": "参数高效微调，只训练 adapter 层",
    }
    for name, r in results.items():
        auc  = r.get("mean_auc",  float("nan"))
        pcoc = r.get("mean_pcoc", float("nan"))
        t    = r.get("train_time", 0)
        desc = descs.get(name, "")
        lines.append(f"| {name} | {auc:.4f} | {pcoc:.3f} | {t:.0f}s | {desc} |\n")

    lines += ["\n## 结论\n\n", "- （实验完成后补充）\n"]
    with open(RESULTS_FILE, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
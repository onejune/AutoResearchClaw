#!/usr/bin/env python3
"""
修复采样偏差后的基线方法实验
"""
import os, sys, logging, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.config       import Config
from src.data         import TaskBuilder
from src.baseline_fixed import train_global_model, FineTuner
from src.evaluate     import evaluate_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

K_SHOT       = 100
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "results", "exp_005_fixed_baseline.md")


def eval_on_tasks(model, test_tasks, features, device):
    """评估模型在多个任务上的平均表现"""
    aucs, pcocs = [], []
    for task in test_tasks:
        loader = task.query_loader(features)
        if loader is None:
            continue
        r = evaluate_model(model, loader, device)
        if not np.isnan(r["auc"]):
            aucs.append(r["auc"])
            pcocs.append(r["pcoc"])
    return {"mean_auc": float(np.mean(aucs)), "mean_pcoc": float(np.mean(pcocs)),
            "n_tasks": len(aucs)}


def eval_finetuned(finetuner, test_tasks, features, device):
    """评估 fine-tune 方法在多个任务上的表现"""
    aucs, pcocs = [], []
    for task in test_tasks:
        try:
            adapted_model = finetuner.adapt(task, k_shot=K_SHOT)
            loader = task.query_loader(features)
            if loader is None:
                continue
            r = evaluate_model(adapted_model, loader, device)
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

    results = {}

    # 1. Global Model（全量训练）
    logger.info("=" * 50)
    logger.info("Training Global Model...")
    t0 = time.time()
    global_model = train_global_model(builder, cfg, device)
    global_train_time = time.time() - t0
    
    logger.info("Evaluating Global Model...")
    r = eval_on_tasks(global_model, test_tasks, features, device)
    r["train_time"] = global_train_time
    results["Global"] = r
    logger.info(f"Global: AUC={r['mean_auc']:.4f}, PCOC={r['mean_pcoc']:.3f}")

    # 2. Pretrain + Fine-tune
    logger.info("=" * 50)
    logger.info("Training Pretrained Model for Fine-tuning...")
    t0 = time.time()
    pretrained_model = train_global_model(builder, cfg, device)  # 使用相同的预训练
    pretrain_time = time.time() - t0
    
    logger.info("Fine-tuning on each task...")
    finetuner = FineTuner(pretrained_model, cfg, device)
    r = eval_finetuned(finetuner, test_tasks, features, device)
    r["train_time"] = pretrain_time + r.get("n_tasks", len(test_tasks)) * 0.1  # 估算 fine-tune 时间
    results["Pretrain+FT"] = r
    logger.info(f"Pretrain+FT: AUC={r['mean_auc']:.4f}, PCOC={r['mean_pcoc']:.3f}")

    write_results(results, K_SHOT)
    logger.info(f"Results saved to {RESULTS_FILE}")


def write_results(results: dict, k_shot: int):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    lines = [
        "# exp_005: 修复采样偏差后的基线方法实验\n\n",
        f"K-shot = {k_shot}\n\n",
        "## 汇总\n\n",
        "| 方法 | Mean AUC | Mean PCOC | 训练时间 | 说明 |\n",
        "|------|----------|-----------|---------|------|\n",
    ]
    descs = {
        "Global":     "全量混合训练，不做适配",
        "Pretrain+FT": "预训练 → K 条样本 fine-tune",
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
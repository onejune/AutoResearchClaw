#!/usr/bin/env python3
"""
exp_001: 基线实验

对比方法：
  1. Global Model    - 全量混合训练，不做适配
  2. Pretrain+FT     - 全量预训练 → K-shot fine-tune
  3. No-Adapt        - 预训练模型直接推理（不 fine-tune）

结果写入 results/exp_001_baseline.md
"""
import os, sys, logging, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.config   import Config
from src.data     import TaskBuilder
from src.models   import PurchaseModel
from src.baseline import train_global_model, FineTuner
from src.evaluate import evaluate_tasks

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

K_SHOT   = 100
FINETUNE_EPOCHS = 5
FINETUNE_LR     = 1e-3
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "results", "exp_001_baseline.md")


def main():
    cfg    = Config(k_shot=K_SHOT)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── 构建 tasks ──────────────────────────────────────
    builder = TaskBuilder(cfg)
    train_tasks, test_tasks = builder.build_tasks(k_shot=K_SHOT)
    features = builder.features
    logger.info(f"Train tasks: {len(train_tasks)}, Test tasks: {len(test_tasks)}")

    results = {}

    # ── 1. Global Model ─────────────────────────────────
    logger.info("=" * 50)
    logger.info("Training Global Model...")
    t0 = time.time()
    global_model = train_global_model(builder, cfg, device)
    r = evaluate_tasks(global_model, test_tasks, features, device)
    results["Global Model"] = r
    logger.info(f"Global Model: AUC={r['mean_auc']:.4f}, PCOC={r['mean_pcoc']:.3f} "
                f"({time.time()-t0:.0f}s)")

    # ── 2. No-Adapt（预训练直接推理）──────────────────────
    logger.info("=" * 50)
    logger.info("No-Adapt (pretrained, no fine-tune)...")
    r = evaluate_tasks(global_model, test_tasks, features, device)
    results["No-Adapt"] = r
    logger.info(f"No-Adapt: AUC={r['mean_auc']:.4f}, PCOC={r['mean_pcoc']:.3f}")

    # ── 3. Pretrain + Fine-tune ──────────────────────────
    logger.info("=" * 50)
    logger.info(f"Pretrain + Fine-tune (K={K_SHOT}, epochs={FINETUNE_EPOCHS})...")
    finetuner = FineTuner(global_model, cfg, device)
    t0 = time.time()

    ft_aucs, ft_pcocs = [], []
    for task in test_tasks:
        loader = task.query_loader(features)
        if loader is None:
            continue
        adapted = finetuner.adapt(task, k_shot=K_SHOT,
                                  finetune_epochs=FINETUNE_EPOCHS,
                                  finetune_lr=FINETUNE_LR)
        from src.evaluate import evaluate_model
        r_task = evaluate_model(adapted, loader, device)
        if not np.isnan(r_task["auc"]):
            ft_aucs.append(r_task["auc"])
            ft_pcocs.append(r_task["pcoc"])

    ft_result = {
        "mean_auc":  float(np.mean(ft_aucs)),
        "mean_pcoc": float(np.mean(ft_pcocs)),
        "n_tasks":   len(ft_aucs),
    }
    results["Pretrain+FT"] = ft_result
    logger.info(f"Pretrain+FT: AUC={ft_result['mean_auc']:.4f}, "
                f"PCOC={ft_result['mean_pcoc']:.3f} ({time.time()-t0:.0f}s)")

    # ── 写结果 ───────────────────────────────────────────
    write_results(results, K_SHOT)
    logger.info(f"Results saved to {RESULTS_FILE}")


def write_results(results: dict, k_shot: int):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    lines = [
        "# exp_001: 基线实验结果\n",
        f"K-shot = {k_shot}\n\n",
        "## 汇总\n\n",
        "| 方法 | Mean AUC | Mean PCOC | 说明 |\n",
        "|------|----------|-----------|------|\n",
    ]
    descs = {
        "Global Model": "全量混合训练，不做适配",
        "No-Adapt":     "预训练模型直接推理",
        "Pretrain+FT":  f"全量预训练 → {k_shot}-shot fine-tune",
    }
    for name, r in results.items():
        auc  = r.get("mean_auc",  float("nan"))
        pcoc = r.get("mean_pcoc", float("nan"))
        desc = descs.get(name, "")
        lines.append(f"| {name} | {auc:.4f} | {pcoc:.3f} | {desc} |\n")

    lines += [
        "\n## 结论\n\n",
        "- （实验完成后补充）\n",
    ]
    with open(RESULTS_FILE, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()

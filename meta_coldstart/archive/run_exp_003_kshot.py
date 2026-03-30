#!/usr/bin/env python3
"""
exp_003: K-shot 敏感性分析（核心贡献图）

对比所有方法在 K = 10 / 50 / 100 / 500 / 1000 下的 AUC 曲线
结果写入 results/exp_003_kshot.md
"""
import os, sys, logging, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.config       import Config
from src.data         import TaskBuilder
from src.baseline     import train_global_model, FineTuner
from src.meta_learner import MAML, ANIL
from src.evaluate     import evaluate_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

K_LIST       = [10, 50, 100, 500, 1000]
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "results", "exp_003_kshot.md")


def eval_at_k(method_fn, test_tasks, features, device, k):
    """method_fn(task, k) -> adapted model"""
    aucs = []
    for task in test_tasks:
        adapted = method_fn(task, k)
        loader  = task.query_loader(features)
        if loader is None:
            continue
        r = evaluate_model(adapted, loader, device)
        if not np.isnan(r["auc"]):
            aucs.append(r["auc"])
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    cfg    = Config(k_shot=100)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 用最大 K 构建 tasks（保证所有 K 都有足够样本）
    builder = TaskBuilder(cfg)
    train_tasks, test_tasks = builder.build_tasks(k_shot=max(K_LIST))
    features    = builder.features
    vocab_sizes = builder.vocab_sizes

    # ── 预训练全局模型（所有方法共用）──────────────────────
    logger.info("Pre-training global model...")
    global_model = train_global_model(builder, cfg, device)
    finetuner    = FineTuner(global_model, cfg, device)

    # ── Meta-train MAML & ANIL（只训练一次，测试不同 K）────
    logger.info("Meta-training MAML...")
    maml = MAML(vocab_sizes, cfg, device, first_order=False)
    maml.meta_train(train_tasks, features)

    logger.info("Meta-training ANIL...")
    anil = ANIL(vocab_sizes, cfg, device)
    anil.meta_train(train_tasks, features)

    # ── 定义各方法的 adapt 函数 ────────────────────────────
    def global_fn(task, k):
        return global_model   # 不做适配

    def finetune_fn(task, k):
        return finetuner.adapt(task, k_shot=k, finetune_epochs=5)

    def maml_fn(task, k):
        return maml.adapt(task, features, k_shot=k)

    def anil_fn(task, k):
        return anil.adapt(task, features, k_shot=k)

    method_fns = {
        "Global (no adapt)": global_fn,
        "Pretrain+FT":       finetune_fn,
        "MAML":              maml_fn,
        "ANIL":              anil_fn,
    }

    # ── 跑 K-shot 曲线 ─────────────────────────────────────
    curves = {name: {} for name in method_fns}

    for k in K_LIST:
        logger.info(f"\n{'='*50}\nK = {k}")
        for name, fn in method_fns.items():
            auc = eval_at_k(fn, test_tasks, features, device, k)
            curves[name][k] = auc
            logger.info(f"  {name}: AUC={auc:.4f}")

    write_results(curves, K_LIST)
    logger.info(f"Results saved to {RESULTS_FILE}")


def write_results(curves: dict, k_list: list):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    lines = [
        "# exp_003: K-shot 敏感性分析\n\n",
        "## AUC vs K-shot\n\n",
        "| 方法 | " + " | ".join(f"K={k}" for k in k_list) + " |\n",
        "|------|" + "|".join(["------"] * len(k_list)) + "|\n",
    ]
    for name, curve in curves.items():
        row = f"| {name} | " + " | ".join(
            f"{curve.get(k, float('nan')):.4f}" for k in k_list
        ) + " |\n"
        lines.append(row)

    lines += [
        "\n## 分析\n\n",
        "- （实验完成后补充）\n",
        "\n## 结论\n\n",
        "- （实验完成后补充）\n",
    ]
    with open(RESULTS_FILE, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()

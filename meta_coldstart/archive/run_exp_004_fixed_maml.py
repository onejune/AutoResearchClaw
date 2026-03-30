#!/usr/bin/env python3
"""
重新运行 MAML 实验，使用修复后的版本
"""
import os, sys, logging, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.config       import Config
from src.data         import TaskBuilder
from src.meta_learner_fixed import MAML, ANIL, ProtoNet  # 使用修复后的版本
from src.evaluate     import evaluate_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

K_SHOT       = 100
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "results", "exp_004_fixed_maml.md")


def eval_adapted(method, test_tasks, features, device):
    aucs, pcocs = [], []
    for task in test_tasks:
        adapted = method.adapt(task, features, k_shot=K_SHOT)
        loader  = task.query_loader(features)
        if loader is None:
            continue
        r = evaluate_model(adapted, loader, device)
        if not np.isnan(r["auc"]):
            aucs.append(r["auc"])
            pcocs.append(r["pcoc"])
    return {"mean_auc": float(np.mean(aucs)), "mean_pcoc": float(np.mean(pcocs)),
            "n_tasks": len(aucs)}


def main():
    cfg    = Config(k_shot=K_SHOT)
    # 使用更稳定的配置
    cfg.meta_lr = 1e-4
    cfg.inner_lr = 5e-3
    cfg.inner_steps = 2
    cfg.meta_batch_size = 4
    cfg.meta_epochs = 20
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    builder = TaskBuilder(cfg)
    train_tasks, test_tasks = builder.build_tasks(k_shot=K_SHOT)
    features    = builder.features
    vocab_sizes = builder.vocab_sizes

    results = {}
    methods = [
        ("MAML",     MAML(vocab_sizes, cfg, device, first_order=True)),  # 使用 FOMAML
        ("ANIL",     ANIL(vocab_sizes, cfg, device)),
        ("ProtoNet", ProtoNet(vocab_sizes, cfg, device)),
    ]

    for name, method in methods:
        logger.info("=" * 50)
        logger.info(f"Meta-training {name}...")
        t0 = time.time()
        method.meta_train(train_tasks, features)
        train_time = time.time() - t0

        logger.info(f"Evaluating {name}...")
        r = eval_adapted(method, test_tasks, features, device)
        r["train_time"] = train_time
        results[name] = r
        logger.info(f"{name}: AUC={r['mean_auc']:.4f}, PCOC={r['mean_pcoc']:.3f} "
                    f"({train_time:.0f}s)")

    write_results(results, K_SHOT)
    logger.info(f"Results saved to {RESULTS_FILE}")


def write_results(results: dict, k_shot: int):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    lines = [
        "# exp_004: 修复后 Meta-Learning 实验结果\n\n",
        f"K-shot = {k_shot}\n\n",
        "## 汇总\n\n",
        "| 方法 | Mean AUC | Mean PCOC | 训练时间 | 说明 |\n",
        "|------|----------|-----------|---------|------|\n",
    ]
    descs = {
        "MAML":     "一阶近似 MAML（FOMAML），计算更高效",
        "ANIL":     "只在 head 层做 inner loop",
        "ProtoNet": "原型网络，embedding 距离分类",
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
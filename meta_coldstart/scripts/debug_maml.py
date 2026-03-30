#!/usr/bin/env python3
"""
MAML 调试脚本

专门用于调试 MAML 训练不稳定的问题，添加详细的日志和梯度监控。
"""
import os, sys, logging, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.config       import Config
from src.data         import TaskBuilder
from src.meta_learner import MAML, ANIL
from src.evaluate     import evaluate_model

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

K_SHOT       = 100
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "results", "debug_maml.md")


def debug_gradients(model, tag=""):
    """打印模型梯度统计"""
    total_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            if param_count <= 5:  # 只打印前5个参数的梯度
                logger.debug(f"  {tag}.{name}: grad_norm={param_norm:.6f}")
    total_norm = total_norm ** (1. / 2)
    logger.debug(f"  {tag}.total_grad_norm: {total_norm:.6f}")


def eval_adapted(method, test_tasks, features, device):
    aucs, pcocs = [], []
    for task in test_tasks[:5]:  # 只评估前5个任务以加速
        adapted = method.adapt(task, features, k_shot=K_SHOT)
        loader  = task.query_loader(features)
        if loader is None:
            continue
        r = evaluate_model(adapted, loader, device)
        if not np.isnan(r["auc"]) and r["auc"] > 0.3:  # 过滤异常值
            aucs.append(r["auc"])
            pcocs.append(r["pcoc"])
    return {"mean_auc": float(np.mean(aucs)) if aucs else 0.5,
            "mean_pcoc": float(np.mean(pcocs)) if pcocs else 1.0,
            "n_tasks": len(aucs)}


def main():
    cfg    = Config(k_shot=K_SHOT)
    # 调试配置
    cfg.meta_lr = 1e-4
    cfg.inner_lr = 1e-3
    cfg.inner_steps = 3
    cfg.meta_batch_size = 2  # 减少batch size以便调试
    cfg.meta_epochs = 10
    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    builder = TaskBuilder(cfg)
    train_tasks, test_tasks = builder.build_tasks(k_shot=K_SHOT)
    
    # 为了调试，只取少量tasks
    train_tasks = train_tasks[:10]
    test_tasks = test_tasks[:5]
    
    features    = builder.features
    vocab_sizes = builder.vocab_sizes

    # 初始化 MAML
    maml = MAML(vocab_sizes, cfg, device, first_order=True)  # 使用 FOMAML 更稳定
    logger.info(f"MAML initialized with {sum(p.numel() for p in maml.model.parameters()):,} parameters")

    # 训练循环，添加详细日志
    logger.info("=" * 50)
    logger.info("Starting MAML training with detailed logging...")
    
    for epoch in range(cfg.meta_epochs):
        # 随机采样 task
        import random
        batch_tasks = random.sample(train_tasks,
                                    min(cfg.meta_batch_size, len(train_tasks)))

        maml.meta_optimizer.zero_grad()
        outer_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for i, task in enumerate(batch_tasks):
            support_loader = task.support_loader(features, batch_size=64)
            query_loader   = task.query_loader(features, batch_size=256)
            if query_loader is None:
                continue

            # Inner loop：在 support set 上适配
            adapted = maml.model  # 先复制原始模型
            adapted_copy = type(maml.model)(vocab_sizes, cfg.embedding_dim, cfg.mlp_dims, cfg.dropout).to(device)
            adapted_copy.load_state_dict(maml.model.state_dict())
            
            # 执行 inner loop
            from src.meta_learner import inner_loop
            adapted = inner_loop(
                adapted_copy, support_loader,
                cfg.inner_lr, cfg.inner_steps,
                device, first_order=maml.first_order
            )

            # Outer loss：在 query set 上计算
            batch_count = 0
            for X, y in query_loader:
                X, y = X.to(device), y.to(device)  # 修正这里：y.to(device) 而不是 y.to(y.device)
                try:
                    loss_val = adapted.loss(X, y)
                    outer_loss = outer_loss + loss_val
                    batch_count += 1
                    if batch_count >= 2:  # 每个 task 最多用 2 个 batch
                        break
                except Exception as e:
                    logger.warning(f"Error computing loss for task {task.campaignset_id}: {e}")
                    continue

        if batch_count > 0:
            outer_loss = outer_loss / (len(batch_tasks) * batch_count)
        
        # 梯度检查
        if outer_loss.requires_grad:
            try:
                outer_loss.backward()
                # 检查梯度
                debug_gradients(maml.model, f"epoch_{epoch}_before_clip")
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(maml.model.parameters(), 1.0)
                
                # 更新参数
                maml.meta_optimizer.step()
                
                logger.info(f"[MAML-debug] epoch={epoch+1}/{cfg.meta_epochs} "
                            f"outer_loss={outer_loss.item():.4f}")
                
            except RuntimeError as e:
                logger.error(f"Error during backward pass: {e}")
                continue
        else:
            logger.warning(f"[MAML-debug] epoch={epoch+1}/{cfg.meta_epochs} "
                           f"outer_loss={outer_loss.item():.4f} (no grad)")
            continue

    # 训练完成后评估
    logger.info("Evaluating trained MAML...")
    r = eval_adapted(maml, test_tasks, features, device)
    logger.info(f"MAML final result: AUC={r['mean_auc']:.4f}, PCOC={r['mean_pcoc']:.3f}")
    
    logger.info("Debug training completed.")


if __name__ == "__main__":
    main()
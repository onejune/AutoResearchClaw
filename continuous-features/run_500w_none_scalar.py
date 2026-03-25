"""
临时脚本：500万样本 Criteo 标准版，只跑 NoneEncoder + ScalarEncoder
优化：预加载数据到内存 tensor，避免 DataLoader 重复 IO 开销
"""
import os
import sys
import time
import random
import logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data import get_dataloaders
from feature_encoders import build_encoder, BucketEncoder
from models import DeepFM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collect_to_tensors(loader, desc=""):
    """将 DataLoader 的数据全部收集到内存 Tensor"""
    all_xc, all_xcat, all_y = [], [], []
    t0 = time.time()
    for xc, xcat, y in loader:
        all_xc.append(xc)
        all_xcat.append(xcat)
        all_y.append(y)
    X_cont = torch.cat(all_xc, dim=0)
    X_cat = torch.cat(all_xcat, dim=0)
    Y = torch.cat(all_y, dim=0)
    logger.info(f"{desc}: {len(Y):,} 样本，收集耗时 {time.time()-t0:.1f}s")
    return X_cont, X_cat, Y


def eval_auc(model, X_cont, X_cat, Y, batch_size=131072):
    """在内存 tensor 上计算 AUC"""
    model.eval()
    preds_list = []
    n = len(Y)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xc = X_cont[i:i+batch_size]
            xcat = X_cat[i:i+batch_size]
            preds_list.append(model(xc, xcat).numpy())
    preds = np.concatenate(preds_list)
    return roc_auc_score(Y.numpy(), preds)


def run_experiment(encoder_name, config, X_cont, X_cat, Y, X_cont_test, X_cat_test, Y_test):
    set_seed(config.seed)
    config.encoder = encoder_name

    print(f"\n{'─'*50}")
    print(f"▶ 运行编码器：{encoder_name}")
    print(f"{'─'*50}")

    encoder = build_encoder(config)

    if isinstance(encoder, BucketEncoder):
        print("  拟合分桶边界...")
        n_fit = min(100_000, len(X_cont))
        encoder.fit(X_cont[:n_fit].numpy())

    cat_vocab_sizes = [config.cat_vocab_size] * config.n_categorical
    model = DeepFM(
        feature_encoder=encoder,
        cat_vocab_sizes=cat_vocab_sizes,
        embedding_dim=config.embedding_dim,
        n_continuous=config.n_continuous,
        n_categorical=config.n_categorical,
    )

    n_params = count_parameters(model)
    print(f"  参数量：{n_params:,} ({n_params/1000:.1f} K)")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.BCELoss()

    bs = 65536
    n = len(Y)
    n_batches = (n + bs - 1) // bs

    best_auc = 0.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        idx = torch.randperm(n)

        for i in range(n_batches):
            batch_idx = idx[i*bs:(i+1)*bs]
            xc = X_cont[batch_idx]
            xcat = X_cat[batch_idx]
            y = Y[batch_idx]

            optimizer.zero_grad()
            out = model(xc, xcat)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        test_auc = eval_auc(model, X_cont_test, X_cat_test, Y_test)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch}/{config.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"AUC: {test_auc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        print(
            f"  Epoch {epoch}/{config.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"AUC: {test_auc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if test_auc > best_auc:
            best_auc = test_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    total_time = time.time() - start_time
    print(f"  最佳 AUC：{best_auc:.4f}，总时间：{total_time:.1f}s")
    return best_auc, n_params, total_time


def main():
    config = Config()
    config.sample_size = 5_000_000
    config.dataset = "criteo_std"
    config.epochs = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    print(f"数据集：{config.dataset}，sample_size={config.sample_size:,}")
    print(f"epochs={config.epochs}")

    print(f"\n加载数据（500万样本）...")
    train_loader, test_loader, dataset = get_dataloaders(config)
    print(f"连续特征：{config.n_continuous}，类别特征：{config.n_categorical}")
    print(f"训练集：{len(train_loader.dataset):,}，测试集：{len(test_loader.dataset):,}")

    # 预加载到内存 tensor（避免每 epoch 重复 DataLoader IO）
    print("\n预加载数据到内存 tensor...")
    X_cont, X_cat, Y = collect_to_tensors(train_loader, "训练集")
    X_cont_test, X_cat_test, Y_test = collect_to_tensors(test_loader, "测试集")

    encoders_to_run = ["none", "scalar"]
    results = {}

    for enc_name in encoders_to_run:
        try:
            auc, n_params, elapsed = run_experiment(
                enc_name, config,
                X_cont, X_cat, Y,
                X_cont_test, X_cat_test, Y_test,
            )
            results[enc_name] = {"auc": auc, "params": n_params, "time": elapsed}
        except Exception as e:
            logger.error(f"编码器 {enc_name} 运行失败：{e}", exc_info=True)
            results[enc_name] = {"auc": float("nan"), "params": 0, "time": 0, "error": str(e)}

    # 打印最终结果
    print("\n" + "="*60)
    print("最终结果（500万样本 Criteo 标准版）")
    print("="*60)
    display_names = {"none": "NoneEncoder", "scalar": "ScalarEncoder"}
    for enc_name in encoders_to_run:
        r = results[enc_name]
        dn = display_names[enc_name]
        if "error" in r:
            print(f"{dn:<14} ERROR: {r['error']}")
        else:
            params_k = r["params"] / 1000
            print(f"{dn:<14} AUC={r['auc']:.4f}  params={params_k:.1f}K  time={r['time']:.1f}s")
    print("="*60)


if __name__ == "__main__":
    main()

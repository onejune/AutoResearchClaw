"""
临时脚本：Criteo 标准版，500 万样本，只跑 NumericEmbedding + PeriodicEncoder
优化：向量化编码器 + 单表 embedding 加速（CPU 下快 ~10x）
"""
import os
import sys
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data import get_dataloaders
from feature_encoders import BaseEncoder
from evaluate import compute_auc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 向量化 NumericEmbeddingEncoder
# ─────────────────────────────────────────────
class FastNumericEmbeddingEncoder(BaseEncoder):
    """
    向量化版 NumericEmbedding：批量矩阵乘法代替 Python 循环，~6x 加速。
    等价于原 NumericEmbeddingEncoder。
    """

    def __init__(self, n_continuous: int = 13, embedding_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.n_continuous = n_continuous
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.W1 = nn.Parameter(torch.randn(n_continuous, hidden_dim, 1) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(n_continuous, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(n_continuous, embedding_dim, hidden_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(n_continuous, embedding_dim))

    @property
    def output_dim(self) -> int:
        return self.n_continuous * self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)  # (batch, n_cont, 1)
        h = torch.einsum('bfi,fji->bfj', x, self.W1) + self.b1.unsqueeze(0)
        h = F.relu(h)
        out = torch.einsum('bfi,fji->bfj', h, self.W2) + self.b2.unsqueeze(0)
        return out.reshape(out.shape[0], -1)


# ─────────────────────────────────────────────
# 向量化 PeriodicEncoder（原版已向量化）
# ─────────────────────────────────────────────
class FastPeriodicEncoder(BaseEncoder):
    def __init__(self, n_continuous: int = 13, n_frequencies: int = 16, sigma: float = 1.0):
        super().__init__()
        self.n_continuous = n_continuous
        self.n_frequencies = n_frequencies
        self.w = nn.Parameter(torch.randn(n_continuous, n_frequencies) * sigma)
        self.b = nn.Parameter(torch.randn(n_continuous, n_frequencies) * sigma)

    @property
    def output_dim(self) -> int:
        return self.n_continuous * 2 * self.n_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.unsqueeze(-1) * self.w.unsqueeze(0) + self.b.unsqueeze(0)
        out = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        return out.reshape(out.shape[0], -1)


# ─────────────────────────────────────────────
# 优化版 DeepFM（单表 embedding，避免 26 次循环）
# ─────────────────────────────────────────────
class FastDeepFM(nn.Module):
    """
    DeepFM with:
    - 单表 embedding（offset trick）代替 26 个独立 Embedding，~40x 加速
    - 向量化连续特征编码器
    """

    def __init__(
        self,
        feature_encoder: BaseEncoder,
        n_categorical: int,
        cat_vocab_size: int,
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.n_cat = n_categorical
        self.cat_vocab_size = cat_vocab_size
        self.embedding_dim = embedding_dim

        # 单表 embedding（FM embedding）
        self.cat_embedding = nn.Embedding(n_categorical * cat_vocab_size, embedding_dim, padding_idx=0)
        # 单表 linear embedding（FM 一阶）
        self.cat_linear_emb = nn.Embedding(n_categorical * cat_vocab_size, 1, padding_idx=0)

        cont_out_dim = feature_encoder.output_dim
        self.cont_out_dim = cont_out_dim
        self.has_cont = cont_out_dim > 0

        if self.has_cont:
            self.cont_linear = nn.Linear(cont_out_dim, 1, bias=False)
            self.cont_fm_proj = nn.Linear(cont_out_dim, embedding_dim, bias=False)

        # Deep
        deep_input_dim = cont_out_dim + n_categorical * embedding_dim
        self.deep = nn.Sequential(
            nn.Linear(deep_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Precompute offsets (register as buffer so it moves with model.to(device))
        offsets = torch.arange(n_categorical, dtype=torch.long) * cat_vocab_size
        self.register_buffer('offsets', offsets)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def forward(self, x_cont: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # Offset cat indices: (batch, n_cat)
        x_cat_off = x_cat + self.offsets.unsqueeze(0)

        # ── 连续特征编码 ──
        cont_emb = self.feature_encoder(x_cont) if self.has_cont else None

        # ── 类别特征 embedding ──
        cat_emb_stack = self.cat_embedding(x_cat_off)  # (batch, n_cat, emb_dim)

        # ── FM 一阶 ──
        fm1_cat = self.cat_linear_emb(x_cat_off).sum(dim=1)  # (batch, 1)
        if self.has_cont:
            fm1 = self.cont_linear(cont_emb) + fm1_cat
        else:
            fm1 = fm1_cat

        # ── FM 二阶 ──
        if self.has_cont:
            cont_fm = self.cont_fm_proj(cont_emb).unsqueeze(1)  # (batch, 1, emb_dim)
            all_embs = torch.cat([cont_fm, cat_emb_stack], dim=1)
        else:
            all_embs = cat_emb_stack
        sum_sq = all_embs.sum(dim=1) ** 2
        sq_sum = (all_embs ** 2).sum(dim=1)
        fm2 = 0.5 * (sum_sq - sq_sum).sum(dim=-1, keepdim=True)

        # ── Deep ──
        cat_flat = cat_emb_stack.reshape(cat_emb_stack.shape[0], -1)
        if self.has_cont:
            deep_input = torch.cat([cont_emb, cat_flat], dim=-1)
        else:
            deep_input = cat_flat
        deep_out = self.deep(deep_input)

        logit = fm1 + fm2 + deep_out
        return torch.sigmoid(logit.squeeze(-1))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Trainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.criterion = nn.BCELoss()

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for x_cont, x_cat, labels in train_loader:
            x_cont = x_cont.to(self.device)
            x_cat = x_cat.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(x_cont, x_cat)
            loss = self.criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / n_batches

    def fit(self, train_loader, test_loader):
        best_auc = 0.0
        patience_counter = 0
        start_time = time.time()
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader)
            test_auc = compute_auc(self.model, test_loader, self.device)
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"Loss: {train_loss:.4f} | AUC: {test_auc:.4f} | Time: {epoch_time:.1f}s"
            )
            print(
                f"  Epoch {epoch}/{self.config.epochs} | "
                f"Loss: {train_loss:.4f} | AUC: {test_auc:.4f} | Time: {epoch_time:.1f}s"
            )
            if test_auc > best_auc:
                best_auc = test_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        total_time = time.time() - start_time
        return best_auc, total_time


def run_experiment(enc_display_name, encoder, config, train_loader, test_loader, device):
    set_seed(config.seed)
    print(f"\n{'─'*50}")
    print(f"▶ 运行编码器：{enc_display_name}")
    print(f"{'─'*50}")

    model = FastDeepFM(
        feature_encoder=encoder,
        n_categorical=config.n_categorical,
        cat_vocab_size=config.cat_vocab_size,
        embedding_dim=config.embedding_dim,
    )
    n_params = count_parameters(model)
    print(f"  参数量：{n_params:,} ({n_params/1000:.1f} K)")

    trainer = Trainer(model, config, device)
    best_auc, total_time = trainer.fit(train_loader, test_loader)
    print(f"  最佳 AUC：{best_auc:.4f}，总时间：{total_time:.1f}s")
    return best_auc, n_params, total_time


def main():
    config = Config()
    config.sample_size = 5_000_000
    config.dataset = "criteo_std"
    config.epochs = 3
    config.batch_size = 65536  # larger batch for CPU efficiency

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    print(f"数据集：{config.dataset}，样本量：{config.sample_size:,}")

    print("\n加载数据（500 万行）...")
    train_loader, test_loader, dataset = get_dataloaders(config)
    print(f"连续特征：{config.n_continuous}，类别特征：{config.n_categorical}")
    print(f"训练集：{len(train_loader.dataset):,}，测试集：{len(test_loader.dataset):,}")

    experiments = [
        ("NumericEmbedding", FastNumericEmbeddingEncoder(
            n_continuous=config.n_continuous,
            embedding_dim=config.embedding_dim,
        )),
        ("PeriodicEncoder", FastPeriodicEncoder(
            n_continuous=config.n_continuous,
            n_frequencies=config.n_frequencies,
        )),
    ]

    results = {}
    for display_name, encoder in experiments:
        try:
            auc, n_params, elapsed = run_experiment(
                display_name, encoder, config, train_loader, test_loader, device
            )
            results[display_name] = {"auc": auc, "params": n_params, "time": elapsed}
        except Exception as e:
            logger.error(f"编码器 {display_name} 失败：{e}", exc_info=True)
            results[display_name] = {"auc": float("nan"), "params": 0, "time": 0, "error": str(e)}

    # 打印最终结果
    print("\n" + "=" * 60)
    print("最终结果（Criteo 标准版，500 万样本）")
    print("=" * 60)
    for display_name, _ in experiments:
        r = results[display_name]
        if "error" in r:
            print(f"{display_name:<20} ERROR: {r['error']}")
        else:
            params_k = r["params"] / 1000
            print(f"{display_name:<20} AUC={r['auc']:.4f}  params={params_k:.1f}K  time={r['time']:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()

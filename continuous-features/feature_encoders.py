"""
连续特征处理方法集合

统一接口：
    输入：(batch, n_continuous) float
    输出：(batch, output_dim)   float
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# N_CONTINUOUS 已移除模块级常量，改为从 config.n_continuous 动态传入
# 各 Encoder 类的默认值仅作兼容保留，实际使用时请显式传入 n_continuous


class BaseEncoder(nn.Module):
    """所有编码器的基类"""

    @property
    def output_dim(self) -> int:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ─────────────────────────────────────────────
# 方法 1：ScalarEncoder（baseline）
# ─────────────────────────────────────────────
class NoneEncoder(BaseEncoder):
    """不使用连续特征（ablation: 纯类别特征）"""

    def __init__(self, n_continuous: int = 8):
        super().__init__()
        self.n_continuous = n_continuous

    @property
    def output_dim(self) -> int:
        return 0  # 不输出任何维度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 返回空 tensor，shape (batch, 0)
        return x[:, :0]


class ScalarEncoder(BaseEncoder):
    """直接使用连续值，不做 embedding"""

    def __init__(self, n_continuous: int = 8):
        super().__init__()
        self.n_continuous = n_continuous

    @property
    def output_dim(self) -> int:
        return self.n_continuous

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  # (batch, n_continuous)


# ─────────────────────────────────────────────
# 方法 2：BucketEncoder（等频分桶 + embedding）
# ─────────────────────────────────────────────
class BucketEncoder(BaseEncoder):
    """
    等频分桶后做 embedding lookup。
    分桶边界在 fit() 时由训练数据确定。
    """

    def __init__(
        self,
        n_continuous: int = 8,
        n_buckets: int = 10,
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.n_continuous = n_continuous
        self.n_buckets = n_buckets
        self.embedding_dim = embedding_dim

        # 每个特征独立 embedding 表（n_buckets + 1 个桶，含边界外）
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_buckets + 1, embedding_dim)
            for _ in range(n_continuous)
        ])

        # 分桶边界（训练后 fit）
        self.register_buffer(
            "boundaries",
            torch.zeros(n_continuous, n_buckets - 1)
        )
        self._fitted = False

    @property
    def output_dim(self) -> int:
        return self.n_continuous * self.embedding_dim

    def fit(self, x: np.ndarray):
        """用训练数据计算等频分桶边界"""
        boundaries = []
        for i in range(self.n_continuous):
            col = x[:, i]
            quantiles = np.percentile(col, np.linspace(0, 100, self.n_buckets + 1)[1:-1])
            boundaries.append(quantiles)
        self.boundaries = torch.tensor(
            np.array(boundaries, dtype=np.float32),
            device=self.boundaries.device
        )
        self._fitted = True

    def _bucketize(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_continuous)
        返回: (batch, n_continuous) int64 桶索引
        """
        batch = x.shape[0]
        bucket_ids = torch.zeros(batch, self.n_continuous, dtype=torch.long, device=x.device)
        for i in range(self.n_continuous):
            # boundaries[i]: (n_buckets-1,)
            b = self.boundaries[i]  # (n_buckets-1,)
            # torch.bucketize: 返回 0 ~ n_buckets-1
            bucket_ids[:, i] = torch.bucketize(x[:, i].contiguous(), b)
        return bucket_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bucket_ids = self._bucketize(x)  # (batch, n_continuous)
        embs = []
        for i in range(self.n_continuous):
            embs.append(self.embeddings[i](bucket_ids[:, i]))  # (batch, emb_dim)
        return torch.cat(embs, dim=-1)  # (batch, n_continuous * emb_dim)


# ─────────────────────────────────────────────
# 方法 3：AutoDisEncoder
# 参考：AutoDis (KDD 2021) https://arxiv.org/abs/2012.08986
# ─────────────────────────────────────────────
class AutoDisEncoder(BaseEncoder):
    """
    AutoDis：每个特征 H 个元 embedding，用小网络学软权重（softmax 加权）。
    最终 embedding = Σ softmax(w_i) * meta_emb_i
    """

    def __init__(
        self,
        n_continuous: int = 8,
        n_meta_embeddings: int = 16,
        embedding_dim: int = 16,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_continuous = n_continuous
        self.n_meta_embeddings = n_meta_embeddings
        self.embedding_dim = embedding_dim
        self.temperature = temperature

        # 元 embedding：每个特征独立
        # shape: (n_continuous, H, emb_dim)
        self.meta_embeddings = nn.Parameter(
            torch.randn(n_continuous, n_meta_embeddings, embedding_dim) * 0.01
        )

        # 权重网络：scalar → H 个权重
        # 每个特征独立一个 MLP（1 → 64 → H）
        self.weight_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, n_meta_embeddings),
            )
            for _ in range(n_continuous)
        ])

    @property
    def output_dim(self) -> int:
        return self.n_continuous * self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_continuous)
        batch = x.shape[0]
        outputs = []
        for i in range(self.n_continuous):
            xi = x[:, i:i+1]  # (batch, 1)
            logits = self.weight_nets[i](xi)  # (batch, H)
            weights = F.softmax(logits / self.temperature, dim=-1)  # (batch, H)
            # meta_embeddings[i]: (H, emb_dim)
            emb = torch.matmul(weights, self.meta_embeddings[i])  # (batch, emb_dim)
            outputs.append(emb)
        return torch.cat(outputs, dim=-1)  # (batch, n_continuous * emb_dim)


# ─────────────────────────────────────────────
# 方法 4：NumericEmbeddingEncoder
# 每个特征独立小 MLP：scalar → embedding
# ─────────────────────────────────────────────
class NumericEmbeddingEncoder(BaseEncoder):
    """
    用 MLP 把每个标量映射成 embedding 向量。
    每个特征独立：Linear(1, hidden) → ReLU → Linear(hidden, emb_dim)
    """

    def __init__(
        self,
        n_continuous: int = 8,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_continuous = n_continuous
        self.embedding_dim = embedding_dim

        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
            for _ in range(n_continuous)
        ])

    @property
    def output_dim(self) -> int:
        return self.n_continuous * self.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(self.n_continuous):
            xi = x[:, i:i+1]  # (batch, 1)
            outputs.append(self.mlps[i](xi))  # (batch, emb_dim)
        return torch.cat(outputs, dim=-1)  # (batch, n_continuous * emb_dim)


# ─────────────────────────────────────────────
# 方法 5：FTTransformerEncoder
# 参考：FT-Transformer (NeurIPS 2021) https://arxiv.org/abs/2106.11959
# ─────────────────────────────────────────────
class FTTransformerEncoder(BaseEncoder):
    """
    FT-Transformer：
    1. 每个连续特征 token = scalar * W_i + b_i（线性投影到 d_model）
    2. 所有 token 过 Transformer Encoder（n_layers 层，n_heads 头）
    3. 输出所有 token 拼接
    """

    def __init__(
        self,
        n_continuous: int = 8,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.n_continuous = n_continuous
        self.d_model = d_model

        # 每个特征独立的线性投影：scalar → d_model
        # W: (n_continuous, d_model), b: (n_continuous, d_model)
        self.W = nn.Parameter(torch.randn(n_continuous, d_model) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_continuous, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    @property
    def output_dim(self) -> int:
        return self.n_continuous * self.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_continuous)
        # 线性投影：每个特征 scalar * W_i + b_i
        # x.unsqueeze(-1): (batch, n_continuous, 1)
        # W: (n_continuous, d_model) → broadcast
        tokens = x.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)
        # tokens: (batch, n_continuous, d_model)

        # Transformer
        out = self.transformer(tokens)  # (batch, n_continuous, d_model)

        # 展平
        return out.reshape(out.shape[0], -1)  # (batch, n_continuous * d_model)


# ─────────────────────────────────────────────
# 方法 6：PeriodicEncoder
# 参考：Periodic Activations (NeurIPS 2021)
# ─────────────────────────────────────────────
class PeriodicEncoder(BaseEncoder):
    """
    周期性激活函数编码：
    x → [sin(w1*x+b1), cos(w1*x+b1), ..., sin(wK*x+bK), cos(wK*x+bK)]
    w, b 可学习，K=16
    output_dim = n_continuous * 2K
    """

    def __init__(
        self,
        n_continuous: int = 8,
        n_frequencies: int = 16,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.n_continuous = n_continuous
        self.n_frequencies = n_frequencies

        # 每个特征独立的 w, b
        # w: (n_continuous, K), b: (n_continuous, K)
        self.w = nn.Parameter(torch.randn(n_continuous, n_frequencies) * sigma)
        self.b = nn.Parameter(torch.randn(n_continuous, n_frequencies) * sigma)

    @property
    def output_dim(self) -> int:
        return self.n_continuous * 2 * self.n_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_continuous)
        # x.unsqueeze(-1): (batch, n_continuous, 1)
        # w: (n_continuous, K) → (1, n_continuous, K)
        z = x.unsqueeze(-1) * self.w.unsqueeze(0) + self.b.unsqueeze(0)
        # z: (batch, n_continuous, K)
        sin_z = torch.sin(z)  # (batch, n_continuous, K)
        cos_z = torch.cos(z)  # (batch, n_continuous, K)
        # 拼接 sin 和 cos
        out = torch.cat([sin_z, cos_z], dim=-1)  # (batch, n_continuous, 2K)
        return out.reshape(out.shape[0], -1)  # (batch, n_continuous * 2K)


# ─────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────
def build_encoder(config) -> BaseEncoder:
    """根据 config.encoder 字符串构建对应编码器，n_continuous 从 config 动态读取"""
    name = config.encoder.lower()
    emb_dim = config.embedding_dim
    n_cont = config.n_continuous  # 由数据集决定，加载后自动填充

    if n_cont <= 0:
        raise ValueError(
            "config.n_continuous 未设置（值为 0），请先调用 get_dataloaders(config) "
            "以自动填充 n_continuous"
        )

    if name == "none":
        return NoneEncoder(n_continuous=n_cont)
    elif name == "scalar":
        return ScalarEncoder(n_continuous=n_cont)
    elif name == "bucket":
        return BucketEncoder(
            n_continuous=n_cont,
            n_buckets=config.n_buckets,
            embedding_dim=emb_dim,
        )
    elif name == "autodis":
        return AutoDisEncoder(
            n_continuous=n_cont,
            n_meta_embeddings=config.n_meta_embeddings,
            embedding_dim=emb_dim,
        )
    elif name == "numeric":
        return NumericEmbeddingEncoder(
            n_continuous=n_cont,
            embedding_dim=emb_dim,
        )
    elif name == "fttransformer":
        return FTTransformerEncoder(
            n_continuous=n_cont,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
        )
    elif name == "periodic":
        return PeriodicEncoder(
            n_continuous=n_cont,
            n_frequencies=config.n_frequencies,
        )
    else:
        raise ValueError(f"未知编码器：{name}，可选：none/scalar/bucket/autodis/numeric/fttransformer/periodic")

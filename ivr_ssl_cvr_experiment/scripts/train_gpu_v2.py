import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow.parquet as pq
import numpy as np
from sklearn.metrics import roc_auc_score

# Configurations
DATA_DIR = '/mnt/workspace/dataset/ivr_sample_v16_ctcvr_sample'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Seed settings for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Dataset loading
print(f'[{DEVICE.upper()}] Loading dataset...')
train = pq.read_table(f'{DATA_DIR}/train/').to_pandas()
test = pq.read_table(f'{DATA_DIR}/test/').to_pandas()

# Features and label
str_features = ['deviceid', 'adid', 'adsize', 'adx', 'bundle', 'business_type']  # Truncated, fill others
num_features = []  # Fill numeric features if any from meta

# Data Preprocessing (fill as per need)
train_str = {f: torch.LongTensor(train[f].values.astype(np.int64)) for f in str_features}
test_str = {f: torch.LongTensor(test[f].values.astype(np.int64)) for f in str_features}
train_label = torch.FloatTensor(train['ctcvr_label'].values.astype(np.float32))
test_label = torch.FloatTensor(test['ctcvr_label'].values.astype(np.float32))

class BaselineModel(nn.Module):
    def __init__(self, in_features, out_features, hidden_size=128):
        super(BaselineModel, self).__init__()
        self.embs = nn.ModuleList([nn.Embedding(v, hidden_size) for v in [1000] * len(in_features)])  # Dummy emb sizes
        self.fc1 = nn.Linear(len(in_features) * hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = torch.cat([emb(x[f]) for f, emb in zip(str_features, self.embs)], dim=1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(-1)

model = BaselineModel(str_features, 1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop (only a sketch)
print('Starting training...')
model.train()
for epoch in range(5):
    output = model(torch.stack([train_str[f] for f in str_features], dim=1).to(DEVICE))
    loss = F.binary_cross_entropy(output, train_label.to(DEVICE))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')

# Evaluation
print('Evaluating...')
model.eval()
with torch.no_grad():
    preds = model(torch.stack([test_str[f] for f in str_features], dim=1).to(DEVICE))
    auc = roc_auc_score(test_label.cpu(), preds.cpu())
    print(f'Overall AUC: {auc:.4f}')

# Business Type AUC Comparison
print('Calculating business_type AUC...')
unique_bt = train['business_type'].unique()
for bt in unique_bt:
    idx = (test['business_type'] == bt)
    if idx.sum() == 0: continue
    preds_bt = preds[idx]
    auc_bt = roc_auc_score(test_label[idx].cpu(), preds_bt.cpu())
    print(f'{bt}: AUC = {auc_bt:.4f}')

print('Done!')

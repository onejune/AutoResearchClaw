import sys, os, logging, time, random
import numpy as np
import torch

sys.path.insert(0, '/mnt/workspace/open_research/autoresearch/exp_multitask')
os.chdir('/mnt/workspace/open_research/autoresearch/exp_multitask')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

from config import Config
from data import build_dataloaders
from models import build_model
from trainer import Trainer

MODELS = ['direct_ctcvr', 'shared_bottom', 'esmm', 'mmoe', 'escm2']
DATA_PATH = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/taobao/UserBehavior.csv'

all_results = {}

for temporal in [False, True]:
    tag = 'temporal' if temporal else 'random'
    all_results[tag] = {}
    print(f'\n{"="*60}')
    print(f'切分方式: {tag}')
    print(f'{"="*60}')

    config = Config()
    config.dataset = 'taobao'
    config.data_path = DATA_PATH
    config.sample_size = 2_000_000
    config.taobao_temporal = temporal
    config.epochs = 1
    config.early_stopping_patience = 1
    config.batch_size = 4096
    config.seed = 42

    train_loader, val_loader, test_loader, feature_info = build_dataloaders(config)
    print(f'dense_dim={feature_info["dense_dim"]}', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_name in MODELS:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        config.model_name = model_name
        model = build_model(feature_info, config)
        trainer = Trainer(model, config)
        t0 = time.time()
        metrics = trainer.fit(train_loader, val_loader, test_loader)
        elapsed = time.time() - t0
        all_results[tag][model_name] = {'metrics': metrics, 'elapsed': elapsed}
        print(f'RESULT [{tag}] {model_name}: CTR={metrics["ctr_auc"]:.4f} CVR={metrics["cvr_auc"]:.4f} CTCVR={metrics["ctcvr_auc"]:.4f} {elapsed:.1f}s', flush=True)

print('\n' + '='*70)
print('对比结果：随机切分 vs 时序切分')
print('='*70)
print(f'{"模型":<16} {"随机-CTR":>10} {"随机-CTCVR":>12} {"时序-CTR":>10} {"时序-CTCVR":>12}')
print('-'*70)
for m in MODELS:
    r = all_results['random'][m]['metrics']
    t = all_results['temporal'][m]['metrics']
    print(f'{m:<16} {r["ctr_auc"]:>10.4f} {r["ctcvr_auc"]:>12.4f} {t["ctr_auc"]:>10.4f} {t["ctcvr_auc"]:>12.4f}')
print('='*70)
print('ALL DONE')

from .baseline import Baseline
from .shared_bottom import SharedBottom
from .mmoe import MMoE
from .ple import PLE
from .star import STAR

MODEL_REGISTRY = {
    "baseline": Baseline,
    "shared_bottom": SharedBottom,
    "mmoe": MMoE,
    "ple": PLE,
    "star": STAR,
}

def build_model(model_type: str, **kwargs):
    """根据 model_type 实例化模型。"""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](**kwargs)

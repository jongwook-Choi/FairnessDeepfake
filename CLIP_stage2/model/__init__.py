"""
CLIP Stage2 Model Module
"""

from model.additive_adapter import AdditiveAdapter, ResidualAdditiveAdapter
from model.stage2_model import Stage2LinearProbingModel
from model.cross_attention_fusion import CrossAttentionFusionWithDynamicGate, SimpleFusion
from model.stage2_independent_adapter_model import Stage2IndependentAdapterModel

__all__ = [
    'AdditiveAdapter',
    'ResidualAdditiveAdapter',
    'Stage2LinearProbingModel',
    'CrossAttentionFusionWithDynamicGate',
    'SimpleFusion',
    'Stage2IndependentAdapterModel',
]

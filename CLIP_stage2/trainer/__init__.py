"""
CLIP Stage2 Trainer Module
"""

from trainer.stage2_trainer import Stage2Trainer
from trainer.stage2_layerwise_trainer import Stage2LayerwiseTrainer, LayerWiseWarmupScheduler

__all__ = [
    'Stage2Trainer',
    'Stage2LayerwiseTrainer',
    'LayerWiseWarmupScheduler',
]

"""
CLIP Stage2 Loss Module
"""

from losses.detection_loss import DetectionLoss, FocalLoss
from losses.stage2_fairness_loss import (
    Stage2FairnessLoss,
    Stage2GlobalFairnessLoss,
    CombinedStage2FairnessLoss
)
from losses.dynamic_loss_weighting import (
    DynamicFairnessLossWeighting,
    AdaptiveFairnessLossWeighting,
    GradNormLossWeighting,
    FixedLossWeighting
)

__all__ = [
    # Detection losses
    'DetectionLoss',
    'FocalLoss',
    # Fairness losses
    'Stage2FairnessLoss',
    'Stage2GlobalFairnessLoss',
    'CombinedStage2FairnessLoss',
    # Loss weighting
    'DynamicFairnessLossWeighting',
    'AdaptiveFairnessLossWeighting',
    'GradNormLossWeighting',
    'FixedLossWeighting',
]

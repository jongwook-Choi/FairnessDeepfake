"""
CLIP Stage1 Text 모델 모듈
"""

from .additive_adapter import AdditiveAdapter
from .fairness_adapter_txt import FairnessAdapterWithText, FairnessAdapterWithTextBinaryClassifier

__all__ = [
    'AdditiveAdapter',
    'FairnessAdapterWithText',
    'FairnessAdapterWithTextBinaryClassifier',
]

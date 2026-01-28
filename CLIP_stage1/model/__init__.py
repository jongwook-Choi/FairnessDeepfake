"""
CLIP Stage1 모델 모듈
"""

from .additive_adapter import AdditiveAdapter
from .fairness_adapter import FairnessAdapter

__all__ = ['AdditiveAdapter', 'FairnessAdapter']

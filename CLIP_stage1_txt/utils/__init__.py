"""
CLIP Stage1 Text Utils 모듈
"""

from .fairness_metrics import (
    compute_subgroup_feature_stats,
    compute_subgroup_cosine_similarity,
    compute_feature_variance_ratio,
    compute_subgroup_sinkhorn_distance
)

from .prompt_templates import (
    SUBGROUP_PROMPTS,
    SUBGROUP_NAMES,
    SUBGROUP_MAPPING,
    GENDER_MAPPING,
    RACE_MAPPING,
    get_subgroup_prompts,
    get_prompts_for_subgroup,
    compute_subgroup_id,
    get_subgroup_name,
)

__all__ = [
    # Fairness metrics
    'compute_subgroup_feature_stats',
    'compute_subgroup_cosine_similarity',
    'compute_feature_variance_ratio',
    'compute_subgroup_sinkhorn_distance',
    # Prompt templates
    'SUBGROUP_PROMPTS',
    'SUBGROUP_NAMES',
    'SUBGROUP_MAPPING',
    'GENDER_MAPPING',
    'RACE_MAPPING',
    'get_subgroup_prompts',
    'get_prompts_for_subgroup',
    'compute_subgroup_id',
    'get_subgroup_name',
]

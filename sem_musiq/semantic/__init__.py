"""
Semantic module for Sem-MUSIQ.

Provides semantic vector generation using SAM (Segment Anything).
"""

from .sam_feature_extractor import SAMFeatureExtractor, extract_sam_features
from .consistency_scorer import (
    SemanticConsistencyScorer,
    LearnableConsistencyScorer,
    compute_semantic_consistency,
)

__all__ = [
    'SAMFeatureExtractor',
    'extract_sam_features',
    'SemanticConsistencyScorer',
    'LearnableConsistencyScorer',
    'compute_semantic_consistency',
]

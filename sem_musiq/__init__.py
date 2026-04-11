"""
Sem-MUSIQ: Semantic-aware Multi-scale Image Quality Transformer

双分支架构：语义 Transformer + MUSIQ 主干
"""

from .archs.musiq_arch import MUSIQ, CatSemanticEmbs
from .semantic.sam_feature_extractor import SAMFeatureExtractor, extract_sam_features
from .transformer.semantic_transformer import SemanticTransformerEncoder, build_semantic_embeddings
from .semantic.consistency_scorer import SemanticConsistencyScorer
from .fusion import SemMUSIQFusion, QualityScoreFusion

__version__ = '2.0.0'
__all__ = [
    # 核心模型
    'MUSIQ',
    'SemMUSIQFusion',
    'QualityScoreFusion',
    # 语义分支
    'SAMFeatureExtractor',
    'extract_sam_features',
    'SemanticTransformerEncoder',
    'build_semantic_embeddings',
    'SemanticConsistencyScorer',
]

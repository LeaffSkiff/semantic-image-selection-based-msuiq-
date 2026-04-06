"""
Sem-MUSIQ: Semantic-aware Multi-scale Image Quality Transformer

精简版 MUSIQ，支持语义嵌入功能。
"""

from .archs.musiq_arch import MUSIQ, AddSemanticEmbs
from .semantic.vector_generator import SemanticVectorGenerator, generate_semantic_vectors
from .fusion import SemMUSIQFusion, QualityScoreFusion

__version__ = '1.0.0'
__all__ = [
    'MUSIQ',
    'AddSemanticEmbs',
    'SemanticVectorGenerator',
    'generate_semantic_vectors',
    'SemMUSIQFusion',
    'QualityScoreFusion',
]

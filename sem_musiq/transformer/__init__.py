"""
Transformer module for Sem-MUSIQ.

Provides semantic Transformer encoder for multi-scale patch processing.
"""

from .semantic_transformer import (
    SemanticTransformerEncoder,
    ScaleSemanticTransformer,
    build_semantic_embeddings,
)

__all__ = [
    'SemanticTransformerEncoder',
    'ScaleSemanticTransformer',
    'build_semantic_embeddings',
]
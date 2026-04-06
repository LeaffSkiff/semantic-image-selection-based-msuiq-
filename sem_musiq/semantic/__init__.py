"""
Semantic module for Sem-MUSIQ.

Provides semantic vector generation using SAM (Segment Anything).
"""

from .vector_generator import SemanticVectorGenerator, generate_semantic_vectors

__all__ = ['SemanticVectorGenerator', 'generate_semantic_vectors']

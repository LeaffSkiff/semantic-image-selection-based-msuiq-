"""
Sem-MUSIQ: Semantic-aware Multi-scale Image Quality Transformer

精简版 MUSIQ，支持语义嵌入功能。
"""

from .archs.musiq_arch import MUSIQ, AddSemanticEmbs

__version__ = '1.0.0'
__all__ = ['MUSIQ', 'AddSemanticEmbs']

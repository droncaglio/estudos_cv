# cv_lib/filtragem/__init__.py
"""
📐 Módulo de Filtragem Espacial

Este módulo contém implementações de filtros espaciais (convolução)
para processamento de imagens no domínio espacial.

📚 **REFERÊNCIAS ACADÊMICAS:**
- **Gonzalez & Woods, Digital Image Processing 4e, Seção 3.4-3.6**
  Fundamentos de filtragem espacial, convolução, filtros lineares e não-lineares

Submódulos:
    - espacial: Filtros espaciais lineares (convolução, suavização, aguçamento)
"""

from .espacial import (
    convolucao_2d,
    correlacao_2d,
    filtro_media,
    filtro_gaussiano,
    filtro_sobel,
    filtro_prewitt,
    filtro_laplaciano,
    unsharp_masking
)

__all__ = [
    'convolucao_2d',
    'correlacao_2d',
    'filtro_media',
    'filtro_gaussiano',
    'filtro_sobel',
    'filtro_prewitt',
    'filtro_laplaciano',
    'unsharp_masking'
]

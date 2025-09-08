# cv_lib/espacos_cor/__init__.py
"""
🌈 Módulo de Espaços de Cor

Este módulo implementa conversões entre diferentes espaços de cor utilizados
em visão computacional, com foco educacional e explicações detalhadas.

Espaços Implementados:
- RGB: Padrão para displays e câmeras
- Escala de Cinza: 7 métodos diferentes (BT.601, BT.709, etc.)
- YCbCr: Separação luminância/crominância para compressão
- HSV: Intuitivo para manipulação artística 
- Lab: Perceptualmente uniforme para comparações precisas

Referências:
- Gonzalez & Woods, Digital Image Processing, Cap. 6
- Szeliski, Computer Vision: Algorithms and Applications, Cap. 2
"""

# Imports das conversões disponíveis
from .rgb import rgb_para_cinza
from .ycbcr import rgb_para_ycbcr, ycbcr_para_rgb
from .hsv import rgb_para_hsv, hsv_para_rgb
from .lab import rgb_para_lab, lab_para_rgb

# Lista de todas as funções exportadas
__all__ = [
    'rgb_para_cinza',
    'rgb_para_ycbcr', 'ycbcr_para_rgb',
    'rgb_para_hsv', 'hsv_para_rgb', 
    'rgb_para_lab', 'lab_para_rgb'
]
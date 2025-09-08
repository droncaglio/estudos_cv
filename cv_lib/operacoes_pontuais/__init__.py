# cv_lib/operacoes_pontuais/__init__.py
"""
⚙️ Módulo de Operações Pontuais

Este módulo implementa transformações aplicadas pixel por pixel, onde o novo
valor de cada pixel depende APENAS do seu valor original, sem considerar vizinhos.

Características das Operações Pontuais:
- Independência espacial: Cada pixel processado isoladamente
- Função de mapeamento: novo_valor = f(valor_original)  
- Preservação da estrutura: Forma e contornos mantidos
- Alteração de intensidade: Modifica brilho, contraste, gama

Operações Implementadas:
- Transformações lineares: Brilho e contraste
- Transformações power-law: Correção gama
- Normalização: Padronização de ranges
- Operações aritméticas: Entre múltiplas imagens

Referências:
- Gonzalez & Woods, Digital Image Processing, Cap. 3
- Jain, A.K. (1989). Fundamentals of Digital Image Processing
"""

# Imports das operações disponíveis
from .brilho_contraste import ajustar_brilho_contraste
from .gama import correcao_gama
from .normalizacao import normalizar_imagem
from .aritmetica import operacao_entre_imagens

# Lista de todas as funções exportadas
__all__ = [
    'ajustar_brilho_contraste',
    'correcao_gama', 
    'normalizar_imagem',
    'operacao_entre_imagens'
]
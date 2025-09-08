# cv_lib/__init__.py
"""
🔍 Biblioteca CV Educacional - Implementação Manual de Algoritmos

Esta biblioteca implementa algoritmos clássicos de visão computacional
from-scratch, com foco educacional e explicações detalhadas.

Estrutura Modular:
- espacos_cor/: Conversões entre espaços de cor (RGB, YCbCr, HSV, Lab)
- operacoes_pontuais/: Transformações pixel-a-pixel (brilho, contraste, gama)
- utils/: Utilitários compartilhados (validação, etc.)

Filosofia Educacional:
- Evita "caixas-pretas" de bibliotecas como OpenCV
- Prioriza clareza e compreensão sobre performance
- Documentação rica com aplicações práticas
- Implementações pixel-por-pixel para máximo aprendizado

Uso:
    from cv_lib import processamento  # Compatibilidade com notebooks antigos
    from cv_lib.espacos_cor import rgb_para_hsv  # Imports específicos
    from cv_lib.operacoes_pontuais import correcao_gama  # Imports diretos

Referências:
- Gonzalez & Woods, Digital Image Processing
- Szeliski, Computer Vision: Algorithms and Applications
"""

# ============================================================================
# IMPORTS PARA COMPATIBILIDADE COM NOTEBOOKS EXISTENTES
# ============================================================================

# Re-exporta todas as funções dos módulos para manter API simples
# Permite: from cv_lib import processamento; processamento.rgb_para_cinza()

from .espacos_cor.rgb import (
    rgb_para_cinza,
    obter_pesos_luminancia,
    estatisticas_canais_rgb
)

from .espacos_cor.ycbcr import (
    rgb_para_ycbcr, 
    ycbcr_para_rgb,
    extrair_luminancia,
    extrair_crominancia,
    obter_coeficientes_ycbcr
)

from .espacos_cor.hsv import (
    rgb_para_hsv, 
    hsv_para_rgb,
    extrair_hue,
    filtrar_por_cor,
    obter_ranges_hsv
)

from .espacos_cor.lab import (
    rgb_para_lab, 
    lab_para_rgb,
    calcular_delta_e,
    extrair_luminosidade,
    obter_white_point_d65,
    obter_matriz_srgb_para_xyz,
    obter_matriz_xyz_para_srgb
)

from .operacoes_pontuais.brilho_contraste import (
    ajustar_brilho_contraste,
    mapear_intervalo,
    aplicar_curva_linear_por_partes,
    calcular_estatisticas_transformacao
)

from .operacoes_pontuais.gama import (
    correcao_gama,
    gama_adaptativo_por_regioes,
    curva_gama_invertida,
    detectar_gama_imagem,
    gerar_curva_gama
)

from .operacoes_pontuais.normalizacao import (
    normalizar_imagem,
    normalizar_z_score,
    normalizar_por_percentis,
    normalizar_adaptativa_local,
    equalizar_histograma_local,
    calcular_metricas_normalizacao
)

from .operacoes_pontuais.aritmetica import (
    operacao_entre_imagens,
    blending_linear,
    subtracao_de_fundo,
    correcao_flat_field,
    fusao_multiplas_exposicoes,
    aplicar_mascara,
    detectar_movimento_temporal,
    calcular_metricas_comparacao
)

# ============================================================================
# NAMESPACE "processamento" PARA COMPATIBILIDADE
# ============================================================================

# Cria um módulo simples ao invés de classe para evitar conflitos
import types
processamento = types.SimpleNamespace()

# Espaços de cor
processamento.rgb_para_cinza = rgb_para_cinza
processamento.rgb_para_ycbcr = rgb_para_ycbcr
processamento.ycbcr_para_rgb = ycbcr_para_rgb
processamento.rgb_para_hsv = rgb_para_hsv
processamento.hsv_para_rgb = hsv_para_rgb
processamento.rgb_para_lab = rgb_para_lab
processamento.lab_para_rgb = lab_para_rgb

# Operações pontuais
processamento.ajustar_brilho_contraste = ajustar_brilho_contraste
processamento.correcao_gama = correcao_gama
processamento.normalizar_imagem = normalizar_imagem
processamento.operacao_entre_imagens = operacao_entre_imagens

# Funções auxiliares
processamento.extrair_luminancia = extrair_luminancia
processamento.extrair_hue = extrair_hue
processamento.calcular_delta_e = calcular_delta_e
processamento.blending_linear = blending_linear
processamento.subtracao_de_fundo = subtracao_de_fundo

# ============================================================================
# LISTA DE EXPORTS PÚBLICOS
# ============================================================================

__all__ = [
    # Namespace principal para compatibilidade
    'processamento',
    
    # Funções de espaços de cor
    'rgb_para_cinza', 'rgb_para_ycbcr', 'ycbcr_para_rgb', 
    'rgb_para_hsv', 'hsv_para_rgb', 'rgb_para_lab', 'lab_para_rgb',
    
    # Operações pontuais  
    'ajustar_brilho_contraste', 'correcao_gama', 'normalizar_imagem',
    'operacao_entre_imagens',
    
    # Funções avançadas
    'blending_linear', 'subtracao_de_fundo', 'filtrar_por_cor',
    'calcular_delta_e', 'detectar_movimento_temporal'
]

# ============================================================================
# INFORMAÇÕES DA BIBLIOTECA
# ============================================================================

__version__ = "1.0.0"
__author__ = "Estudos em Visão Computacional"
__description__ = "Biblioteca educacional de CV com implementações from-scratch"

def info():
    """Exibe informações sobre a biblioteca."""
    print(f"🔍 Biblioteca CV Educacional v{__version__}")
    print(f"📚 {__description__}")
    print(f"")
    print(f"📂 Módulos disponíveis:")
    print(f"  • espacos_cor: Conversões RGB, YCbCr, HSV, Lab")
    print(f"  • operacoes_pontuais: Brilho, contraste, gama, normalização")
    print(f"  • utils: Validação e utilitários")
    print(f"")
    print(f"💡 Uso:")
    print(f"  from cv_lib import processamento  # Compatibilidade")
    print(f"  from cv_lib.espacos_cor import rgb_para_hsv  # Direto")
    print(f"")
    print(f"📖 Documentação: Veja docstrings das funções individuais")
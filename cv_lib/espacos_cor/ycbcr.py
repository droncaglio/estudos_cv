# cv_lib/espacos_cor/ycbcr.py
"""
📺 Conversões RGB ↔ YCbCr (Luminância + Crominância)

YCbCr separa informação de brilho (Y) da informação de cor (Cb, Cr),
fundamental para compressão de vídeo e TV digital.

Componentes:
- Y: Luminância (brilho) - similar ao que vemos em P&B
- Cb: Crominância azul-amarelo (Blue-Yellow chroma)  
- Cr: Crominância vermelho-verde (Red-Green chroma)

Aplicações:
- Compressão JPEG/MPEG (reduz resolução Cb/Cr)
- TV digital e broadcasting
- Detecção de pele (componentes Cb/Cr robustas)
- Processamento independente de luminância e cor

Referências:
- ITU-R BT.601: Studio Encoding Parameters
- JPEG Standard (ISO/IEC 10918)
- Poynton, C. (2003). Digital Video and HDTV
"""

import numpy as np
from ..utils.validacao import validar_imagem_rgb, garantir_uint8


def rgb_para_ycbcr(imagem_rgb):
    """
    Converte imagem RGB para espaço de cor YCbCr (luminância + crominância).
    
    Args:
        imagem_rgb: Array RGB shape (altura, largura, 3)
        
    Returns:
        np.ndarray: Imagem YCbCr shape (altura, largura, 3) dtype uint8
    
    Fórmulas ITU-R BT.601:
        Y  = 0.299*R + 0.587*G + 0.114*B      (luminância)
        Cb = -0.169*R - 0.331*G + 0.500*B + 128  (azul-amarelo)
        Cr = 0.500*R - 0.419*G - 0.081*B + 128   (vermelho-verde)
    
    Por que +128? Para centralizar os valores de crominância no meio 
    do range [0-255], já que Cb e Cr podem ser negativos.
    
    Aplicações:
        - Compressão de vídeo/imagem (JPEG subsample Cb/Cr)
        - Separação de luminância e cor para processamento independente
        - TV digital e broadcasting
        - Detecção de pele humana (usando componentes Cb/Cr)
    """
    altura, largura, canais = validar_imagem_rgb(imagem_rgb, "imagem_rgb")
    imagem_ycbcr = np.zeros((altura, largura, 3), dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            r, g, b = imagem_rgb[y, x, 0], imagem_rgb[y, x, 1], imagem_rgb[y, x, 2]
            
            # Calcula componentes YCbCr usando fórmulas ITU-R BT.601
            Y = 0.299 * r + 0.587 * g + 0.114 * b
            Cb = -0.169 * r - 0.331 * g + 0.500 * b + 128
            Cr = 0.500 * r - 0.419 * g - 0.081 * b + 128
            
            imagem_ycbcr[y, x] = [Y, Cb, Cr]

    # Garante valores no intervalo [0, 255]
    imagem_ycbcr = np.clip(imagem_ycbcr, 0, 255)
    return garantir_uint8(imagem_ycbcr)


def ycbcr_para_rgb(imagem_ycbcr):
    """
    Converte imagem YCbCr de volta para RGB.
    
    Args:
        imagem_ycbcr: Array YCbCr shape (altura, largura, 3)
        
    Returns:
        np.ndarray: Imagem RGB shape (altura, largura, 3) dtype uint8
    
    Fórmulas inversas ITU-R BT.601:
        R = Y + 1.402*(Cr-128)
        G = Y - 0.344*(Cb-128) - 0.714*(Cr-128)  
        B = Y + 1.772*(Cb-128)
        
    Nota: A conversão não é perfeitamente reversível devido a:
        - Arredondamentos em ponto flutuante
        - Clipping para range [0-255]
        - Quantização uint8
    """
    altura, largura, _ = imagem_ycbcr.shape
    imagem_rgb = np.zeros((altura, largura, 3), dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            Y, Cb, Cr = imagem_ycbcr[y, x, 0], imagem_ycbcr[y, x, 1], imagem_ycbcr[y, x, 2]
            
            # Converte de volta para RGB usando fórmulas inversas
            R = Y + 1.402 * (Cr - 128)
            G = Y - 0.344 * (Cb - 128) - 0.714 * (Cr - 128)
            B = Y + 1.772 * (Cb - 128)
            
            imagem_rgb[y, x] = [R, G, B]

    # Garante valores no intervalo [0, 255]
    imagem_rgb = np.clip(imagem_rgb, 0, 255)
    return garantir_uint8(imagem_rgb)


def extrair_luminancia(imagem_rgb):
    """
    Extrai apenas o canal de luminância (Y) de uma imagem RGB.
    
    Args:
        imagem_rgb: Array RGB shape (altura, largura, 3)
        
    Returns:
        np.ndarray: Canal Y shape (altura, largura) dtype uint8
        
    Equivalente a rgb_para_cinza(imagem, tipo='luminancia') mas mais
    semânticamente claro quando o objetivo é obter luminância YCbCr.
    """
    altura, largura, _ = validar_imagem_rgb(imagem_rgb)
    luminancia = np.zeros((altura, largura), dtype=np.float64)
    
    for y in range(altura):
        for x in range(largura):
            r, g, b = imagem_rgb[y, x, 0], imagem_rgb[y, x, 1], imagem_rgb[y, x, 2]
            luminancia[y, x] = 0.299 * r + 0.587 * g + 0.114 * b
    
    return garantir_uint8(luminancia)


def extrair_crominancia(imagem_rgb, componente='cb'):
    """
    Extrai apenas um canal de crominância de uma imagem RGB.
    
    Args:
        imagem_rgb: Array RGB shape (altura, largura, 3)
        componente: 'cb' (azul-amarelo) ou 'cr' (vermelho-verde)
        
    Returns:
        np.ndarray: Canal Cb ou Cr shape (altura, largura) dtype uint8
        
    Útil para:
        - Análise isolada de informação de cor
        - Detecção de pele (Cb/Cr são robustas a mudanças de iluminação)
        - Estudos de compressão (JPEG reduz resolução destes canais)
    """
    if componente not in ['cb', 'cr']:
        raise ValueError("Componente deve ser 'cb' ou 'cr'")
        
    altura, largura, _ = validar_imagem_rgb(imagem_rgb)
    crominancia = np.zeros((altura, largura), dtype=np.float64)
    
    for y in range(altura):
        for x in range(largura):
            r, g, b = imagem_rgb[y, x, 0], imagem_rgb[y, x, 1], imagem_rgb[y, x, 2]
            
            if componente == 'cb':
                # Cb: Crominância azul-amarelo
                valor = -0.169 * r - 0.331 * g + 0.500 * b + 128
            else:  # componente == 'cr'
                # Cr: Crominância vermelho-verde  
                valor = 0.500 * r - 0.419 * g - 0.081 * b + 128
                
            crominancia[y, x] = valor
    
    crominancia = np.clip(crominancia, 0, 255)
    return garantir_uint8(crominancia)


def obter_coeficientes_ycbcr():
    """
    Retorna os coeficientes da transformação RGB→YCbCr (BT.601).
    
    Returns:
        dict: Coeficientes organizados por componente
        
    Útil para:
        - Implementações matriciais/vetorizadas
        - Análise teórica das transformações
        - Comparação com outros padrões
    """
    return {
        'Y': {'R': 0.299, 'G': 0.587, 'B': 0.114},
        'Cb': {'R': -0.169, 'G': -0.331, 'B': 0.500, 'offset': 128},
        'Cr': {'R': 0.500, 'G': -0.419, 'B': -0.081, 'offset': 128}
    }
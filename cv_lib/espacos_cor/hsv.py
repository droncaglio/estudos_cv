# cv_lib/espacos_cor/hsv.py
"""
🎨 Conversões RGB ↔ HSV (Hue, Saturation, Value)

HSV representa cores de forma intuitiva para manipulação artística:
- H: Hue (Matiz) - a cor em si (0-360° na roda de cores)
- S: Saturation (Saturação) - pureza/intensidade da cor (0-100%)
- V: Value (Valor/Brilho) - quão clara/escura é a cor (0-100%)

Vantagens sobre RGB:
- Seleção por cor mais robusta que RGB
- Menos sensível a mudanças de iluminação
- Ajustes artísticos intuitivos (saturação sem afetar cor)
- Melhor para segmentação baseada em cor

Aplicações:
- Detecção de objetos coloridos (bolas, carros, etc.)
- Segmentação baseada em cor
- Aplicações de realidade aumentada  
- Interfaces de seleção de cor

Referências:
- Smith, A.R. (1978). Color Gamut Transform Pairs
- Foley & van Dam, Computer Graphics: Principles and Practice
"""

import numpy as np
from ..utils.validacao import validar_imagem_rgb, garantir_uint8


def rgb_para_hsv(imagem_rgb):
    """
    Converte imagem RGB para espaço de cor HSV (Hue, Saturation, Value).
    
    Args:
        imagem_rgb: Array RGB shape (altura, largura, 3)
        
    Returns:
        np.ndarray: Imagem HSV shape (altura, largura, 3) dtype uint8
        
    Ranges de Saída:
        - H: [0-179] (para caber em uint8, dividido por 2 de 0-360°)
        - S: [0-255] (0-100% mapeado para 0-255)
        - V: [0-255] (0-100% mapeado para 0-255)
    
    ALGORITMO DETALHADO DO HUE (MATIZ):
    A roda de cores HSV é dividida em 6 setores de 60 graus cada:
    - Setor 0: 0°-60°   (Vermelho → Amarelo)
    - Setor 1: 60°-120° (Amarelo → Verde) 
    - Setor 2: 120°-180°(Verde → Ciano)
    - Setor 3: 180°-240°(Ciano → Azul)
    - Setor 4: 240°-300°(Azul → Magenta)
    - Setor 5: 300°-360°(Magenta → Vermelho)
    """
    altura, largura, canais = validar_imagem_rgb(imagem_rgb, "imagem_rgb")
    imagem_hsv = np.zeros((altura, largura, 3), dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            # Normaliza RGB para [0,1] para facilitar cálculos
            r = imagem_rgb[y, x, 0] / 255.0
            g = imagem_rgb[y, x, 1] / 255.0
            b = imagem_rgb[y, x, 2] / 255.0
            
            # PASSO 1: Encontra valores máximo, mínimo e diferença (delta)
            max_val = max(r, g, b)  # Componente com maior intensidade
            min_val = min(r, g, b)  # Componente com menor intensidade
            delta = max_val - min_val  # Diferença = "intensidade da cor"
            
            # PASSO 2: Calcula VALUE (brilho) - simplesmente o valor máximo
            v = max_val  # Value = quão "brilhante" é a cor (0-1)
            
            # PASSO 3: Calcula SATURATION (saturação) - quão "pura" é a cor
            if max_val == 0:
                # Se max_val = 0, a cor é preta (sem saturação)
                s = 0
            else:
                # Saturação = diferença / brilho
                # Se delta=0, cor é acinzentada (sem saturação)  
                # Se delta=max_val, cor é "pura" (saturação máxima)
                s = delta / max_val
            
            # PASSO 4: Calcula HUE (matiz) - a cor em si na roda de cores
            if delta == 0:
                # Se não há diferença entre RGB, a cor é acinzentada (sem matiz definido)
                h = 0
                
            elif max_val == r:
                # VERMELHO é dominante - estamos nos setores 0° ou 5° da roda
                # Fórmula: posição dentro do setor baseada na diferença G-B
                posicao_no_setor = (g - b) / delta
                
                # Multiplica por 60° (tamanho do setor) e usa % 6 para lidar com valores negativos
                # % 6 garante que valores negativos "deem a volta" na roda (ex: -1 vira 5)
                h = 60 * (posicao_no_setor % 6)
                
            elif max_val == g:
                # VERDE é dominante - estamos no setor 1° (60°-120°)
                # +2 = pula 2 setores (120° de offset)
                posicao_no_setor = (b - r) / delta  # Posição baseada em B-R
                h = 60 * (posicao_no_setor + 2)     # +2 setores = +120°
                
            elif max_val == b:
                # AZUL é dominante - estamos no setor 2° (240°-300°)  
                # +4 = pula 4 setores (240° de offset)
                posicao_no_setor = (r - g) / delta  # Posição baseada em R-G  
                h = 60 * (posicao_no_setor + 4)     # +4 setores = +240°
            
            # PASSO 5: Converte para ranges convencionais de armazenamento
            # OpenCV usa H em [0-179] para caber em uint8 (180 valores)
            h = h / 2        # Divide por 2: [0-360°] → [0-180]
            s = s * 255      # S: [0-1] → [0-255]  
            v = v * 255      # V: [0-1] → [0-255]
            
            imagem_hsv[y, x] = [h, s, v]

    return garantir_uint8(imagem_hsv)


def hsv_para_rgb(imagem_hsv):
    """
    Converte imagem HSV de volta para RGB.
    
    Args:
        imagem_hsv: Array HSV shape (altura, largura, 3)
        
    Returns:
        np.ndarray: Imagem RGB shape (altura, largura, 3) dtype uint8
        
    Algoritmo:
        1. Desnormaliza H de [0-179] para [0-360°]
        2. Calcula chroma (intensidade da cor)
        3. Determina componentes primários por setor
        4. Adiciona offset para brilho final
    """
    altura, largura, _ = imagem_hsv.shape
    imagem_rgb = np.zeros((altura, largura, 3), dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            # Desnormaliza os valores HSV
            h = imagem_hsv[y, x, 0] * 2        # [0-179] → [0-360°]
            s = imagem_hsv[y, x, 1] / 255.0    # [0-255] → [0-1]
            v = imagem_hsv[y, x, 2] / 255.0    # [0-255] → [0-1]
            
            # Algoritmo de conversão HSV → RGB
            c = v * s  # Chroma (intensidade da cor saturada)
            x_val = c * (1 - abs((h / 60) % 2 - 1))  # Segundo componente
            m = v - c  # Offset para ajustar brilho
            
            # Determina RGB baseado no setor do Hue
            if 0 <= h < 60:         # Setor 0: Vermelho → Amarelo
                r, g, b = c, x_val, 0
            elif 60 <= h < 120:     # Setor 1: Amarelo → Verde
                r, g, b = x_val, c, 0
            elif 120 <= h < 180:    # Setor 2: Verde → Ciano
                r, g, b = 0, c, x_val
            elif 180 <= h < 240:    # Setor 3: Ciano → Azul
                r, g, b = 0, x_val, c
            elif 240 <= h < 300:    # Setor 4: Azul → Magenta
                r, g, b = x_val, 0, c
            elif 300 <= h < 360:    # Setor 5: Magenta → Vermelho
                r, g, b = c, 0, x_val
            else:
                r, g, b = 0, 0, 0   # Caso edge (não deveria acontecer)
            
            # Adiciona m (offset de brilho) e converte para [0-255]
            imagem_rgb[y, x] = [(r + m) * 255, (g + m) * 255, (b + m) * 255]

    return garantir_uint8(imagem_rgb)


def extrair_hue(imagem_rgb):
    """
    Extrai apenas o canal Hue (matiz) de uma imagem RGB.
    
    Args:
        imagem_rgb: Array RGB shape (altura, largura, 3)
        
    Returns:
        np.ndarray: Canal H shape (altura, largura) dtype uint8 [0-179]
        
    Útil para:
        - Segmentação baseada apenas em cor (ignorando brilho/saturação)
        - Análise de distribuição de cores
        - Detecção de objetos coloridos específicos
    """
    imagem_hsv = rgb_para_hsv(imagem_rgb)
    return imagem_hsv[:, :, 0]  # Retorna apenas canal H


def filtrar_por_cor(imagem_rgb, cor_alvo_hsv, tolerancia_h=10, tolerancia_s=50, tolerancia_v=50):
    """
    Cria máscara binária para pixels de uma cor específica no espaço HSV.
    
    Args:
        imagem_rgb: Array RGB shape (altura, largura, 3)
        cor_alvo_hsv: tuple (h, s, v) da cor desejada em ranges HSV
        tolerancia_h: Tolerância para Hue [0-179]
        tolerancia_s: Tolerância para Saturação [0-255]
        tolerancia_v: Tolerância para Value [0-255]
        
    Returns:
        np.ndarray: Máscara binária shape (altura, largura) dtype uint8 (0 ou 255)
        
    Aplicações:
        - Detecção de objetos coloridos (bolas, carros)
        - Segmentação de regiões por cor
        - Tracking de objetos coloridos
        - Realidade aumentada baseada em cor
    """
    imagem_hsv = rgb_para_hsv(imagem_rgb)
    altura, largura, _ = imagem_hsv.shape
    
    h_alvo, s_alvo, v_alvo = cor_alvo_hsv
    mascara = np.zeros((altura, largura), dtype=np.uint8)
    
    for y in range(altura):
        for x in range(largura):
            h, s, v = imagem_hsv[y, x, 0], imagem_hsv[y, x, 1], imagem_hsv[y, x, 2]
            
            # Verifica se pixel está dentro das tolerâncias
            # Nota: Hue é circular, precisa considerar wrap-around (0° = 360°)
            diff_h = min(abs(h - h_alvo), 180 - abs(h - h_alvo))  # Distância circular
            diff_s = abs(s - s_alvo)
            diff_v = abs(v - v_alvo)
            
            if (diff_h <= tolerancia_h and 
                diff_s <= tolerancia_s and 
                diff_v <= tolerancia_v):
                mascara[y, x] = 255  # Pixel da cor alvo
            else:
                mascara[y, x] = 0    # Pixel de outra cor
                
    return mascara


def obter_ranges_hsv():
    """
    Retorna ranges típicos para cores comuns no espaço HSV.
    
    Returns:
        dict: Ranges HSV para cores básicas
        
    Útil para:
        - Configuração inicial de filtros de cor
        - Referência rápida para segmentação
        - Debugging de algoritmos baseados em cor
        
    Nota: Valores em ranges OpenCV (H: 0-179, S/V: 0-255)
    """
    return {
        'vermelho_1': {'h': (0, 10), 's': (50, 255), 'v': (50, 255)},    # 0°-20°
        'vermelho_2': {'h': (170, 179), 's': (50, 255), 'v': (50, 255)}, # 340°-360°
        'laranja': {'h': (10, 25), 's': (50, 255), 'v': (50, 255)},      # 20°-50°
        'amarelo': {'h': (25, 35), 's': (50, 255), 'v': (50, 255)},      # 50°-70°
        'verde': {'h': (35, 85), 's': (50, 255), 'v': (50, 255)},        # 70°-170°
        'azul': {'h': (85, 125), 's': (50, 255), 'v': (50, 255)},        # 170°-250°
        'roxo': {'h': (125, 155), 's': (50, 255), 'v': (50, 255)},       # 250°-310°
        'rosa': {'h': (155, 170), 's': (50, 255), 'v': (50, 255)}        # 310°-340°
    }
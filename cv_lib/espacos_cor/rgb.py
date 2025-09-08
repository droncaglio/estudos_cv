# cv_lib/espacos_cor/rgb.py
"""
🔴🟢🔵 Conversões RGB para Escala de Cinza

Este módulo implementa 7 métodos diferentes de conversão RGB→Cinza,
cada um otimizado para aplicações específicas.

Métodos Implementados:
1. Luminância/BT.601: Padrão clássico baseado na visão humana
2. BT.709: Padrão HDTV moderno
3. Média: Simples média aritmética  
4. Desaturação: Preserva contraste em cores saturadas
5-7. Canais individuais: R, G, B isolados

Referências:
- ITU-R BT.601: Standard for Studio Encoding Parameters
- ITU-R BT.709: Parameter Values for HDTV Standards
- Poynton, C. (2003). Digital Video and HDTV Algorithms
"""

import numpy as np
from ..utils.validacao import validar_imagem_rgb, garantir_uint8


def rgb_para_cinza(imagem_rgb, tipo='luminancia'):
    """
    Converte uma imagem RGB para escala de cinza usando diferentes métodos.
    
    Args:
        imagem_rgb: Array NumPy com shape (altura, largura, 3) ou (altura, largura)
        tipo: Método de conversão:
            'luminancia' ou 'bt601': Fórmula padrão (0.299*R + 0.587*G + 0.114*B)
            'bt709': Fórmula HDTV (0.2126*R + 0.7152*G + 0.0722*B) 
            'media': Média aritmética simples
            'desaturacao': (max + min) / 2
            'canal_r': Apenas canal vermelho
            'canal_g': Apenas canal verde  
            'canal_b': Apenas canal azul
            
    Returns:
        np.ndarray: Imagem em escala de cinza (altura, largura) dtype uint8
        
    Aplicações por Método:
        - luminancia/bt601: Processamento geral, compatibilidade
        - bt709: TV digital, monitores modernos
        - media: Algoritmos simples, prototipagem
        - desaturacao: Arte digital, preservar contraste
        - canal_r: Medicina, detecção de pele
        - canal_g: Agricultura, visão noturna  
        - canal_b: Hidrologia, análise atmosférica
    """
    # Verifica se a imagem já está em escala de cinza
    if len(imagem_rgb.shape) == 2:
        return garantir_uint8(imagem_rgb)
    
    altura, largura, canais = validar_imagem_rgb(imagem_rgb, "imagem_rgb")
    
    if canais != 3:
        raise ValueError(f"Imagem deve ter 3 canais (RGB), mas tem {canais}")
    
    # Cria array de resultado
    imagem_cinza = np.zeros((altura, largura), dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            pixel = imagem_rgb[y, x]
            r, g, b = pixel[0], pixel[1], pixel[2]
            
            if tipo in ['luminancia', 'bt601']:
                # ITU-R BT.601 (SDTV): Padrão clássico baseado na sensibilidade do olho humano
                # Aplicações: Processamento geral, compatibilidade com sistemas antigos
                # O olho é mais sensível ao verde (58.7%), depois vermelho (29.9%) e azul (11.4%)
                valor_cinza = 0.299 * r + 0.587 * g + 0.114 * b
                
            elif tipo == 'bt709':
                # ITU-R BT.709 (HDTV): Padrão moderno para TV digital e monitores
                # Aplicações: Processamento para displays modernos, conteúdo HD/4K
                # Coeficientes ajustados para fósforos modernos (mais peso no verde: 71.52%)
                valor_cinza = 0.2126 * r + 0.7152 * g + 0.0722 * b
                
            elif tipo == 'media':
                # Média aritmética simples: Trata todos os canais igualmente
                # Aplicações: Algoritmos simples, prototipagem rápida, quando não há preferência de canal
                # Pode resultar em imagens "chapadas" pois ignora sensibilidade do olho humano
                valor_cinza = (r + g + b) / 3
                
            elif tipo == 'desaturacao':
                # Desaturação: Média entre valor máximo e mínimo dos canais RGB
                # Aplicações: Arte digital, quando se quer preservar contraste de cores saturadas
                # Mantém melhor os detalhes em áreas muito coloridas comparado à luminância
                valor_cinza = (max(r, g, b) + min(r, g, b)) / 2
                
            elif tipo == 'canal_r':
                # Canal vermelho isolado
                # Aplicações: Detecção de sangue/vasos sanguíneos, análise de vegetação (contraste com clorofila),
                # fotografia infravermelha, detecção de pele em imagens médicas
                valor_cinza = r
                
            elif tipo == 'canal_g':
                # Canal verde isolado  
                # Aplicações: Análise de vegetação (clorofila), detecção de plantas em agricultura,
                # melhor canal para detecção de bordas (mais detalhado), visão noturna
                valor_cinza = g
                
            elif tipo == 'canal_b':
                # Canal azul isolado
                # Aplicações: Detecção de água, análise de céu/atmosfera, detecção de veias,
                # contraste em imagens médicas, análise de poluição atmosférica
                valor_cinza = b
                
            else:
                raise ValueError(f"Tipo '{tipo}' não reconhecido. Tipos válidos: "
                               f"'luminancia', 'bt601', 'bt709', 'media', 'desaturacao', "
                               f"'canal_r', 'canal_g', 'canal_b'")
                
            imagem_cinza[y, x] = valor_cinza

    return garantir_uint8(imagem_cinza)


def obter_pesos_luminancia(tipo='bt601'):
    """
    Retorna os pesos de luminância para diferentes padrões.
    
    Args:
        tipo: Padrão de luminância ('bt601', 'bt709')
        
    Returns:
        tuple: (peso_r, peso_g, peso_b)
        
    Útil para:
        - Cálculos analíticos sem processar imagem completa
        - Implementações vetorizadas (NumPy advanced)
        - Comparações entre padrões
    """
    if tipo in ['bt601', 'luminancia']:
        return (0.299, 0.587, 0.114)
    elif tipo == 'bt709':
        return (0.2126, 0.7152, 0.0722)
    else:
        raise ValueError(f"Tipo '{tipo}' não reconhecido. Use 'bt601' ou 'bt709'")


def estatisticas_canais_rgb(imagem_rgb):
    """
    Calcula estatísticas individuais de cada canal RGB.
    
    Args:
        imagem_rgb: Imagem RGB
        
    Returns:
        dict: Estatísticas de cada canal (min, max, média, desvio)
        
    Útil para:
        - Análise de distribuição de cores
        - Detecção de problemas de calibração
        - Escolha do melhor método de conversão
    """
    altura, largura, canais = validar_imagem_rgb(imagem_rgb)
    
    if canais != 3:
        raise ValueError("Função requer imagem RGB com 3 canais")
    
    stats = {}
    nomes_canais = ['R', 'G', 'B']
    
    for i, nome in enumerate(nomes_canais):
        canal = imagem_rgb[:, :, i]
        stats[nome] = {
            'min': int(canal.min()),
            'max': int(canal.max()),
            'media': float(canal.mean()),
            'desvio': float(canal.std())
        }
    
    return stats
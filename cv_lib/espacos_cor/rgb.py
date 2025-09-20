# cv_lib/espacos_cor/rgb.py
"""
🔴🟢🔵 Conversões RGB para Escala de Cinza

📚 **REFERÊNCIAS ACADÊMICAS:**
- **Szeliski, Computer Vision 2e, Eq.2.113, p.122**
  Y'₆₀₁ = 0.299R' + 0.587G' + 0.114B' (BT.601 para SDTV)
- **Szeliski, Computer Vision 2e, Eq.2.114, p.122**
  Y'₇₀₉ = 0.2125R' + 0.7154G' + 0.0721B' (BT.709 para HDTV)
- **ITU-R BT.601**: Standard for Studio Encoding Parameters
- **ITU-R BT.709**: Parameter Values for HDTV Standards
- **Poynton, C. (2003). Digital Video and HDTV Algorithms**

🧮 **FUNDAMENTO MATEMÁTICO:**
Conversão para luminância baseada na sensibilidade espectral do olho humano:
- **Verde**: Maior sensibilidade (58.7% BT.601, 71.54% BT.709)
- **Vermelho**: Sensibilidade média (29.9% BT.601, 21.26% BT.709)
- **Azul**: Menor sensibilidade (11.4% BT.601, 7.21% BT.709)

🎯 **MÉTODOS IMPLEMENTADOS (7 tipos):**
1. **Luminância/BT.601**: Padrão clássico SDTV baseado na visão humana
2. **BT.709**: Padrão HDTV moderno para monitores atuais
3. **Média**: Simples média aritmética (R+G+B)/3
4. **Desaturação**: (max+min)/2 - preserva contraste saturado
5-7. **Canais individuais**: R, G, B isolados para aplicações específicas

⚠️ **NOTA IMPORTANTE:** As fórmulas usam R'G'B' (gamma-comprimido)
conforme especificado nos padrões ITU, não RGB linear.
"""

import numpy as np
from ..utils.validacao import validar_imagem_rgb, garantir_uint8


def rgb_para_cinza(imagem_rgb, tipo='luminancia'):
    """
    Converte uma imagem RGB para escala de cinza usando diferentes métodos.

    📚 **Referências:**
    - Szeliski, Computer Vision 2e, Eq.2.113-2.114, p.122
    - ITU-R BT.601/BT.709 Standards

    🧮 **Fórmulas Acadêmicas:**
    - BT.601: Y' = 0.299R' + 0.587G' + 0.114B' (SDTV)
    - BT.709: Y' = 0.2126R' + 0.7152G' + 0.0722B' (HDTV)

    Args:
        imagem_rgb: Array NumPy com shape (altura, largura, 3) ou (altura, largura)
        tipo: Método de conversão acadêmico:
            'luminancia'/'bt601': Eq.2.113 Szeliski (SDTV padrão)
            'bt709': Eq.2.114 Szeliski (HDTV moderno)
            'media': Média aritmética (R+G+B)/3
            'desaturacao': (max+min)/2 para contraste saturado
            'canal_r': Canal vermelho isolado
            'canal_g': Canal verde isolado
            'canal_b': Canal azul isolado

    Returns:
        np.ndarray: Imagem em escala de cinza (altura, largura) dtype uint8

    📈 **Aplicações Citadas na Literatura:**
        - luminancia/bt601: Compatibilidade com sistemas SDTV legados
        - bt709: TV digital, monitores HD/4K modernos
        - media: Algoritmos simples, prototipagem rápida
        - desaturacao: Arte digital, preservação de contraste
        - canal_r: Medicina (detecção sangue), análise de pele
        - canal_g: Agricultura (vegetação), visão noturna
        - canal_b: Hidrologia (água), análise atmosférica
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
                # ITU-R BT.709 (HDTV): Eq.2.114 Szeliski p.122
                # Aplicações: TV digital, monitores HD/4K modernos
                # Coeficientes para fósforos modernos: Y'₇₀₉ = 0.2125R' + 0.7154G' + 0.0721B'
                valor_cinza = 0.2125 * r + 0.7154 * g + 0.0721 * b
                
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
        return (0.299, 0.587, 0.114)  # Eq.2.113 Szeliski
    elif tipo == 'bt709':
        return (0.2125, 0.7154, 0.0721)  # Eq.2.114 Szeliski
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
# cv_lib/operacoes_pontuais/brilho_contraste.py
"""
💡 Transformações Lineares: Brilho e Contraste

📚 **REFERÊNCIAS ACADÊMICAS:**
- **Szeliski, Computer Vision 2e, Seção 3.1.1, p.137**
  "Pixel transforms" - operações ponto-a-ponto g(i,j) = h(f(i,j))
- **Gonzalez & Woods, Digital Image Processing 4e, Seção 3.2.1**
  "Linear intensity transformations" - transformações básicas de intensidade
- **Crane, R. (1997). A Simplified Approach to Image Processing**
  Definição clássica de "point processes"

🧮 **FÓRMULA MATEMÁTICA FUNDAMENTAL:**
f(x,y) = α × g(x,y) + β

Onde:
- α = contraste (gain) - inclinação da função linear
- β = brilho (bias) - deslocamento vertical
- g(x,y) = intensidade do pixel original
- f(x,y) = intensidade do pixel resultante

📈 **INTERPRETAÇÃO GRÁFICA (Szeliski p.137):**
- **Função linear**: Gráfico entrada vs. saída é uma reta
- **Inclinação**: Determina contraste (α)
- **Intercepto Y**: Determina brilho (β)
- **Domínio**: [0,255] → **Range**: [0,255] com clipping

🎯 **APLICAÇÕES ACADÊMICAS CITADAS:**
- **Brightness scaling**: Correção de imagens subexpostas/superexpostas
- **Image addition**: Combinação linear de múltiplas imagens
- **Display adjustment**: Calibração para diferentes monitores
- **Preprocessing**: Normalização antes de algoritmos de visão

⚙️ **PARÂMETROS TÍPICOS:**
- α ∈ [0.5, 3.0]: Range prático de contraste
- β ∈ [-100, +100]: Range típico de brilho
- Clipping necessário para manter [0,255]
"""

import numpy as np
from ..utils.validacao import validar_imagem_rgb, validar_parametro_numerico, garantir_uint8


def ajustar_brilho_contraste(imagem, brilho=0, contraste=1.0):
    """
    Aplica transformação linear de brilho e contraste.

    📚 **Referência:** Szeliski, Computer Vision 2e, Seção 3.1.1, p.137
                      "Pixel transforms: g(i,j) = h(f(i,j))"

    🧮 **Fórmula:** f(x,y) = α × g(x,y) + β
       onde α=contraste, β=brilho, g=entrada, f=saída

    Args:
        imagem: Array NumPy da imagem (escala de cinza ou RGB)
        brilho: Valor β adicionado a cada pixel (validado: -255 a +255)
            - β > 0: Deslocamento para cima (mais claro)
            - β < 0: Deslocamento para baixo (mais escuro)
            - β = 0: Sem alteração de brilho
        contraste: Fator α multiplicativo (validado: ≥ 0.0)
            - α > 1: Amplifica diferenças (mais contraste)
            - α = 1: Preserva contraste original
            - 0 < α < 1: Reduz diferenças (menos contraste)
            - α = 0: Imagem uniforme (todos pixels = β)

    Returns:
        np.ndarray: Imagem transformada dtype uint8
        
    Fórmula: f(x,y) = α × g(x,y) + β
    
    Aplicações Específicas:
        - Correção de subexposição: brilho=+30~80, contraste=1.0~1.3
        - Correção de superexposição: brilho=-30~-80, contraste=1.0~1.3  
        - Aumentar contraste: brilho=0, contraste=1.2~2.0
        - Diminuir contraste: brilho=0, contraste=0.5~0.8
        - Efeito "lavado": brilho=+50, contraste=0.5
        - Efeito dramático: brilho=-20, contraste=1.8
    """
    # Validações de entrada
    validar_parametro_numerico(brilho, "brilho", -255, 255)
    validar_parametro_numerico(contraste, "contraste", 0.0)
    
    if not isinstance(imagem, np.ndarray):
        raise ValueError("imagem deve ser um array NumPy")
    
    # Converte para float64 para evitar overflow durante os cálculos
    imagem_float = imagem.astype(np.float64)
    
    # Aplica a transformação linear: f = α×g + β  
    resultado = contraste * imagem_float + brilho
    
    # Garante que os valores ficam no intervalo [0, 255] (clipping)
    # Importante: valores fora do range são "cortados" nos extremos
    resultado = np.clip(resultado, 0, 255)
    
    return garantir_uint8(resultado)


def mapear_intervalo(imagem, intervalo_origem, intervalo_destino):
    """
    Mapeia valores de um intervalo para outro usando transformação linear.
    
    Args:
        imagem: Array NumPy da imagem
        intervalo_origem: tuple (min_origem, max_origem)
        intervalo_destino: tuple (min_destino, max_destino)
        
    Returns:
        np.ndarray: Imagem mapeada dtype uint8
        
    Fórmula de Mapeamento Linear:
        y = (x - min_orig) / (max_orig - min_orig) × (max_dest - min_dest) + min_dest
        
    Aplicações:
        - Converter imagem [50-200] para [0-255] (expansão de contraste)
        - Converter imagem [0-255] para [100-150] (redução de contraste)
        - Mapeamento para ranges específicos de algoritmos
        
    Exemplo:
        # Expande contraste de imagem com baixa faixa dinâmica
        img_expandida = mapear_intervalo(img, (50, 200), (0, 255))
    """
    min_orig, max_orig = intervalo_origem
    min_dest, max_dest = intervalo_destino
    
    # Validações
    if min_orig >= max_orig:
        raise ValueError(f"intervalo_origem inválido: {intervalo_origem}")
    if min_dest >= max_dest:
        raise ValueError(f"intervalo_destino inválido: {intervalo_destino}")
    
    # Aplica mapeamento linear
    imagem_float = imagem.astype(np.float64)
    
    # Normaliza para [0, 1] baseado no intervalo de origem
    normalizada = (imagem_float - min_orig) / (max_orig - min_orig)
    
    # Mapeia para intervalo de destino
    resultado = normalizada * (max_dest - min_dest) + min_dest
    
    # Clipping e conversão final
    resultado = np.clip(resultado, 0, 255)
    return garantir_uint8(resultado)


def aplicar_curva_linear_por_partes(imagem, pontos_curva):
    """
    Aplica curva de transformação linear por partes (piecewise linear).
    
    Args:
        imagem: Array NumPy da imagem
        pontos_curva: Lista de tuples [(x1,y1), (x2,y2), ...] definindo a curva
        
    Returns:
        np.ndarray: Imagem transformada dtype uint8
        
    Pontos da curva devem estar ordenados por x crescente e no range [0-255].
    
    Aplicações Avançadas:
        - Correção de contraste não-linear
        - Simulação de curvas de filme fotográfico
        - Ajustes artísticos complexos
        - Correção de gamma personalizada
        
    Exemplo:
        # Curva em S (aumenta contraste nos meio-tons)
        pontos = [(0,0), (64,45), (128,128), (192,210), (255,255)]
        img_s = aplicar_curva_linear_por_partes(img, pontos)
    """
    if len(pontos_curva) < 2:
        raise ValueError("pontos_curva deve ter pelo menos 2 pontos")
    
    # Ordena pontos por x e valida ranges
    pontos_ordenados = sorted(pontos_curva)
    for x, y in pontos_ordenados:
        if not (0 <= x <= 255 and 0 <= y <= 255):
            raise ValueError(f"Ponto ({x},{y}) fora do range [0-255]")
    
    altura, largura = imagem.shape[:2]
    resultado = np.zeros_like(imagem, dtype=np.float64)
    
    # Aplica transformação pixel por pixel
    for i in range(altura):
        for j in range(largura):
            if len(imagem.shape) == 2:  # Escala de cinza
                valor_orig = imagem[i, j]
                resultado[i, j] = _interpolar_linear_por_partes(valor_orig, pontos_ordenados)
            else:  # RGB
                for canal in range(imagem.shape[2]):
                    valor_orig = imagem[i, j, canal]
                    resultado[i, j, canal] = _interpolar_linear_por_partes(valor_orig, pontos_ordenados)
    
    return garantir_uint8(resultado)


def _interpolar_linear_por_partes(x, pontos):
    """
    Interpola valor x usando curva linear por partes definida pelos pontos.
    
    Args:
        x: Valor de entrada [0-255]
        pontos: Lista ordenada de pontos [(x1,y1), (x2,y2), ...]
        
    Returns:
        float: Valor interpolado
    """
    # Casos extremos
    if x <= pontos[0][0]:
        return pontos[0][1]
    if x >= pontos[-1][0]:
        return pontos[-1][1]
    
    # Encontra segmento apropriado
    for i in range(len(pontos) - 1):
        x1, y1 = pontos[i]
        x2, y2 = pontos[i + 1]
        
        if x1 <= x <= x2:
            # Interpolação linear entre os dois pontos
            if x2 == x1:  # Evita divisão por zero
                return y1
            
            t = (x - x1) / (x2 - x1)  # Parâmetro de interpolação [0,1]
            return y1 + t * (y2 - y1)
    
    return x  # Fallback (não deveria acontecer)


def calcular_estatisticas_transformacao(imagem_original, imagem_transformada):
    """
    Calcula estatísticas comparativas entre imagem original e transformada.
    
    Args:
        imagem_original: Array da imagem antes da transformação
        imagem_transformada: Array da imagem após transformação
        
    Returns:
        dict: Estatísticas detalhadas da transformação
        
    Útil para:
        - Análise quantitativa dos efeitos da transformação
        - Debugging de algoritmos
        - Documentação de resultados
        - Escolha de parâmetros ótimos
    """
    stats = {
        'original': {
            'min': int(imagem_original.min()),
            'max': int(imagem_original.max()),
            'media': float(imagem_original.mean()),
            'desvio': float(imagem_original.std()),
            'mediana': float(np.median(imagem_original))
        },
        'transformada': {
            'min': int(imagem_transformada.min()),
            'max': int(imagem_transformada.max()),
            'media': float(imagem_transformada.mean()),
            'desvio': float(imagem_transformada.std()),
            'mediana': float(np.median(imagem_transformada))
        }
    }
    
    # Cálculos derivados
    stats['mudanca_media'] = stats['transformada']['media'] - stats['original']['media']
    stats['mudanca_desvio'] = stats['transformada']['desvio'] - stats['original']['desvio']
    stats['fator_contraste'] = stats['transformada']['desvio'] / stats['original']['desvio'] if stats['original']['desvio'] > 0 else float('inf')
    
    return stats
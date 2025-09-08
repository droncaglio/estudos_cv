# cv_lib/operacoes_pontuais/normalizacao.py
"""
📊 Normalização de Imagens - Padronização de Valores

Implementa técnicas de normalização para ajustar ranges e distribuições
de intensidade, fundamentais para:

- Padronização de imagens com diferentes faixas dinâmicas
- Pré-processamento para algoritmos de Machine Learning  
- Melhoria de contraste quando imagem não usa toda a faixa [0-255]
- Normalização para comparação entre imagens

Tipos de Normalização:
- Linear: Mapeamento proporcional para novo range
- Min-Max: Expansão para usar toda faixa disponível
- Z-Score: Normalização estatística (média=0, desvio=1)
- Percentile: Baseada em percentis para robustez

Referências:
- Gonzalez & Woods, Digital Image Processing, Sec. 3.3
- Jain, A.K. (1989). Fundamentals of Digital Image Processing
- Bishop, C.M. (2006). Pattern Recognition and Machine Learning
"""

import numpy as np
from ..utils.validacao import validar_parametro_numerico, garantir_uint8


def normalizar_imagem(imagem, novo_min=0, novo_max=255):
    """
    Normaliza os valores da imagem para um intervalo especificado.
    
    Args:
        imagem: Array NumPy da imagem
        novo_min: Valor mínimo do intervalo de destino
        novo_max: Valor máximo do intervalo de destino
        
    Returns:
        np.ndarray: Imagem normalizada dtype uint8
        
    Fórmula de Normalização Linear:
        f(x) = (x - min_orig) / (max_orig - min_orig) × (max_novo - min_novo) + min_novo
        
    Aplicações:
        - Padronização de imagens com baixa faixa dinâmica
        - Pré-processamento para algoritmos de ML (ex: [0,1] ou [-1,1])
        - Melhoria de contraste máximo
        - Comparação quantitativa entre imagens
        
    Quando Usar:
        - Imagem usa apenas parte do range [0-255]
        - Preparar dados para redes neurais
        - Equalizar multiple imagens para processamento batch
        - Maximizar uso da faixa dinâmica disponível
    """
    # Validações
    validar_parametro_numerico(novo_min, "novo_min", 0, 254)
    validar_parametro_numerico(novo_max, "novo_max", novo_min + 1, 255)
    
    if not isinstance(imagem, np.ndarray):
        raise ValueError("imagem deve ser um array NumPy")
    
    # Calcula min e max da imagem original
    min_original = float(imagem.min())
    max_original = float(imagem.max())
    
    # Evita divisão por zero se a imagem for uniforme (todos pixels iguais)
    if max_original == min_original:
        return np.full_like(imagem, novo_min, dtype=np.uint8)
    
    # Aplica normalização linear
    imagem_float = imagem.astype(np.float64)
    
    # Passo 1: Normaliza para [0, 1]
    normalizada = (imagem_float - min_original) / (max_original - min_original)
    
    # Passo 2: Mapeia para [novo_min, novo_max]
    resultado = normalizada * (novo_max - novo_min) + novo_min
    
    # Garante range válido (proteção contra erros de arredondamento)
    resultado = np.clip(resultado, 0, 255)
    return garantir_uint8(resultado)


def normalizar_z_score(imagem, nova_media=127.5, novo_desvio=50.0):
    """
    Normaliza imagem usando Z-score (média=0, desvio=1) e depois reescala.
    
    Args:
        imagem: Array NumPy da imagem
        nova_media: Média desejada na imagem resultante
        novo_desvio: Desvio padrão desejado na imagem resultante
        
    Returns:
        np.ndarray: Imagem normalizada dtype uint8
        
    Fórmula Z-Score:
        z = (x - média_orig) / desvio_orig
        resultado = z × novo_desvio + nova_media
        
    Aplicações:
        - Padronização estatística para ML
        - Correção de imagens com distribuições anômalas
        - Pré-processamento para análise estatística
        - Normalização robusta independente de outliers extremos
        
    Vantagens:
        - Preserva forma da distribuição original
        - Menos sensível a outliers que min-max
        - Controle direto sobre média e desvio resultantes
        - Melhor para dados que seguem distribuição normal
    """
    if not isinstance(imagem, np.ndarray):
        raise ValueError("imagem deve ser um array NumPy")
    
    # Calcula estatísticas da imagem original
    media_original = float(imagem.mean())
    desvio_original = float(imagem.std())
    
    # Evita divisão por zero se desvio for muito pequeno
    if desvio_original < 1e-6:
        return np.full_like(imagem, nova_media, dtype=np.uint8)
    
    # Aplica Z-score e reescala
    imagem_float = imagem.astype(np.float64)
    
    # Passo 1: Normaliza para z-score (média=0, desvio=1)
    z_score = (imagem_float - media_original) / desvio_original
    
    # Passo 2: Reescala para nova média e desvio
    resultado = z_score * novo_desvio + nova_media
    
    # Clipping para range válido
    resultado = np.clip(resultado, 0, 255)
    return garantir_uint8(resultado)


def normalizar_por_percentis(imagem, percentil_min=2, percentil_max=98):
    """
    Normaliza usando percentis, mais robusta a outliers extremos.
    
    Args:
        imagem: Array NumPy da imagem  
        percentil_min: Percentil inferior (será mapeado para 0)
        percentil_max: Percentil superior (será mapeado para 255)
        
    Returns:
        np.ndarray: Imagem normalizada dtype uint8
        
    Vantagens sobre Min-Max:
        - Ignora outliers extremos (pixels muito escuros/claros isolados)
        - Mais robusta para imagens com ruído
        - Preserva detalhes na maioria dos pixels
        - Menos sensível a valores extremos artificiais
        
    Aplicações:
        - Imagens médicas com artefatos
        - Fotografias com áreas super/subexpostas pequenas
        - Processamento de imagens com ruído
        - Normalização robusta para análise automatizada
        
    Exemplo:
        percentil_min=5, percentil_max=95 ignora 5% dos pixels mais extremos
    """
    # Validações
    if not (0 <= percentil_min < percentil_max <= 100):
        raise ValueError("Percentis devem satisfazer: 0 ≤ min < max ≤ 100")
    
    if not isinstance(imagem, np.ndarray):
        raise ValueError("imagem deve ser um array NumPy")
    
    # Calcula valores dos percentis
    valor_min = np.percentile(imagem, percentil_min)
    valor_max = np.percentile(imagem, percentil_max)
    
    # Evita divisão por zero
    if valor_max == valor_min:
        return np.full_like(imagem, 127, dtype=np.uint8)
    
    # Aplica normalização baseada nos percentis
    imagem_float = imagem.astype(np.float64)
    
    # Clipping baseado nos percentis
    imagem_clipada = np.clip(imagem_float, valor_min, valor_max)
    
    # Normalização linear para [0, 255]
    resultado = (imagem_clipada - valor_min) / (valor_max - valor_min) * 255
    
    return garantir_uint8(resultado)


def normalizar_adaptativa_local(imagem, tamanho_janela=64, sobreposicao=0.5):
    """
    Aplica normalização adaptativa baseada em janelas locais.
    
    Args:
        imagem: Array da imagem (deve ser escala de cinza)
        tamanho_janela: Tamanho da janela para análise local
        sobreposicao: Fração de sobreposição entre janelas [0-1]
        
    Returns:
        np.ndarray: Imagem normalizada adaptativamente
        
    Algoritmo:
        1. Divide imagem em janelas sobrepostas
        2. Normaliza cada janela independentemente
        3. Faz blend suave entre janelas sobrepostas
        4. Reconstrói imagem final
        
    Aplicações:
        - Correção de iluminação não-uniforme
        - Melhoria de imagens com gradientes de luminosidade
        - Pré-processamento para OCR em documentos antigos
        - Realce de detalhes em imagens médicas
        
    Vantagens:
        - Adapta-se a características locais da imagem
        - Preserva detalhes em todas as regiões
        - Corrige problemas de iluminação global
        - Menos artifacts que normalização global
    """
    if len(imagem.shape) != 2:
        raise ValueError("normalizar_adaptativa_local requer imagem em escala de cinza")
    
    altura, largura = imagem.shape
    resultado = np.zeros_like(imagem, dtype=np.float64)
    contador = np.zeros_like(imagem, dtype=np.float64)
    
    # Calcula step baseado na sobreposição
    step = int(tamanho_janela * (1 - sobreposicao))
    
    # Processa janelas sobrepostas
    for y in range(0, altura - tamanho_janela + 1, step):
        for x in range(0, largura - tamanho_janela + 1, step):
            # Define limites da janela
            y2 = min(y + tamanho_janela, altura)
            x2 = min(x + tamanho_janela, largura)
            
            # Extrai janela
            janela = imagem[y:y2, x:x2]
            
            # Normaliza janela localmente
            janela_norm = normalizar_imagem(janela, 0, 255)
            
            # Cria máscara de peso (fade nas bordas para blend suave)
            peso = _criar_mascara_peso(y2-y, x2-x, fade_size=tamanho_janela//8)
            
            # Acumula resultado ponderado
            resultado[y:y2, x:x2] += janela_norm.astype(np.float64) * peso
            contador[y:y2, x:x2] += peso
    
    # Normaliza por contador para obter média ponderada
    contador[contador == 0] = 1  # Evita divisão por zero
    resultado = resultado / contador
    
    return garantir_uint8(resultado)


def equalizar_histograma_local(imagem, tamanho_janela=64):
    """
    Aplica equalização de histograma local (CLAHE simplificado).
    
    Args:
        imagem: Array da imagem (escala de cinza)
        tamanho_janela: Tamanho da janela para equalização local
        
    Returns:
        np.ndarray: Imagem com histograma equalizado localmente
        
    Diferença da Equalização Global:
        - Global: Uma função de transformação para toda imagem
        - Local: Função diferente para cada região
        - Resultado: Melhor realce de detalhes locais
        
    Aplicações:
        - Imagens médicas (raios-X, ressonância)
        - Fotografias com iluminação não-uniforme
        - Documentos antigos com manchas
        - Melhoria de contraste local preservando detalhes
    """
    if len(imagem.shape) != 2:
        raise ValueError("equalizar_histograma_local requer imagem em escala de cinza")
    
    altura, largura = imagem.shape
    resultado = np.copy(imagem)
    
    # Processa janelas não-sobrepostas
    for y in range(0, altura, tamanho_janela):
        for x in range(0, largura, tamanho_janela):
            # Define limites da janela
            y2 = min(y + tamanho_janela, altura)
            x2 = min(x + tamanho_janela, largura)
            
            # Extrai janela
            janela = imagem[y:y2, x:x2]
            
            # Aplica equalização na janela
            janela_eq = _equalizar_histograma_simples(janela)
            
            # Atualiza resultado
            resultado[y:y2, x:x2] = janela_eq
    
    return resultado


def _criar_mascara_peso(altura, largura, fade_size):
    """
    Cria máscara de peso com fade nas bordas para blend suave.
    """
    mascara = np.ones((altura, largura), dtype=np.float64)
    
    # Aplica fade nas bordas
    for i in range(fade_size):
        peso = i / fade_size
        # Bordas verticais
        if i < largura:
            mascara[:, i] *= peso
            mascara[:, -i-1] *= peso
        # Bordas horizontais  
        if i < altura:
            mascara[i, :] *= peso
            mascara[-i-1, :] *= peso
    
    return mascara


def _equalizar_histograma_simples(imagem):
    """
    Equalização de histograma básica usando CDF.
    """
    # Calcula histograma
    hist, bins = np.histogram(imagem.flatten(), 256, [0, 256])
    
    # Calcula CDF (Cumulative Distribution Function)
    cdf = hist.cumsum()
    cdf_normalizada = cdf * 255 / cdf[-1]  # Normaliza para [0-255]
    
    # Aplica transformação usando CDF como LUT (Look-Up Table)
    imagem_eq = np.interp(imagem.flatten(), bins[:-1], cdf_normalizada)
    
    return imagem_eq.reshape(imagem.shape).astype(np.uint8)


def calcular_metricas_normalizacao(imagem_original, imagem_normalizada):
    """
    Calcula métricas para avaliar qualidade da normalização.
    
    Returns:
        dict: Métricas detalhadas da normalização
    """
    return {
        'range_original': (int(imagem_original.min()), int(imagem_original.max())),
        'range_normalizada': (int(imagem_normalizada.min()), int(imagem_normalizada.max())),
        'utilizacao_faixa_original': (imagem_original.max() - imagem_original.min()) / 255,
        'utilizacao_faixa_normalizada': (imagem_normalizada.max() - imagem_normalizada.min()) / 255,
        'media_original': float(imagem_original.mean()),
        'media_normalizada': float(imagem_normalizada.mean()),
        'desvio_original': float(imagem_original.std()),
        'desvio_normalizada': float(imagem_normalizada.std()),
        'entropia_original': _calcular_entropia(imagem_original),
        'entropia_normalizada': _calcular_entropia(imagem_normalizada)
    }


def _calcular_entropia(imagem):
    """Calcula entropia da distribuição de intensidades."""
    hist, _ = np.histogram(imagem, bins=256, range=(0, 255))
    hist = hist / hist.sum()  # Normaliza para probabilidades
    hist = hist[hist > 0]  # Remove zeros para evitar log(0)
    return -np.sum(hist * np.log2(hist))
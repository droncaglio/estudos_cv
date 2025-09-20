# cv_lib/operacoes_pontuais/aritmetica.py
"""
🧮 Operações Aritméticas entre Imagens

📚 **REFERÊNCIAS ACADÊMICAS:**
- **Gonzalez & Woods, Digital Image Processing 4e, Seção 2.6, p.92**
  "Image multiplication for masking (ROI operations)" e shading correction
- **Szeliski, Computer Vision 2e, Seção 8.4.4, p.570**
  "Blending images to compensate for exposure differences"
- **Gonzalez & Woods, Fig.2.34, p.92**
  Masking operations: "ROI with 1's in ROI and 0's elsewhere"
- **Szeliski, Poisson image blending, Eq.8.75-8.77**
  Advanced blending techniques para composição

🧮 **OPERAÇÕES FUNDAMENTAIS:**

**1. Adição:** `f(x,y) = g₁(x,y) + g₂(x,y)`
- **Aplicação**: Combinação de imagens, noise averaging

**2. Subtração:** `f(x,y) = g₁(x,y) - g₂(x,y)`
- **Aplicação**: Background subtraction, motion detection

**3. Multiplicação:** `f(x,y) = g₁(x,y) × g₂(x,y)`
- **Aplicação**: Masking (ROI), shading correction (G&W p.92)

**4. Blending:** `f(x,y) = α×g₁(x,y) + (1-α)×g₂(x,y)`
- **Aplicação**: "Compensate for exposure differences" (Szeliski p.570)

🎯 **APLICAÇÕES ACADÊMICAS CITADAS:**
- **Shading correction**: "Using estimate of shading pattern" (G&W p.92)
- **ROI operations**: "Multiplying by mask with 1's in ROI" (G&W Fig.2.34)
- **Exposure compensation**: "Blending to compensate differences" (Szeliski p.570)
- **Motion detection**: Background subtraction para detecção
- **HDR imaging**: Multiple exposure fusion
- **Medical imaging**: Flat-field correction para uniformidade

⚠️ **CONSIDERAÇÕES DE IMPLEMENTAÇÃO:**
- **Overflow handling**: Clipping necessário para [0,255]
- **Data type promotion**: Use float64 durante cálculos
- **Shape compatibility**: Imagens devem ter mesmas dimensões
- **Feathering**: "High exponent in feathering" para transições (Szeliski)
"""

import numpy as np
from ..utils.validacao import validar_shapes_compativeis, validar_parametro_numerico, garantir_uint8


def operacao_entre_imagens(img1, img2, operacao='soma', peso1=1.0, peso2=1.0):
    """
    Realiza operações aritméticas entre duas imagens.
    
    Args:
        img1, img2: Arrays NumPy das imagens (devem ter o mesmo shape)
        operacao: Tipo de operação
            - 'soma': Adição pixel por pixel
            - 'subtracao': Subtração img1 - img2
            - 'multiplicacao': Multiplicação normalizada  
            - 'divisao': Divisão com proteção contra divisão por zero
            - 'media_ponderada': Média com pesos específicos
            - 'diferenca_absoluta': |img1 - img2|
            - 'maximo': max(img1, img2) pixel por pixel
            - 'minimo': min(img1, img2) pixel por pixel
        peso1, peso2: Pesos para as imagens (usado em soma e média ponderada)
        
    Returns:
        np.ndarray: Imagem resultante dtype uint8
        
    Aplicações por Operação:
        - soma: Combinação de imagens, acumulação temporal
        - subtracao: Background subtraction, detecção de mudanças  
        - multiplicacao: Aplicação de máscaras binárias ou suaves
        - divisao: Flat-field correction, normalização por referência
        - media_ponderada: Fusão controlada, transições suaves
        - diferenca_absoluta: Detecção de movimento, comparação
        - maximo/minimo: Seleção de valores extremos, combinação robusta
    """
    # Validações de entrada
    validar_shapes_compativeis(img1, img2, "img1", "img2")
    
    operacoes_validas = ['soma', 'subtracao', 'multiplicacao', 'divisao', 
                        'media_ponderada', 'diferenca_absoluta', 'maximo', 'minimo']
    if operacao not in operacoes_validas:
        raise ValueError(f"Operação '{operacao}' não reconhecida. Válidas: {operacoes_validas}")
    
    # Converte para float64 para evitar overflow durante cálculos
    img1_float = img1.astype(np.float64)
    img2_float = img2.astype(np.float64)
    
    if operacao == 'soma':
        # Soma ponderada: peso1×img1 + peso2×img2
        validar_parametro_numerico(peso1, "peso1", 0.0)
        validar_parametro_numerico(peso2, "peso2", 0.0)
        resultado = peso1 * img1_float + peso2 * img2_float
        
    elif operacao == 'subtracao':
        # Subtração: img1 - img2
        resultado = img1_float - img2_float
        
    elif operacao == 'multiplicacao':
        # Multiplicação elemento a elemento (normalizada)
        # Divide por 255 para manter range [0-255]
        resultado = (img1_float * img2_float) / 255.0
        
    elif operacao == 'divisao':
        # Divisão com proteção contra divisão por zero
        img2_safe = np.where(img2_float == 0, 1, img2_float)  # Substitui 0 por 1
        resultado = (img1_float / img2_safe) * 255.0
        
    elif operacao == 'media_ponderada':
        # Média ponderada normalizada pelos pesos
        validar_parametro_numerico(peso1, "peso1", 0.0)
        validar_parametro_numerico(peso2, "peso2", 0.0)
        
        soma_pesos = peso1 + peso2
        if soma_pesos == 0:
            return np.zeros_like(img1, dtype=np.uint8)
        
        resultado = (peso1 * img1_float + peso2 * img2_float) / soma_pesos
        
    elif operacao == 'diferenca_absoluta':
        # Valor absoluto da diferença
        resultado = np.abs(img1_float - img2_float)
        
    elif operacao == 'maximo':
        # Máximo pixel por pixel
        resultado = np.maximum(img1_float, img2_float)
        
    elif operacao == 'minimo':
        # Mínimo pixel por pixel
        resultado = np.minimum(img1_float, img2_float)
    
    # Garante valores no intervalo [0, 255] e converte para uint8
    resultado = np.clip(resultado, 0, 255)
    return garantir_uint8(resultado)


def blending_linear(img1, img2, alpha=0.5):
    """
    Combina duas imagens usando blending linear (alpha blending).
    
    Args:
        img1, img2: Imagens a serem combinadas
        alpha: Peso da primeira imagem [0-1]
            - alpha=0: 100% img2
            - alpha=0.5: 50% cada imagem
            - alpha=1: 100% img1
            
    Returns:
        np.ndarray: Imagem combinada
        
    Fórmula: resultado = alpha × img1 + (1-alpha) × img2
    
    Aplicações:
        - Transições suaves entre imagens
        - Sobreposição de elementos gráficos
        - Criação de panoramas
        - Efeitos de transparência
        - Fusão de múltiplas exposições
    """
    validar_parametro_numerico(alpha, "alpha", 0.0, 1.0)
    return operacao_entre_imagens(img1, img2, 'media_ponderada', peso1=alpha, peso2=1-alpha)


def subtracao_de_fundo(imagem_atual, imagem_fundo, limiar=30):
    """
    Detecção de movimento usando subtração de fundo.
    
    Args:
        imagem_atual: Imagem atual
        imagem_fundo: Imagem de referência (fundo)
        limiar: Threshold para binarização da diferença
        
    Returns:
        tuple: (diferenca, mascara_binaria)
        - diferenca: Diferença absoluta entre imagens
        - mascara_binaria: Máscara onde diferença > limiar
        
    Algoritmo:
        1. Calcula diferença absoluta
        2. Aplica limiar para criar máscara binária
        3. Pixels com diferença > limiar = movimento detectado
        
    Aplicações:
        - Segurança e vigilância
        - Detecção de intrusos
        - Análise de tráfego
        - Tracking de objetos
        - Contagem de pessoas/veículos
    """
    validar_shapes_compativeis(imagem_atual, imagem_fundo)
    validar_parametro_numerico(limiar, "limiar", 0, 255)
    
    # Calcula diferença absoluta
    diferenca = operacao_entre_imagens(imagem_atual, imagem_fundo, 'diferenca_absoluta')
    
    # Cria máscara binária baseada no limiar
    if len(diferenca.shape) == 3:
        # Para imagens RGB, usa intensidade máxima dos canais
        intensidade_diff = np.max(diferenca, axis=2)
    else:
        intensidade_diff = diferenca
    
    mascara_binaria = (intensidade_diff > limiar).astype(np.uint8) * 255
    
    return diferenca, mascara_binaria


def correcao_flat_field(imagem, flat_field, dark_field=None):
    """
    Correção de flat-field para remover não-uniformidades de iluminação.
    
    Args:
        imagem: Imagem a ser corrigida
        flat_field: Imagem de campo plano (iluminação uniforme)
        dark_field: Imagem de campo escuro (opcional, para correção de offset)
        
    Returns:
        np.ndarray: Imagem corrigida
        
    Fórmula:
        Com dark_field: (imagem - dark_field) / (flat_field - dark_field)
        Sem dark_field: imagem / flat_field
        
    Aplicações:
        - Microscopia (correção de iluminação não-uniforme)
        - Imagens médicas (correção de sensibilidade do detector)
        - Astronomia (correção de vinheting e dust spots)
        - Fotografia (correção de lens vignetting)
        
    Procedimento Típico:
        1. Capturar flat_field: Imagem de superfície uniformemente iluminada
        2. Capturar dark_field: Imagem com tampa/sem luz (offset do sensor)
        3. Aplicar correção na imagem real
    """
    validar_shapes_compativeis(imagem, flat_field)
    
    if dark_field is not None:
        validar_shapes_compativeis(imagem, dark_field)
        # Correção completa com dark field
        numerador = operacao_entre_imagens(imagem, dark_field, 'subtracao')
        denominador = operacao_entre_imagens(flat_field, dark_field, 'subtracao')
    else:
        # Correção simples sem dark field
        numerador = imagem
        denominador = flat_field
    
    # Aplica divisão para correção
    resultado = operacao_entre_imagens(numerador, denominador, 'divisao')
    
    return resultado


def fusao_multiplas_exposicoes(imagens, pesos=None):
    """
    Combina múltiplas exposições usando média ponderada.
    
    Args:
        imagens: Lista de arrays das imagens
        pesos: Lista de pesos (opcional, padrão = pesos iguais)
        
    Returns:
        np.ndarray: Imagem fusionada
        
    Técnica de HDR (High Dynamic Range) simples que:
        1. Combina múltiplas exposições
        2. Preserva detalhes em sombras E highlights
        3. Evita sobre/subexposição
        
    Aplicações:
        - Fotografia HDR
        - Imagens médicas com alto contraste
        - Microscopia com diferentes iluminações
        - Astronomia (stacking de imagens)
    """
    if not imagens:
        raise ValueError("Lista de imagens não pode estar vazia")
    
    if len(imagens) == 1:
        return imagens[0]
    
    # Valida shapes consistentes
    shape_ref = imagens[0].shape
    for i, img in enumerate(imagens):
        if img.shape != shape_ref:
            raise ValueError(f"Imagem {i} tem shape {img.shape}, esperado {shape_ref}")
    
    # Define pesos padrão se não fornecidos
    if pesos is None:
        pesos = [1.0] * len(imagens)
    
    if len(pesos) != len(imagens):
        raise ValueError(f"Número de pesos ({len(pesos)}) != número de imagens ({len(imagens)})")
    
    # Inicializa resultado e soma de pesos
    resultado = np.zeros_like(imagens[0], dtype=np.float64)
    soma_pesos = 0.0
    
    # Acumula imagens ponderadas
    for img, peso in zip(imagens, pesos):
        resultado += img.astype(np.float64) * peso
        soma_pesos += peso
    
    # Normaliza pelos pesos totais
    if soma_pesos > 0:
        resultado = resultado / soma_pesos
    
    return garantir_uint8(resultado)


def aplicar_mascara(imagem, mascara, valor_fundo=0):
    """
    Aplica máscara binária ou suave em uma imagem.
    
    Args:
        imagem: Imagem de entrada
        mascara: Máscara (0-255 ou 0.0-1.0)
        valor_fundo: Valor para regiões mascaradas
        
    Returns:
        np.ndarray: Imagem mascarada
        
    Tipos de Máscara:
        - Binária: 0 ou 255 (máscara rígida)
        - Suave: Gradiente 0-255 (transição suave)
        - Alpha: 0.0-1.0 float (transparência)
        
    Aplicações:
        - Segmentação de regiões de interesse
        - Remoção de background
        - Aplicação seletiva de filtros
        - Composição de imagens
        - Proteção de regiões durante processamento
    """
    validar_shapes_compativeis(imagem, mascara)
    
    # Normaliza máscara para [0,1] independente do tipo
    if mascara.dtype == np.uint8:
        mascara_norm = mascara.astype(np.float64) / 255.0
    else:
        mascara_norm = np.clip(mascara.astype(np.float64), 0.0, 1.0)
    
    # Aplica máscara
    imagem_float = imagem.astype(np.float64)
    resultado = imagem_float * mascara_norm + valor_fundo * (1 - mascara_norm)
    
    return garantir_uint8(resultado)


def detectar_movimento_temporal(imagens_sequencia, metodo='diferenca_consecutiva'):
    """
    Detecta movimento em sequência temporal de imagens.
    
    Args:
        imagens_sequencia: Lista de imagens em ordem temporal
        metodo: Método de detecção
            - 'diferenca_consecutiva': Diferença entre frames consecutivos
            - 'diferenca_referencia': Diferença com primeiro frame
            - 'background_running': Background adaptativo
            
    Returns:
        list: Lista de mapas de movimento (um para cada transição)
        
    Aplicações:
        - Detecção de movimento em vídeo
        - Tracking de objetos
        - Análise de comportamento
        - Segurança e vigilância
        - Estudos de dinâmica de fluidos
    """
    if len(imagens_sequencia) < 2:
        raise ValueError("Sequência deve ter pelo menos 2 imagens")
    
    mapas_movimento = []
    
    if metodo == 'diferenca_consecutiva':
        # Compara cada frame com o anterior
        for i in range(1, len(imagens_sequencia)):
            img_atual = imagens_sequencia[i]
            img_anterior = imagens_sequencia[i-1]
            
            diferenca = operacao_entre_imagens(img_atual, img_anterior, 'diferenca_absoluta')
            mapas_movimento.append(diferenca)
            
    elif metodo == 'diferenca_referencia':
        # Compara todos os frames com o primeiro
        img_referencia = imagens_sequencia[0]
        
        for i in range(1, len(imagens_sequencia)):
            img_atual = imagens_sequencia[i]
            diferenca = operacao_entre_imagens(img_atual, img_referencia, 'diferenca_absoluta')
            mapas_movimento.append(diferenca)
            
    elif metodo == 'background_running':
        # Background adaptativo (média móvel simples)
        background = imagens_sequencia[0].astype(np.float64)
        alpha_bg = 0.1  # Taxa de adaptação do background
        
        for i in range(1, len(imagens_sequencia)):
            img_atual = imagens_sequencia[i]
            
            # Detecta movimento
            diferenca = operacao_entre_imagens(img_atual, background.astype(np.uint8), 'diferenca_absoluta')
            mapas_movimento.append(diferenca)
            
            # Atualiza background adaptativamente
            background = alpha_bg * img_atual.astype(np.float64) + (1 - alpha_bg) * background
    
    else:
        raise ValueError(f"Método '{metodo}' não reconhecido")
    
    return mapas_movimento


def calcular_metricas_comparacao(img1, img2):
    """
    Calcula métricas para comparação quantitativa entre duas imagens.
    
    Returns:
        dict: Métricas de similaridade e diferença
    """
    validar_shapes_compativeis(img1, img2)
    
    # Converte para float para cálculos precisos
    img1_float = img1.astype(np.float64)
    img2_float = img2.astype(np.float64)
    
    # Diferença absoluta média (MAE - Mean Absolute Error)
    mae = np.mean(np.abs(img1_float - img2_float))
    
    # Erro quadrático médio (MSE - Mean Squared Error)
    mse = np.mean((img1_float - img2_float) ** 2)
    
    # PSNR (Peak Signal-to-Noise Ratio)
    if mse > 0:
        psnr = 10 * np.log10((255**2) / mse)
    else:
        psnr = float('inf')  # Imagens idênticas
    
    # Correlação normalizada
    mean1, mean2 = img1_float.mean(), img2_float.mean()
    std1, std2 = img1_float.std(), img2_float.std()
    
    if std1 > 0 and std2 > 0:
        correlacao = np.mean((img1_float - mean1) * (img2_float - mean2)) / (std1 * std2)
    else:
        correlacao = 0.0
    
    return {
        'mae': mae,
        'mse': mse,
        'psnr': psnr,
        'correlacao': correlacao,
        'ssim_simples': correlacao,  # Aproximação simples do SSIM
        'diferenca_maxima': np.max(np.abs(img1_float - img2_float)),
        'diferenca_media': np.mean(img1_float - img2_float),
        'pixels_identicos': np.sum(img1 == img2),
        'percentual_identicos': np.sum(img1 == img2) / img1.size * 100
    }
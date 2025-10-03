# cv_lib/filtragem/espacial.py
"""
📐 Filtragem Espacial - Convolução e Filtros Lineares

📚 **REFERÊNCIAS ACADÊMICAS:**
- **Gonzalez & Woods, Digital Image Processing 4e**
  - **Seção 3.4, p.159-161**: "Fundamentals of Spatial Filtering"
  - **Eq.3-35, p.159**: Convolução 2D
  - **Seção 3.5, p.167-173**: "Smoothing (Lowpass) Spatial Filters"
  - **Eq.3-45, p.169**: Filtro Gaussiano
  - **Seção 3.6, p.179-196**: "Sharpening (Highpass) Spatial Filters"
  - **Cap.10, p.720-722**: Gradiente (Sobel, Prewitt)

🧮 **DEFINIÇÕES MATEMÁTICAS:**

**1. Convolução 2D (Gonzalez & Woods Eq.3-35):**
```
(w ⊛ f)(x,y) = Σ Σ w(s,t) × f(x-s, y-t)
              s  t
```
Onde:
- w(s,t) = kernel/máscara de tamanho m×n
- f(x,y) = imagem de entrada
- Kernel é rotacionado 180° antes da operação

**2. Filtro Gaussiano (G&W Eq.3-45):**
```
G(s,t) = K × e^(-(s² + t²)/(2σ²))
```
Onde σ controla a largura da gaussiana (suavização)

**3. Gradiente (G&W Cap.10, Eq.10-17):**
```
∇f = [gx, gy]  onde gx = ∂f/∂x, gy = ∂f/∂y
|∇f| = √(gx² + gy²)
```

⚠️ **IMPLEMENTAÇÃO:**
- Convolução manual com loops educacionais
- Padding para bordas
- Tratamento de tipos de dados (uint8/float)
"""

import numpy as np
from ..utils.validacao import garantir_uint8


def convolucao_2d(imagem, kernel, padding='same', modo_borda='reflect'):
    """
    Realiza convolução 2D entre imagem e kernel.

    📚 **Referência:** Gonzalez & Woods, Digital Image Processing 4e
    - **Equação 3-35, p.159**: "(w ⊛ f)(x,y) = Σ Σ w(s,t) f(x-s, y-t)"
    - **Seção 3.4.1, p.159-161**: "Linear spatial filtering and spatial convolution are synonymous"
    - **Citação direta (p.159)**: "This equation implements the sum of products process
      to which we refer throughout the book as linear spatial filtering"

    🧮 **Fórmula Matemática (Gonzalez & Woods Eq.3-35):**
    ```
    (w ⊛ f)(x,y) = Σ(s=-a to a) Σ(t=-b to b) w(s,t) × f(x-s, y-t)
    ```
    Onde:
    - w = kernel de tamanho (2a+1) × (2b+1)
    - f = imagem
    - Kernel é **rotacionado 180°** (diferença da correlação)

    🎯 **Aplicações Citadas (G&W p.159-196):**
    - **Smoothing/Blurring**: Redução de ruído, suavização
    - **Sharpening**: Realce de bordas e detalhes
    - **Edge detection**: Detecção de transições de intensidade
    - **Feature extraction**: Extração de características locais

    🔢 **Tipos de Padding (G&W p.160):**
    - **'same'**: Mantém tamanho original (adiciona padding)
    - **'valid'**: Apenas pixels válidos (reduz tamanho)

    Args:
        imagem: Array NumPy (H, W) ou (H, W, C)
        kernel: Array NumPy do kernel/máscara (k_h, k_w)
        padding: 'same' ou 'valid'
        modo_borda: Modo de padding ('reflect', 'constant', 'edge')

    Returns:
        np.ndarray: Imagem convoluída

    Algoritmo (G&W p.159):
        1. Rotacionar kernel 180° (diferença da correlação)
        2. Adicionar padding se necessário
        3. Para cada pixel (x,y):
           - Posicionar centro do kernel em (x,y)
           - Multiplicar elementos correspondentes
           - Somar produtos → valor de saída
    """
    # Converte para float para operações
    if imagem.dtype == np.uint8:
        imagem = imagem.astype(np.float64)

    kernel = kernel.astype(np.float64)

    # Rotaciona kernel 180° (convolução vs correlação)
    kernel_rot = np.rot90(kernel, 2)

    # Dimensões
    if len(imagem.shape) == 2:
        img_h, img_w = imagem.shape
        num_canais = 1
        imagem = imagem[:, :, np.newaxis]
    else:
        img_h, img_w, num_canais = imagem.shape

    k_h, k_w = kernel_rot.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    # Aplica padding
    if padding == 'same':
        imagem_pad = np.pad(
            imagem,
            ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode=modo_borda
        )
    else:  # valid
        imagem_pad = imagem

    # Calcula tamanho de saída
    if padding == 'same':
        out_h, out_w = img_h, img_w
    else:
        out_h = img_h - k_h + 1
        out_w = img_w - k_w + 1

    # Inicializa saída
    resultado = np.zeros((out_h, out_w, num_canais), dtype=np.float64)

    # Convolução manual (educacional)
    for c in range(num_canais):
        for y in range(out_h):
            for x in range(out_w):
                # Extrai região da imagem
                regiao = imagem_pad[y:y+k_h, x:x+k_w, c]
                # Produto elemento a elemento e soma
                resultado[y, x, c] = np.sum(regiao * kernel_rot)

    # Remove dimensão extra se entrada era 2D
    if num_canais == 1:
        resultado = resultado[:, :, 0]

    return resultado


def correlacao_2d(imagem, kernel, padding='same', modo_borda='reflect'):
    """
    Realiza correlação 2D (convolução sem rotação do kernel).

    📚 **Referência:** Gonzalez & Woods Eq.3-34, p.158-159
    "Correlation: (w ⊙ f)(x,y) = Σ Σ w(s,t) f(x+s, y+t)"

    Diferença da convolução: kernel NÃO é rotacionado.

    Args:
        imagem: Array NumPy
        kernel: Array NumPy do kernel
        padding: 'same' ou 'valid'
        modo_borda: Modo de padding

    Returns:
        np.ndarray: Imagem correlacionada
    """
    # Correlação = convolução sem rotação
    # Então rotamos o kernel antes de passar para convolucao_2d
    kernel_rot = np.rot90(kernel, 2)
    return convolucao_2d(imagem, kernel_rot, padding, modo_borda)


# ============================================================================
# FILTROS DE SUAVIZAÇÃO (LOWPASS)
# ============================================================================

def filtro_media(tamanho=(3, 3)):
    """
    Cria kernel de filtro de média (box filter).

    📚 **Referência:** Gonzalez & Woods, Seção 3.5.1, p.166-167
    "Box filter: simple averaging filter"

    Kernel:
    ```
    1/(m×n) × [1 1 ... 1]
                [1 1 ... 1]
                [... ... ...]
    ```

    Args:
        tamanho: Tupla (altura, largura) do kernel

    Returns:
        np.ndarray: Kernel normalizado
    """
    m, n = tamanho
    kernel = np.ones((m, n), dtype=np.float64) / (m * n)
    return kernel


def filtro_gaussiano(tamanho=(5, 5), sigma=1.0):
    """
    Cria kernel de filtro Gaussiano.

    📚 **Referência:** Gonzalez & Woods, Digital Image Processing 4e
    - **Equação 3-45, p.169**: "G(s,t) = K × e^(-(s²+t²)/(2σ²))"
    - **Citação (p.169)**: "Gaussian kernels are the only circularly symmetric
      kernels that are also separable"

    🧮 **Fórmula (G&W Eq.3-45):**
    ```
    G(s,t) = K × e^(-(s² + t²)/(2σ²))
    ```
    Onde:
    - σ = desvio padrão (controla largura da gaussiana)
    - K = constante de normalização (soma = 1)
    - (s,t) = coordenadas relativas ao centro

    🎯 **Aplicações (G&W p.167-173):**
    - **Noise reduction**: Redução de ruído gaussiano
    - **Image blurring**: Suavização preservando bordas
    - **Preprocessing**: Pré-processamento para detecção de bordas

    🔢 **Parâmetros Típicos:**
    - σ = 0.5 a 1.0: Suavização leve
    - σ = 1.0 a 2.0: Suavização moderada
    - σ = 2.0 a 5.0: Suavização forte (blur)

    Args:
        tamanho: Tupla (altura, largura) - deve ser ímpar
        sigma: Desvio padrão da gaussiana

    Returns:
        np.ndarray: Kernel gaussiano normalizado
    """
    m, n = tamanho
    # Centro do kernel
    centro_y = m // 2
    centro_x = n // 2

    # Cria grade de coordenadas
    kernel = np.zeros((m, n), dtype=np.float64)

    for y in range(m):
        for x in range(n):
            # Distância do centro
            s = x - centro_x
            t = y - centro_y

            # Fórmula gaussiana (G&W Eq.3-45)
            kernel[y, x] = np.exp(-(s**2 + t**2) / (2 * sigma**2))

    # Normaliza (soma = 1)
    kernel = kernel / np.sum(kernel)

    return kernel


# ============================================================================
# FILTROS DE AGUÇAMENTO (HIGHPASS) - GRADIENTE
# ============================================================================

def filtro_sobel(direcao='ambos'):
    """
    Retorna kernels do operador Sobel para detecção de bordas.

    📚 **Referência:** Gonzalez & Woods, Fig.3.50 e Cap.10 p.720-722
    Sobel operator para aproximação do gradiente

    Kernels Sobel 3×3:
    ```
    Gx (horizontal):        Gy (vertical):
    [-1  0  +1]            [-1 -2 -1]
    [-2  0  +2]            [ 0  0  0]
    [-1  0  +1]            [+1 +2 +1]
    ```

    Magnitude: |∇f| = √(Gx² + Gy²)
    Direção: θ = arctan(Gy/Gx)

    Args:
        direcao: 'x', 'y', ou 'ambos'

    Returns:
        np.ndarray ou tuple: Kernel(s) Sobel
    """
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)

    if direcao == 'x':
        return sobel_x
    elif direcao == 'y':
        return sobel_y
    else:
        return sobel_x, sobel_y


def filtro_prewitt(direcao='ambos'):
    """
    Retorna kernels do operador Prewitt.

    📚 **Referência:** Gonzalez & Woods, Fig.3.50 e Cap.10
    Similar ao Sobel mas com pesos uniformes

    Kernels Prewitt 3×3:
    ```
    Gx:                Gy:
    [-1  0  +1]       [-1 -1 -1]
    [-1  0  +1]       [ 0  0  0]
    [-1  0  +1]       [+1 +1 +1]
    ```

    Args:
        direcao: 'x', 'y', ou 'ambos'

    Returns:
        np.ndarray ou tuple: Kernel(s) Prewitt
    """
    prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float64)

    prewitt_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float64)

    if direcao == 'x':
        return prewitt_x
    elif direcao == 'y':
        return prewitt_y
    else:
        return prewitt_x, prewitt_y


def filtro_laplaciano(tipo='basico'):
    """
    Retorna kernel Laplaciano (derivada de 2ª ordem).

    📚 **Referência:** Gonzalez & Woods, Seção 3.6.2, p.182-185
    Eq.3-51: "∇²f = ∂²f/∂x² + ∂²f/∂y²"

    Kernels Laplaciano:
    ```
    Básico:          Diagonal:
    [ 0 -1  0]       [-1 -1 -1]
    [-1  4 -1]       [-1  8 -1]
    [ 0 -1  0]       [-1 -1 -1]
    ```

    🎯 **Aplicações (G&W p.182-185):**
    - **Edge enhancement**: Realce de bordas
    - **Image sharpening**: Aguçamento de detalhes
    - Resposta zero em regiões constantes

    ⚠️ **Limitação:** Sensível a ruído (derivada de 2ª ordem)

    Args:
        tipo: 'basico' ou 'diagonal'

    Returns:
        np.ndarray: Kernel Laplaciano
    """
    if tipo == 'basico':
        laplaciano = np.array([
            [ 0, -1,  0],
            [-1,  4, -1],
            [ 0, -1,  0]
        ], dtype=np.float64)
    else:  # diagonal
        laplaciano = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float64)

    return laplaciano


def unsharp_masking(imagem, sigma=1.0, quantidade=1.5):
    """
    Unsharp masking - técnica de aguçamento por subtração.

    📚 **Referência:** Gonzalez & Woods, Seção 3.6.3, p.186-190
    "Sharpening by high-boost filtering"

    Algoritmo:
    1. Suavizar imagem: f_suave = G(σ) ⊛ f
    2. Máscara = f - f_suave
    3. Aguçada = f + α × máscara

    Args:
        imagem: Imagem de entrada
        sigma: Desvio padrão do Gaussiano
        quantidade: Fator de aguçamento (α)

    Returns:
        np.ndarray: Imagem aguçada
    """
    # Suaviza com Gaussiano
    kernel_gauss = filtro_gaussiano(tamanho=(5, 5), sigma=sigma)
    imagem_suave = convolucao_2d(imagem, kernel_gauss)

    # Máscara (detalhes de alta frequência)
    mascara = imagem - imagem_suave

    # Aguçamento
    imagem_agucada = imagem + quantidade * mascara

    return imagem_agucada

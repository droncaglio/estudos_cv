# cv_lib/analise/histogramas.py
"""
📊 Histogramas e Análise Estatística de Imagens

📚 **REFERÊNCIAS ACADÊMICAS:**
- **Gonzalez & Woods, Digital Image Processing 4e, Seção 3.3, p.154**
  "Histogram: h(rₖ) = nₖ, where nₖ is the number of pixels with intensity rₖ"
- **Gonzalez & Woods, Eq.3-10, p.154**
  Definição matemática formal de histograma digital
- **Gonzalez & Woods, Eq.3-11, p.155**
  Histograma normalizado: p(rₖ) = nₖ/MN para interpretação probabilística
- **Szeliski, Computer Vision 2e, Seção 3.1.4, p.141**
  "Histograms reveal important characteristics about the image content"
- **Jain, A.K. (1989). Fundamentals of Digital Image Processing**
  Statistical measures derivation from histograms

🧮 **DEFINIÇÕES MATEMÁTICAS:**

**1. Histograma Básico (Gonzalez & Woods Eq.3-10):**
```
h(rₖ) = nₖ
```
Onde rₖ é o k-ésimo nível de intensidade e nₖ é o número de pixels com intensidade rₖ

**2. Histograma Normalizado (Gonzalez & Woods Eq.3-11):**
```
p(rₖ) = nₖ/(M×N)
```
Interpretação probabilística: p(rₖ) é a probabilidade de ocorrência do nível rₖ

**3. Histograma Cumulativo:**
```
c(rₖ) = Σ(j=0 to k) h(rⱼ)
```

**4. Entropia (Shannon, 1948):**
```
H = -Σ p(rₖ) × log₂(p(rₖ))
```

🎯 **APLICAÇÕES ACADÊMICAS CITADAS:**
- **Image quality assessment**: "Histogram shape indicates exposure quality" (G&W p.154)
- **Segmentation preprocessing**: "Bimodal histograms suggest clear object-background separation"
- **Contrast analysis**: "Histogram spread correlates with image contrast" (Szeliski p.141)
- **Automatic exposure**: "Histogram-based algorithms for camera systems"
- **Content analysis**: "Statistical measures for image similarity" (Jain 1989)

⚠️ **LIMITAÇÕES IDENTIFICADAS:**
- **Spatial information loss**: "Histograms discard spatial relationships" (Szeliski p.141)
- **Different images, same histogram**: Não é uma representação única
- **Noise sensitivity**: Ruído pode alterar significativamente a distribuição
"""

import numpy as np
from ..utils.validacao import validar_imagem_rgb, garantir_uint8
import matplotlib.pyplot as plt


def calcular_histograma(imagem, bins=256, range_vals=(0, 255)):
    """
    Calcula o histograma de uma imagem implementando manualmente a contagem.

    Args:
        imagem: Array NumPy da imagem (escala de cinza ou RGB)
        bins: Número de bins do histograma (padrão 256)
        range_vals: Tupla (min, max) dos valores considerados

    Returns:
        np.ndarray: Histograma com contagens para cada intensidade

    📚 Referência: Gonzalez & Woods Eq.3-10, p.154
    Implementação manual conforme definição matemática h(rₖ) = nₖ

    Aplicações:
        - Análise de distribuição de intensidades
        - Pré-processamento para equalização
        - Avaliação de qualidade de exposição
        - Base para métricas estatísticas

    Algoritmo:
        1. Inicializa array de contagem com zeros
        2. Para cada pixel, incrementa bin correspondente
        3. Retorna distribuição final de frequências
    """
    if not isinstance(imagem, np.ndarray):
        raise ValueError("imagem deve ser um array NumPy")

    # Se imagem é RGB, converte para escala de cinza
    if len(imagem.shape) == 3:
        # Usa conversão BT.601 (implementada em módulo rgb)
        from ..espacos_cor.rgb import rgb_para_cinza_bt601
        imagem = rgb_para_cinza_bt601(imagem)

    # Garante que é uint8
    if imagem.dtype != np.uint8:
        imagem = garantir_uint8(imagem)

    # Inicializa histograma
    histograma = np.zeros(bins, dtype=np.int32)

    # Calcula largura do bin
    min_val, max_val = range_vals
    largura_bin = (max_val - min_val) / bins

    # Conta pixels manualmente (implementação educacional)
    altura, largura = imagem.shape
    for y in range(altura):
        for x in range(largura):
            pixel_val = imagem[y, x]

            # Determina qual bin corresponde a este pixel
            if min_val <= pixel_val <= max_val:
                bin_idx = int((pixel_val - min_val) / largura_bin)
                # Garante que não saia dos limites
                bin_idx = min(bin_idx, bins - 1)
                histograma[bin_idx] += 1

    return histograma


def histograma_normalizado(histograma, total_pixels=None):
    """
    Converte histograma para forma normalizada (interpretação probabilística).

    Args:
        histograma: Array com contagens do histograma
        total_pixels: Total de pixels (se None, calcula da soma)

    Returns:
        np.ndarray: Histograma normalizado p(rₖ) = nₖ/(M×N)

    📚 Referência: Gonzalez & Woods Eq.3-11, p.155
    "Normalized histogram provides probability interpretation"

    Interpretação:
        - p(rₖ): Probabilidade de um pixel ter intensidade rₖ
        - Soma de todos os valores = 1.0
        - Útil para comparação entre imagens de tamanhos diferentes
    """
    if total_pixels is None:
        total_pixels = np.sum(histograma)

    if total_pixels == 0:
        return histograma.astype(np.float64)

    return histograma.astype(np.float64) / total_pixels


def histograma_cumulativo(histograma):
    """
    Calcula histograma cumulativo a partir do histograma básico.

    Args:
        histograma: Array com contagens do histograma

    Returns:
        np.ndarray: Histograma cumulativo c(rₖ) = Σ h(rⱼ)

    📚 Referência: Gonzalez & Woods, Seção 3.4.1, p.160
    "Cumulative histogram is fundamental for histogram equalization"

    Aplicações:
        - Base para equalização de histograma
        - Análise de distribuição acumulada
        - Implementação de transformações baseadas em CDF
        - Percentis e quantis de intensidade
    """
    return np.cumsum(histograma)


def calcular_estatisticas_basicas(imagem):
    """
    Calcula estatísticas básicas da imagem a partir dos valores dos pixels.

    Args:
        imagem: Array NumPy da imagem

    Returns:
        dict: Estatísticas básicas (média, variância, desvio, etc.)

    📚 Referência: Jain, A.K. (1989) - "Statistical measures for image analysis"
    Gonzalez & Woods, Seção 3.3.2, p.157 - "Statistical moments"

    Estatísticas Calculadas:
        - Média (μ): Brilho médio da imagem
        - Variância (σ²): Medida de contraste
        - Desvio padrão (σ): Espalhamento dos valores
        - Assimetria: Tendência da distribuição
        - Curtose: "Pico" da distribuição
    """
    if len(imagem.shape) == 3:
        from ..espacos_cor.rgb import rgb_para_cinza_bt601
        imagem = rgb_para_cinza_bt601(imagem)

    # Converte para float para cálculos precisos
    pixels = imagem.flatten().astype(np.float64)

    # Estatísticas básicas
    media = np.mean(pixels)
    variancia = np.var(pixels)
    desvio_padrao = np.std(pixels)

    # Momentos de ordem superior
    # Assimetria (skewness) - simetria da distribuição
    momento_3 = np.mean((pixels - media) ** 3)
    assimetria = momento_3 / (desvio_padrao ** 3) if desvio_padrao > 0 else 0

    # Curtose - "pico" da distribuição
    momento_4 = np.mean((pixels - media) ** 4)
    curtose = momento_4 / (desvio_padrao ** 4) - 3 if desvio_padrao > 0 else 0

    return {
        'media': media,
        'variancia': variancia,
        'desvio_padrao': desvio_padrao,
        'minimo': float(np.min(pixels)),
        'maximo': float(np.max(pixels)),
        'amplitude': float(np.max(pixels) - np.min(pixels)),
        'mediana': float(np.median(pixels)),
        'assimetria': assimetria,
        'curtose': curtose,
        'total_pixels': len(pixels)
    }


def calcular_entropia(imagem, bins=256):
    """
    Calcula a entropia de Shannon da imagem.

    Args:
        imagem: Array NumPy da imagem
        bins: Número de bins para o histograma

    Returns:
        float: Entropia em bits

    📚 Referência: Shannon, C.E. (1948). "A Mathematical Theory of Communication"
    Gonzalez & Woods, p.157 - "Entropy as measure of information content"

    Interpretação:
        - Entropia alta: Imagem com muita variação (textura rica)
        - Entropia baixa: Imagem uniforme (pouca informação)
        - Máximo teórico: log₂(bins) bits para distribuição uniforme

    Aplicações:
        - Medida de complexidade da imagem
        - Avaliação de conteúdo informacional
        - Critério para compressão
        - Análise de textura
    """
    # Calcula histograma
    hist = calcular_histograma(imagem, bins=bins)

    # Normaliza para probabilidades
    prob = histograma_normalizado(hist)

    # Remove zeros para evitar log(0)
    prob_nonzero = prob[prob > 0]

    # Calcula entropia: H = -Σ p(x) * log₂(p(x))
    entropia = -np.sum(prob_nonzero * np.log2(prob_nonzero))

    return entropia


def calcular_percentis(imagem, percentis=[25, 50, 75, 90, 95, 99]):
    """
    Calcula percentis da distribuição de intensidades.

    Args:
        imagem: Array NumPy da imagem
        percentis: Lista de percentis a calcular

    Returns:
        dict: Valores dos percentis especificados

    📚 Referência: Szeliski p.141 - "Robust statistics for image analysis"

    Aplicações:
        - Normalização robusta (percentis 2-98%)
        - Detecção de outliers
        - Análise de distribuição
        - Ajuste automático de contraste
    """
    if len(imagem.shape) == 3:
        from ..espacos_cor.rgb import rgb_para_cinza_bt601
        imagem = rgb_para_cinza_bt601(imagem)

    pixels = imagem.flatten()

    resultado = {}
    for p in percentis:
        resultado[f'p{p}'] = np.percentile(pixels, p)

    return resultado


def comparar_histogramas(hist1, hist2, metrica='chi_squared'):
    """
    Compara dois histogramas usando diferentes métricas de distância.

    Args:
        hist1, hist2: Arrays dos histogramas a comparar
        metrica: Tipo de métrica ('chi_squared', 'correlation', 'intersection')

    Returns:
        float: Valor da métrica de comparação

    📚 Referência: Swain & Ballard (1991). "Color indexing"
    Rubner et al. (2000). "Earth Mover's Distance"

    Métricas Implementadas:
        - Chi-squared: Teste estatístico de similaridade
        - Correlation: Correlação de Pearson entre histogramas
        - Intersection: Área de sobreposição entre histogramas

    Aplicações:
        - Busca por similaridade em bancos de imagens
        - Detecção de mudanças entre frames
        - Avaliação de qualidade de processamento
    """
    # Normaliza histogramas
    h1_norm = histograma_normalizado(hist1)
    h2_norm = histograma_normalizado(hist2)

    if metrica == 'chi_squared':
        # Chi-squared distance
        # χ² = Σ (h1(i) - h2(i))² / (h1(i) + h2(i))
        denominator = h1_norm + h2_norm
        valid_bins = denominator > 0

        if not np.any(valid_bins):
            return float('inf')

        chi_sq = np.sum((h1_norm[valid_bins] - h2_norm[valid_bins])**2 / denominator[valid_bins])
        return chi_sq

    elif metrica == 'correlation':
        # Correlação de Pearson
        return np.corrcoef(h1_norm, h2_norm)[0, 1]

    elif metrica == 'intersection':
        # Histogram intersection
        return np.sum(np.minimum(h1_norm, h2_norm))

    else:
        raise ValueError(f"Métrica '{metrica}' não reconhecida")


def histograma_conjunto_rgb(imagem_rgb):
    """
    Calcula histogramas separados para cada canal RGB.

    Args:
        imagem_rgb: Array RGB shape (altura, largura, 3)

    Returns:
        dict: Histogramas para cada canal {'r': hist_r, 'g': hist_g, 'b': hist_b}

    📚 Referência: Gonzalez & Woods p.416 - "Color histogram analysis"

    Aplicações:
        - Análise de balance de cores
        - Detecção de dominância cromática
        - Segmentação baseada em cor
        - Correção de white balance
    """
    altura, largura, canais = validar_imagem_rgb(imagem_rgb, "imagem_rgb")

    histogramas = {}
    nomes_canais = ['r', 'g', 'b']

    for i, canal in enumerate(nomes_canais):
        canal_data = imagem_rgb[:, :, i]
        hist_canal = calcular_histograma(canal_data)
        histogramas[canal] = hist_canal

    return histogramas


def plotar_histograma_detalhado(imagem, titulo="Análise de Histograma", bins=256, figsize=(15, 10)):
    """
    Cria visualização completa com histograma e estatísticas.

    Args:
        imagem: Array NumPy da imagem
        titulo: Título da visualização
        bins: Número de bins do histograma
        figsize: Tamanho da figura

    📚 Referência: Gonzalez & Woods Fig.3.12, p.156
    "Comprehensive histogram analysis visualization"

    Componentes da Visualização:
        - Imagem original
        - Histograma básico
        - Histograma cumulativo
        - Estatísticas principais
        - Interpretação visual
    """
    # Calcula dados necessários
    if len(imagem.shape) == 3:
        from ..espacos_cor.rgb import rgb_para_cinza_bt601
        img_cinza = rgb_para_cinza_bt601(imagem)
        is_rgb = True
    else:
        img_cinza = imagem
        is_rgb = False

    hist = calcular_histograma(img_cinza, bins=bins)
    hist_cum = histograma_cumulativo(hist)
    hist_norm = histograma_normalizado(hist)
    stats = calcular_estatisticas_basicas(img_cinza)
    entropia = calcular_entropia(img_cinza, bins=bins)

    # Cria visualização
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # Imagem original
    if is_rgb:
        axes[0, 0].imshow(imagem)
        axes[0, 0].set_title("Imagem RGB Original")
    else:
        axes[0, 0].imshow(imagem, cmap='gray')
        axes[0, 0].set_title("Imagem em Escala de Cinza")
    axes[0, 0].axis('off')

    # Histograma básico
    x_vals = np.arange(bins)
    axes[0, 1].bar(x_vals, hist, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    axes[0, 1].set_title("Histograma\n(Contagens absolutas)")
    axes[0, 1].set_xlabel("Intensidade")
    axes[0, 1].set_ylabel("Frequência")
    axes[0, 1].grid(True, alpha=0.3)

    # Histograma normalizado
    axes[0, 2].bar(x_vals, hist_norm, alpha=0.7, color='green', edgecolor='black', linewidth=0.5)
    axes[0, 2].set_title("Histograma Normalizado\n(Probabilidades)")
    axes[0, 2].set_xlabel("Intensidade")
    axes[0, 2].set_ylabel("Probabilidade")
    axes[0, 2].grid(True, alpha=0.3)

    # Histograma cumulativo
    axes[1, 0].plot(x_vals, hist_cum, color='red', linewidth=2)
    axes[1, 0].set_title("Histograma Cumulativo\n(CDF)")
    axes[1, 0].set_xlabel("Intensidade")
    axes[1, 0].set_ylabel("Pixels Acumulados")
    axes[1, 0].grid(True, alpha=0.3)

    # Estatísticas
    axes[1, 1].axis('off')
    stats_text = f"""📊 Estatísticas (Gonzalez & Woods p.157):

Média (μ): {stats['media']:.1f}
Desvio Padrão (σ): {stats['desvio_padrao']:.1f}
Variância (σ²): {stats['variancia']:.1f}
Mediana: {stats['mediana']:.1f}
Min/Max: {stats['minimo']:.0f}/{stats['maximo']:.0f}
Amplitude: {stats['amplitude']:.0f}
Assimetria: {stats['assimetria']:.3f}
Curtose: {stats['curtose']:.3f}
Entropia: {entropia:.3f} bits

Total Pixels: {stats['total_pixels']:,}"""

    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # Interpretação
    axes[1, 2].axis('off')

    # Análise automática da distribuição
    if stats['media'] < 85:
        brightness_analysis = "Imagem ESCURA"
    elif stats['media'] > 170:
        brightness_analysis = "Imagem CLARA"
    else:
        brightness_analysis = "Imagem BALANCEADA"

    if stats['desvio_padrao'] < 30:
        contrast_analysis = "Baixo CONTRASTE"
    elif stats['desvio_padrao'] > 70:
        contrast_analysis = "Alto CONTRASTE"
    else:
        contrast_analysis = "Contraste MÉDIO"

    if entropia > 6:
        content_analysis = "Rico em DETALHES"
    elif entropia < 4:
        content_analysis = "UNIFORME/Simples"
    else:
        content_analysis = "Complexidade MÉDIA"

    interpretacao_text = f"""🔍 Interpretação Automática:

🔆 Brilho: {brightness_analysis}
🎚️ Contraste: {contrast_analysis}
🖼️ Conteúdo: {content_analysis}

📚 Fundamentação:
• Média < 85: Subexposta
• Média > 170: Superexposta
• σ < 30: Pouco contraste
• σ > 70: Alto contraste
• H > 6: Rica em informação
• H < 4: Pouca informação

Qualidade da distribuição:
{'✅ Boa distribuição' if 50 < stats['media'] < 200 and stats['desvio_padrao'] > 40 else '⚠️ Distribuição limitada'}"""

    axes[1, 2].text(0.05, 0.95, interpretacao_text, transform=axes[1, 2].transAxes,
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f"{titulo}\n(Análise baseada em Gonzalez & Woods Cap.3)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return {
        'histograma': hist,
        'histograma_cumulativo': hist_cum,
        'histograma_normalizado': hist_norm,
        'estatisticas': stats,
        'entropia': entropia
    }


def plotar_comparacao_histogramas(imagens_dict, bins=256, figsize=(16, 10)):
    """
    Compara histogramas de múltiplas imagens lado a lado.

    Args:
        imagens_dict: Dict {'nome': imagem} com imagens a comparar
        bins: Número de bins dos histogramas
        figsize: Tamanho da figura

    📚 Referência: Gonzalez & Woods p.158 - "Histogram comparison analysis"

    Útil para:
        - Comparar efeitos de diferentes processamentos
        - Analisar características de diferentes imagens
        - Validar transformações aplicadas
    """
    n_imagens = len(imagens_dict)
    fig, axes = plt.subplots(2, n_imagens, figsize=figsize)

    if n_imagens == 1:
        axes = axes.reshape(2, 1)

    histogramas = {}
    estatisticas = {}

    for i, (nome, imagem) in enumerate(imagens_dict.items()):
        # Converte para escala de cinza se necessário
        if len(imagem.shape) == 3:
            from ..espacos_cor.rgb import rgb_para_cinza_bt601
            img_cinza = rgb_para_cinza_bt601(imagem)
            axes[0, i].imshow(imagem)
        else:
            img_cinza = imagem
            axes[0, i].imshow(imagem, cmap='gray')

        axes[0, i].set_title(f"{nome}")
        axes[0, i].axis('off')

        # Calcula e plota histograma
        hist = calcular_histograma(img_cinza, bins=bins)
        stats = calcular_estatisticas_basicas(img_cinza)

        histogramas[nome] = hist
        estatisticas[nome] = stats

        axes[1, i].bar(np.arange(bins), hist, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[1, i].set_title(f"Histograma - {nome}\nμ={stats['media']:.1f}, σ={stats['desvio_padrao']:.1f}")
        axes[1, i].set_xlabel("Intensidade")
        axes[1, i].set_ylabel("Frequência")
        axes[1, i].grid(True, alpha=0.3)

    plt.suptitle("Comparação de Histogramas entre Imagens", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Análise de similaridade se há mais de uma imagem
    if n_imagens > 1:
        print("\n📊 Análise de Similaridade entre Histogramas:")
        print("=" * 60)

        nomes = list(imagens_dict.keys())
        for i in range(len(nomes)):
            for j in range(i+1, len(nomes)):
                nome1, nome2 = nomes[i], nomes[j]

                # Calcula diferentes métricas
                chi_sq = comparar_histogramas(histogramas[nome1], histogramas[nome2], 'chi_squared')
                corr = comparar_histogramas(histogramas[nome1], histogramas[nome2], 'correlation')
                intersection = comparar_histogramas(histogramas[nome1], histogramas[nome2], 'intersection')

                print(f"{nome1} ↔ {nome2}:")
                print(f"  Chi-squared: {chi_sq:.4f} (menor = mais similar)")
                print(f"  Correlação:  {corr:.4f} (1.0 = idênticos)")
                print(f"  Interseção:  {intersection:.4f} (1.0 = idênticos)")
                print()

    return histogramas, estatisticas
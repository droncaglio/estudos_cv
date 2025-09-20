# cv_lib/operacoes_pontuais/gama.py
"""
🌊 Correção Gama - Transformação Power-Law

📚 **REFERÊNCIAS ACADÊMICAS:**
- **Gonzalez & Woods, Digital Image Processing 4e, Cap.3, Eq.3-5, p.128**
  "Power-law transformations are useful for general-purpose contrast manipulation"
- **Szeliski, Computer Vision 2e, Seção 3.1.1, p.137**
  "Point processes" - transformações pixel-a-pixel independentes
- **Poynton, C. (2003). Digital Video and HDTV Algorithms**
  Correção gama para dispositivos de exibição

🧮 **FÓRMULA MATEMÁTICA (Eq.3-5, Gonzalez & Woods):**
s = c × r^γ

Onde:
- s = intensidade de saída [0-255]
- r = intensidade de entrada normalizada [0-1]
- c = constante multiplicativa (normalmente 1.0)
- γ = exponente que controla a curvatura

📈 **COMPORTAMENTO DA FUNÇÃO POWER-LAW:**
- γ < 1: Curva côncava (expande tons escuros, clareia sombras)
- γ = 1: Linear (sem alteração)
- γ > 1: Curva convexa (comprime tons claros, escurece highlights)

🎯 **APLICAÇÕES CITADAS NA LITERATURA:**
- **Correção de gama de monitores** (γ ≈ 2.5 típico de CRT)
- **Realce médico** para imagens de ressonância magnética (Exemplo 3.1, p.128)
- **Ajuste de contraste** em imagens predominantemente escuras
- **Simulação de resposta não-linear** do sistema visual humano

📊 **VALORES TÍPICOS MENCIONADOS:**
- γ = 0.4: Expansão dramática de tons escuros (imagens médicas)
- γ = 0.6: Correção de monitores muito escuros
- γ = 1.0: Transformação identidade
- γ = 2.2: Correção padrão para monitores CRT
- γ = 2.5: Valor típico encontrado em monitores não calibrados
"""

import numpy as np
from ..utils.validacao import validar_parametro_numerico, garantir_uint8


def correcao_gama(imagem, gama=1.0, c=1.0):
    """
    Aplica correção gama (power-law transformation).

    📚 **Referência:** Gonzalez & Woods, Digital Image Processing 4e,
                      Cap.3, Eq.3-5, p.128 - "Power-law transformations"

    🧮 **Fórmula Matemática:** s = c × r^γ
       onde s=saída, r=entrada[0-1], c=constante, γ=exponente

    Args:
        imagem: Array NumPy da imagem
        gama: Exponente da transformação (validado: 0.1-5.0)
            - γ < 1: Expande tons escuros (curva côncava)
                * 0.4: Realce dramático de sombras (Exemplo 3.1, p.128)
                * 0.6: Correção de monitores escuros
            - γ = 1: Transformação identidade (linear)
            - γ > 1: Comprime tons claros (curva convexa)
                * 2.2: Correção padrão de monitores CRT
                * 2.5: Gama típico de monitores não calibrados
        c: Constante multiplicativa (padrão 1.0, validado: 0.1-5.0)

    Returns:
        np.ndarray: Imagem corrigida dtype uint8

    📈 **Aplicações Acadêmicas Citadas:**
        - **Correção de gama de dispositivos** (monitores CRT γ≈2.5)
        - **Realce médico** em imagens de ressonância magnética
        - **Expansão de tons escuros** em imagens subexpostas
        - **Simulação de resposta visual** não-linear humana

    ⚠️ **Implementação:**
       - Normalização [0-1] para evitar overflow
       - Clipping para prevenir valores inválidos
       - Proteção contra divisão por zero (ε=1e-7)
    """
    # Validações
    validar_parametro_numerico(gama, "gama", 0.1, 5.0)
    validar_parametro_numerico(c, "c", 0.1, 5.0)
    
    if not isinstance(imagem, np.ndarray):
        raise ValueError("imagem deve ser um array NumPy")
    
    # Normaliza para [0, 1] para evitar problemas com potências
    imagem_norm = imagem.astype(np.float64) / 255.0
    
    # Evita problemas com valor 0 elevado a potência negativa
    # e garante que não há valores fora do range [0,1]
    imagem_norm = np.clip(imagem_norm, 1e-7, 1.0)
    
    # Aplica a transformação gama: f = c × (g)^γ
    resultado_norm = c * np.power(imagem_norm, gama)
    
    # Desnormaliza para [0, 255]
    resultado = resultado_norm * 255.0
    
    # Garante range válido e converte para uint8
    resultado = np.clip(resultado, 0, 255)
    return garantir_uint8(resultado)


def gama_adaptativo_por_regioes(imagem, num_regioes=4, gama_base=1.0, fator_adaptacao=0.5):
    """
    Aplica correção gama adaptativa baseada na luminosidade de regiões.
    
    Args:
        imagem: Array da imagem (deve ser escala de cinza)
        num_regioes: Número de regiões para análise (2, 4, ou 8)
        gama_base: Valor base de gama 
        fator_adaptacao: Intensidade da adaptação [0-1]
        
    Returns:
        np.ndarray: Imagem com gama adaptativo aplicado
        
    Algoritmo:
        1. Divide imagem em regiões
        2. Calcula luminosidade média de cada região
        3. Ajusta gama baseado na luminosidade:
           - Regiões escuras → gama menor (clareia)
           - Regiões claras → gama maior (escurece)
        4. Aplica correção específica para cada região
        
    Aplicações:
        - Correção de iluminação não-uniforme
        - Realce de detalhes em imagens com alto contraste
        - Pré-processamento para OCR ou detecção de texto
        - Melhoria de imagens médicas
    """
    if len(imagem.shape) != 2:
        raise ValueError("gama_adaptativo_por_regioes requer imagem em escala de cinza")
        
    altura, largura = imagem.shape
    resultado = np.copy(imagem).astype(np.float64)
    
    # Calcula tamanho das regiões
    if num_regioes == 2:
        regioes = [(0, altura//2), (altura//2, altura)]
        divisoes_h = [0, largura]
        divisoes_v = [0, altura//2, altura]
    elif num_regioes == 4:
        meio_h, meio_v = altura//2, largura//2
        regioes = [(0, meio_h), (meio_h, altura)]
        divisoes_h = [0, meio_v, largura]  
        divisoes_v = [0, meio_h, altura]
    elif num_regioes == 8:
        terco_h, terco_v = altura//3, largura//3
        regioes = [(0, terco_h), (terco_h, 2*terco_h), (2*terco_h, altura)]
        divisoes_h = [0, terco_v, 2*terco_v, largura]
        divisoes_v = [0, terco_h, 2*terco_h, altura]
    else:
        raise ValueError("num_regioes deve ser 2, 4 ou 8")
    
    # Processa cada região
    for i in range(len(divisoes_v)-1):
        y1, y2 = divisoes_v[i], divisoes_v[i+1]
        for j in range(len(divisoes_h)-1):
            x1, x2 = divisoes_h[j], divisoes_h[j+1]
            
            # Extrai região
            regiao = imagem[y1:y2, x1:x2]
            
            # Calcula luminosidade média da região
            luminosidade_media = regiao.mean()
            
            # Adapta gama baseado na luminosidade
            # Regiões escuras (baixa luminosidade) → gama menor
            # Regiões claras (alta luminosidade) → gama maior
            fator_regiao = (luminosidade_media / 128.0 - 1.0) * fator_adaptacao
            gama_regiao = gama_base + fator_regiao
            gama_regiao = np.clip(gama_regiao, 0.3, 3.0)  # Limita range
            
            # Aplica gama específico na região
            regiao_corrigida = correcao_gama(regiao, gama=gama_regiao)
            resultado[y1:y2, x1:x2] = regiao_corrigida
    
    return garantir_uint8(resultado)


def curva_gama_invertida(imagem, gama_original):
    """
    Reverte correção gama aplicando gama inverso.
    
    Args:
        imagem: Imagem com gama aplicado
        gama_original: Valor de gama usado na correção original
        
    Returns:
        np.ndarray: Imagem com gama revertido
        
    Se imagem foi corrigida com γ=2.2, usar gama_original=2.2
    resultará em gama inverso = 1/2.2 ≈ 0.45
    
    Aplicações:
        - Reverter correções de gama incorretas
        - Preparar imagens para diferentes tipos de display
        - Análise forense de processamento de imagem
        - Conversão entre padrões de cor
    """
    gama_inverso = 1.0 / gama_original
    return correcao_gama(imagem, gama=gama_inverso)


def detectar_gama_imagem(imagem, gamas_teste=None):
    """
    Tenta detectar o gama aplicado em uma imagem através de análise estatística.
    
    Args:
        imagem: Imagem para análise (escala de cinza)
        gamas_teste: Lista de valores de gama para testar
        
    Returns:
        dict: Resultados da análise com gama mais provável
        
    Método:
        1. Testa diferentes valores de gama
        2. Para cada gama, calcula distribuição de intensidades  
        3. Seleciona gama que produz distribuição mais "natural"
        
    Critérios de "naturalidade":
        - Distribuição não muito concentrada nos extremos
        - Variance balanceada entre tons escuros/claros
        - Histograma com forma típica de imagens naturais
        
    Limitações:
        - Funciona melhor com imagens naturais diversificadas
        - Pode falhar em imagens artificiais ou muito processadas
        - Resultado é estimativa, não valor exato
    """
    if len(imagem.shape) != 2:
        raise ValueError("detectar_gama_imagem requer imagem em escala de cinza")
    
    if gamas_teste is None:
        gamas_teste = [0.5, 0.7, 1.0, 1.4, 1.8, 2.2, 2.8]
    
    resultados = {}
    
    for gama in gamas_teste:
        # Aplica gama inverso (tenta "desfazer" o gama da imagem)
        img_teste = correcao_gama(imagem, gama=1.0/gama)
        
        # Calcula métricas de "naturalidade"
        hist, _ = np.histogram(img_teste, bins=256, range=(0, 255))
        
        # Métricas:
        # 1. Concentração nos extremos (ruim se muito alta)
        concentracao_extremos = (hist[0:10].sum() + hist[246:256].sum()) / hist.sum()
        
        # 2. Uniformidade da distribuição (muito uniforme = não natural)
        uniformidade = 1.0 - (np.std(hist) / np.mean(hist))
        
        # 3. Assimetria da distribuição
        assimetria = abs(img_teste.mean() - 127.5) / 127.5
        
        # Score composto (menor = mais natural)
        score_naturalidade = concentracao_extremos + uniformidade * 0.3 + assimetria * 0.2
        
        resultados[gama] = {
            'score': score_naturalidade,
            'concentracao_extremos': concentracao_extremos,
            'uniformidade': uniformidade,
            'assimetria': assimetria,
            'media': img_teste.mean(),
            'desvio': img_teste.std()
        }
    
    # Encontra gama com menor score (mais natural)
    gama_otimo = min(resultados.keys(), key=lambda g: resultados[g]['score'])
    
    return {
        'gama_detectado': gama_otimo,
        'confianca': 1.0 - resultados[gama_otimo]['score'],
        'todos_resultados': resultados
    }


def gerar_curva_gama(gama, pontos=256):
    """
    Gera curva de transformação gama para visualização.
    
    Args:
        gama: Valor de gama
        pontos: Número de pontos na curva
        
    Returns:
        tuple: (entrada, saida) arrays para plotar curva
        
    Útil para:
        - Visualizar efeito da transformação antes de aplicar
        - Análise comparativa entre diferentes valores de gama
        - Documentação e explicação didática
        - Debugging de parâmetros
        
    Exemplo de uso:
        x, y = gerar_curva_gama(2.2, 256)
        plt.plot(x, y, label=f'γ={2.2}')
        plt.xlabel('Entrada')
        plt.ylabel('Saída') 
        plt.legend()
    """
    entrada = np.linspace(0, 255, pontos)
    entrada_norm = entrada / 255.0
    saida_norm = np.power(entrada_norm, gama)
    saida = saida_norm * 255.0
    
    return entrada, saida
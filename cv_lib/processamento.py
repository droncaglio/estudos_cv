# minha_cv_lib/processamento.py

import numpy as np

def rgb_para_cinza(imagem_rgb, tipo='luminancia'):
    """
    Converte uma imagem RGB para escala de cinza usando diferentes metodos.
    
    Argumentos:
    imagem_rgb -- um array NumPy com shape (altura, largura, 3)
    tipo -- o tipo de conversao:
        'luminancia' ou 'bt601': Formula padrao (0.299*R + 0.587*G + 0.114*B)
        'bt709': Formula HDTV (0.2126*R + 0.7152*G + 0.0722*B) 
        'media': Media aritmetica simples
        'desaturacao': (max + min) / 2
        'canal_r': Apenas canal vermelho
        'canal_g': Apenas canal verde  
        'canal_b': Apenas canal azul
    """
    altura, largura, _ = imagem_rgb.shape
    imagem_cinza = np.zeros((altura, largura), dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            pixel = imagem_rgb[y, x]
            r, g, b = pixel[0], pixel[1], pixel[2]
            
            if tipo in ['luminancia', 'bt601']:
                # ITU-R BT.601 (SDTV): Padrao classico baseado na sensibilidade do olho humano
                # Aplicacoes: Processamento geral, compatibilidade com sistemas antigos
                # O olho e mais sensivel ao verde (58.7%), depois vermelho (29.9%) e azul (11.4%)
                valor_cinza = 0.299 * r + 0.587 * g + 0.114 * b
                
            elif tipo == 'bt709':
                # ITU-R BT.709 (HDTV): Padrao moderno para TV digital e monitores
                # Aplicacoes: Processamento para displays modernos, conteudo HD/4K
                # Coefficientes ajustados para fosforos modernos (mais peso no verde: 71.52%)
                valor_cinza = 0.2126 * r + 0.7152 * g + 0.0722 * b
                
            elif tipo == 'media':
                # Media aritmetica simples: Trata todos os canais igualmente
                # Aplicacoes: Algoritmos simples, prototipagem rapida, quando nao ha preferencia de canal
                # Pode resultar em imagens "chapadas" pois ignora sensibilidade do olho humano
                valor_cinza = (r + g + b) / 3
                
            elif tipo == 'desaturacao':
                # Desaturacao: Media entre valor maximo e minimo dos canais RGB
                # Aplicacoes: Arte digital, quando se quer preservar contraste de cores saturadas
                # Mantem melhor os detalhes em areas muito coloridas comparado a luminancia
                valor_cinza = (max(r, g, b) + min(r, g, b)) / 2
                
            elif tipo == 'canal_r':
                # Canal vermelho isolado
                # Aplicacoes: Deteccao de sangue/vasos sanguineos, analise de vegetacao (contraste com clorofila),
                # fotografia infravermelha, deteccao de pele em imagens medicas
                valor_cinza = r
                
            elif tipo == 'canal_g':
                # Canal verde isolado  
                # Aplicacoes: Analise de vegetacao (clorofila), deteccao de plantas em agricultura,
                # melhor canal para deteccao de bordas (mais detalhado), visao noturna
                valor_cinza = g
                
            elif tipo == 'canal_b':
                # Canal azul isolado
                # Aplicacoes: Deteccao de agua, analise de ceu/atmosfera, deteccao de veias,
                # contraste em imagens medicas, analise de poluicao atmosferica
                valor_cinza = b
            else:
                raise ValueError(f"Tipo '{tipo}' nao reconhecido")
                
            imagem_cinza[y, x] = valor_cinza

    imagem_cinza = imagem_cinza.astype(np.uint8)

    return imagem_cinza


def ajustar_brilho_contraste(imagem, brilho=0, contraste=1.0):
    """
    Aplica transformacao linear de brilho e contraste: f(x,y) = contraste * g(x,y) + brilho
    
    Args:
        imagem: Array NumPy da imagem (escala de cinza ou RGB)
        brilho: Valor a ser adicionado a cada pixel (-255 a +255)
        contraste: Fator multiplicativo (0.0 = preto, 1.0 = original, >1.0 = mais contraste)
        
    Aplicacoes:
        - Correcao de imagens muito escuras ou claras
        - Melhoria de visualizacao em monitores
        - Pre-processamento para algoritmos de deteccao
        - Ajuste artistico de fotografias
    """
    # Converte para float64 para evitar overflow durante os calculos
    imagem_float = imagem.astype(np.float64)
    
    # Aplica a transformacao linear
    resultado = contraste * imagem_float + brilho
    
    # Garante que os valores ficam no intervalo [0, 255]
    resultado = np.clip(resultado, 0, 255)
    
    # Converte de volta para uint8
    return resultado.astype(np.uint8)


def correcao_gama(imagem, gama=1.0, c=1.0):
    """
    Aplica correcao gama (power-law): f(x,y) = c * (g(x,y) / 255)^gama * 255
    
    Args:
        imagem: Array NumPy da imagem
        gama: Exponente da transformacao:
            - gama < 1: Imagem mais clara (expande tons escuros)
            - gama = 1: Sem alteracao (transformacao linear)
            - gama > 1: Imagem mais escura (comprime tons claros)
        c: Constante multiplicativa (normalmente 1.0)
        
    Aplicacoes:
        - Correcao de gama de monitores CRT vs LCD
        - Ajuste de imagens capturadas em condicoes de pouca luz
        - Realce de detalhes em sombras (gama < 1) ou highlights (gama > 1)
        - Simulacao de resposta nao-linear do olho humano
    """
    # Normaliza para [0, 1]
    imagem_norm = imagem.astype(np.float64) / 255.0
    
    # Aplica a transformacao gama
    # Evita problemas com valor 0 elevado a potencia negativa
    imagem_norm = np.clip(imagem_norm, 1e-7, 1.0)
    resultado_norm = c * np.power(imagem_norm, gama)
    
    # Desnormaliza para [0, 255]
    resultado = resultado_norm * 255.0
    resultado = np.clip(resultado, 0, 255)
    
    return resultado.astype(np.uint8)


def normalizar_imagem(imagem, novo_min=0, novo_max=255):
    """
    Normaliza os valores da imagem para um intervalo especificado.
    
    Args:
        imagem: Array NumPy da imagem
        novo_min: Valor minimo do intervalo de destino
        novo_max: Valor maximo do intervalo de destino
        
    Aplicacoes:
        - Padronizacao de imagens com diferentes faixas dinamicas
        - Pre-processamento para algoritmos de ML
        - Melhoria de contraste quando a imagem nao usa toda a faixa [0-255]
        - Normalizacao para comparacao entre imagens
    """
    # Valores minimo e maximo da imagem original
    min_original = float(imagem.min())
    max_original = float(imagem.max())
    
    # Evita divisao por zero se a imagem for uniforme
    if max_original == min_original:
        return np.full_like(imagem, novo_min, dtype=np.uint8)
    
    # Aplica a normalizacao linear: (x - min_old) / (max_old - min_old) * (max_new - min_new) + min_new
    imagem_float = imagem.astype(np.float64)
    resultado = (imagem_float - min_original) / (max_original - min_original)
    resultado = resultado * (novo_max - novo_min) + novo_min
    
    # Garante que esta no intervalo correto e converte para uint8
    resultado = np.clip(resultado, 0, 255)
    return resultado.astype(np.uint8)


def operacao_entre_imagens(img1, img2, operacao='soma', peso1=1.0, peso2=1.0):
    """
    Realiza operacoes aritmeticas entre duas imagens.
    
    Args:
        img1, img2: Arrays NumPy das imagens (devem ter o mesmo shape)
        operacao: Tipo de operacao ('soma', 'subtracao', 'multiplicacao', 'divisao', 'media_ponderada')
        peso1, peso2: Pesos para as imagens (usado em soma e media ponderada)
        
    Aplicacoes:
        - soma: Combinacao de imagens, deteccao de movimento
        - subtracao: Deteccao de diferencas, remocao de fundo
        - multiplicacao: Aplicacao de mascaras, realce seletivo
        - divisao: Correcao de iluminacao nao-uniforme
        - media_ponderada: Fusao de imagens com pesos diferentes
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Imagens devem ter o mesmo shape. img1: {img1.shape}, img2: {img2.shape}")
    
    # Converte para float64 para evitar overflow
    img1_float = img1.astype(np.float64)
    img2_float = img2.astype(np.float64)
    
    if operacao == 'soma':
        # Soma ponderada: peso1*img1 + peso2*img2
        resultado = peso1 * img1_float + peso2 * img2_float
        
    elif operacao == 'subtracao':
        # Subtracao: img1 - img2
        resultado = img1_float - img2_float
        
    elif operacao == 'multiplicacao':
        # Multiplicacao elemento a elemento (normalizada)
        resultado = (img1_float * img2_float) / 255.0
        
    elif operacao == 'divisao':
        # Divisao com protecao contra divisao por zero
        img2_safe = np.where(img2_float == 0, 1, img2_float)  # Evita divisao por zero
        resultado = (img1_float / img2_safe) * 255.0
        
    elif operacao == 'media_ponderada':
        # Media ponderada normalizada
        soma_pesos = peso1 + peso2
        if soma_pesos == 0:
            return np.zeros_like(img1, dtype=np.uint8)
        resultado = (peso1 * img1_float + peso2 * img2_float) / soma_pesos
        
    else:
        raise ValueError(f"Operacao '{operacao}' nao reconhecida")
    
    # Garante valores no intervalo [0, 255] e converte para uint8
    resultado = np.clip(resultado, 0, 255)
    return resultado.astype(np.uint8)


def rgb_para_ycbcr(imagem_rgb):
    """
    Converte imagem RGB para espaco de cor YCbCr (luminancia + crominancia).
    
    YCbCr e usado em compressao de video (JPEG, MPEG) e TV digital.
    - Y: Luminancia (brilho), similar ao que vemos em escala de cinza
    - Cb: Crominancia azul-amarelo (Blue-Yellow chroma)
    - Cr: Crominancia vermelho-verde (Red-Green chroma)
    
    Formula ITU-R BT.601:
    Y  = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.169*R - 0.331*G + 0.500*B + 128
    Cr = 0.500*R - 0.419*G - 0.081*B + 128
    
    Aplicacoes:
        - Compressao de video/imagem (JPEG)
        - Separacao de luminancia e cor para processamento independente
        - TV digital e broadcasting
        - Deteccao de pele (usando componentes Cb/Cr)
    """
    altura, largura, _ = imagem_rgb.shape
    imagem_ycbcr = np.zeros((altura, largura, 3), dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            r, g, b = imagem_rgb[y, x, 0], imagem_rgb[y, x, 1], imagem_rgb[y, x, 2]
            
            # Calcula componentes YCbCr
            Y = 0.299 * r + 0.587 * g + 0.114 * b
            Cb = -0.169 * r - 0.331 * g + 0.500 * b + 128
            Cr = 0.500 * r - 0.419 * g - 0.081 * b + 128
            
            imagem_ycbcr[y, x] = [Y, Cb, Cr]

    # Garante valores no intervalo [0, 255]
    imagem_ycbcr = np.clip(imagem_ycbcr, 0, 255)
    return imagem_ycbcr.astype(np.uint8)


def ycbcr_para_rgb(imagem_ycbcr):
    """
    Converte imagem YCbCr de volta para RGB.
    
    Formula inversa ITU-R BT.601:
    R = Y + 1.402*(Cr-128)
    G = Y - 0.344*(Cb-128) - 0.714*(Cr-128)  
    B = Y + 1.772*(Cb-128)
    """
    altura, largura, _ = imagem_ycbcr.shape
    imagem_rgb = np.zeros((altura, largura, 3), dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            Y, Cb, Cr = imagem_ycbcr[y, x, 0], imagem_ycbcr[y, x, 1], imagem_ycbcr[y, x, 2]
            
            # Converte de volta para RGB
            R = Y + 1.402 * (Cr - 128)
            G = Y - 0.344 * (Cb - 128) - 0.714 * (Cr - 128)
            B = Y + 1.772 * (Cb - 128)
            
            imagem_rgb[y, x] = [R, G, B]

    # Garante valores no intervalo [0, 255]
    imagem_rgb = np.clip(imagem_rgb, 0, 255)
    return imagem_rgb.astype(np.uint8)


def rgb_para_hsv(imagem_rgb):
    """
    Converte imagem RGB para espaco de cor HSV (Hue, Saturation, Value).
    
    HSV e mais intuitivo para manipulacao de cores:
    - H: Hue (Matiz) - a cor em si (0-360 graus na roda de cores)
    - S: Saturation (Saturacao) - pureza/intensidade da cor (0-100%)
    - V: Value (Valor/Brilho) - quao clara ou escura e a cor (0-100%)
    
    Aplicacoes:
        - Selecao de objetos por cor (mais robusta que RGB)
        - Ajustes artisticos de cor
        - Segmentacao baseada em cor
        - Deteccao de objetos coloridos
        - Interfaces de selecao de cor
        
    ALGORITMO DETALHADO DO HUE (MATIZ):
    A roda de cores HSV e dividida em 6 setores de 60 graus cada:
    - Setor 0: 0°-60°   (Vermelho -> Amarelo)
    - Setor 1: 60°-120° (Amarelo -> Verde) 
    - Setor 2: 120°-180°(Verde -> Ciano)
    - Setor 3: 180°-240°(Ciano -> Azul)
    - Setor 4: 240°-300°(Azul -> Magenta)
    - Setor 5: 300°-360°(Magenta -> Vermelho)
    """
    altura, largura, _ = imagem_rgb.shape
    imagem_hsv = np.zeros((altura, largura, 3), dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            # Normaliza RGB para [0,1] para facilitar calculos
            r, g, b = imagem_rgb[y, x, 0] / 255.0, imagem_rgb[y, x, 1] / 255.0, imagem_rgb[y, x, 2] / 255.0
            
            # PASSO 1: Encontra valores maximo, minimo e diferenca (delta)
            max_val = max(r, g, b)  # Componente com maior intensidade
            min_val = min(r, g, b)  # Componente com menor intensidade
            delta = max_val - min_val  # Diferenca = "intensidade da cor"
            
            # PASSO 2: Calcula VALUE (brilho) - simplesmente o valor maximo
            v = max_val  # Value = quao "brilhante" e a cor (0-1)
            
            # PASSO 3: Calcula SATURATION (saturacao) - quao "pura" e a cor
            if max_val == 0:
                # Se max_val = 0, a cor e preta (sem saturacao)
                s = 0
            else:
                # Saturacao = diferenca / brilho
                # Se delta=0, cor e acinzentada (sem saturacao)  
                # Se delta=max_val, cor e "pura" (saturacao maxima)
                s = delta / max_val
            
            # PASSO 4: Calcula HUE (matiz) - a cor em si na roda de cores
            if delta == 0:
                # Se nao ha diferenca entre RGB, a cor e acinzentada (sem matiz definido)
                h = 0
                
            elif max_val == r:
                # VERMELHO e dominante - estamos nos setores 0° ou 5° da roda
                # Formula: posicao dentro do setor baseada na diferenca G-B
                posicao_no_setor = (g - b) / delta
                
                # Multiplica por 60° (tamanho do setor) e usa % 6 para lidar com valores negativos
                # % 6 garante que valores negativos "deem a volta" na roda (ex: -1 vira 5)
                h = 60 * (posicao_no_setor % 6)
                
            elif max_val == g:
                # VERDE e dominante - estamos no setor 1° (60°-120°)
                # +2 = pula 2 setores (120° de offset)
                posicao_no_setor = (b - r) / delta  # Posicao baseada em B-R
                h = 60 * (posicao_no_setor + 2)     # +2 setores = +120°
                
            elif max_val == b:
                # AZUL e dominante - estamos no setor 2° (240°-300°)  
                # +4 = pula 4 setores (240° de offset)
                posicao_no_setor = (r - g) / delta  # Posicao baseada em R-G  
                h = 60 * (posicao_no_setor + 4)     # +4 setores = +240°
            
            # PASSO 5: Converte para ranges convencionais de armazenamento
            # OpenCV usa H em [0-179] para caber em uint8 (180 valores)
            h = h / 2        # Divide por 2: [0-360°] -> [0-180]
            s = s * 255      # S: [0-1] -> [0-255]  
            v = v * 255      # V: [0-1] -> [0-255]
            
            imagem_hsv[y, x] = [h, s, v]

    return imagem_hsv.astype(np.uint8)


def hsv_para_rgb(imagem_hsv):
    """
    Converte imagem HSV de volta para RGB.
    """
    altura, largura, _ = imagem_hsv.shape
    imagem_rgb = np.zeros((altura, largura, 3), dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            h, s, v = imagem_hsv[y, x, 0] * 2, imagem_hsv[y, x, 1] / 255.0, imagem_hsv[y, x, 2] / 255.0
            
            # Algoritmo de conversao HSV -> RGB
            c = v * s  # Chroma
            x_val = c * (1 - abs((h / 60) % 2 - 1))
            m = v - c
            
            if 0 <= h < 60:
                r, g, b = c, x_val, 0
            elif 60 <= h < 120:
                r, g, b = x_val, c, 0
            elif 120 <= h < 180:
                r, g, b = 0, c, x_val
            elif 180 <= h < 240:
                r, g, b = 0, x_val, c
            elif 240 <= h < 300:
                r, g, b = x_val, 0, c
            elif 300 <= h < 360:
                r, g, b = c, 0, x_val
            else:
                r, g, b = 0, 0, 0
            
            # Adiciona m e converte para [0-255]
            imagem_rgb[y, x] = [(r + m) * 255, (g + m) * 255, (b + m) * 255]

    return np.clip(imagem_rgb, 0, 255).astype(np.uint8)


def rgb_para_lab(imagem_rgb):
    """
    Converte RGB para espaco de cor Lab (L*a*b*) via XYZ - PROCESSO EM 3 ETAPAS.
    
    Lab e perceptualmente uniforme - distancias numericas correspondem 
    a diferencas visuais percebidas pelo olho humano.
    
    - L: Lightness (Luminosidade) - 0 (preto absoluto) a 100 (branco absoluto)
    - a: Eixo verde-vermelho - negativo=verde, positivo=vermelho
    - b: Eixo azul-amarelo - negativo=azul, positivo=amarelo
    
    PROCESSO DE CONVERSAO (3 ETAPAS OBRIGATORIAS):
    RGB → Linear RGB → XYZ → Lab
    
    ETAPA 1: RGB (sRGB) → Linear RGB
    - Remove "gamma correction" aplicada pelos monitores
    - sRGB usa gamma ≈ 2.2 para otimizar percepcao visual
    - Precisamos reverter isso para calculos corretos
    
    ETAPA 2: Linear RGB → XYZ  
    - XYZ e um espaco "device-independent" (independe do dispositivo)
    - Usa matriz padrao sRGB D65 (luz do dia)
    - X,Y,Z representam estimulos dos cones do olho humano
    
    ETAPA 3: XYZ → Lab
    - Aplica funcoes nao-lineares baseadas na percepcao humana
    - Lab e "perceptualmente uniforme" (distancias = diferencas visuais)
    - Usa illuminant D65 como referencia (luz branca padrao)
    
    Aplicacoes:
        - Correcao de cor profissional
        - Comparacao perceptual de cores (Delta E)
        - Impressao e reproducao de cores
        - Segmentacao robusta por cor
    """
    altura, largura, _ = imagem_rgb.shape
    imagem_lab = np.zeros((altura, largura, 3), dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            # Normaliza RGB de [0-255] para [0-1]
            r, g, b = imagem_rgb[y, x, 0] / 255.0, imagem_rgb[y, x, 1] / 255.0, imagem_rgb[y, x, 2] / 255.0
            
            # =====================================================================
            # ETAPA 1: sRGB -> Linear RGB (Remove gamma correction)
            # =====================================================================
            # Monitores aplicam "gamma correction" (~2.2) para otimizar percepcao
            # Precisamos reverter isso para calculos colorimetricos corretos
            
            def gamma_correction_reversa(componente):
                """
                Reverte a gamma correction do sRGB para obter RGB linear.
                
                sRGB usa gamma aproximado de 2.2, mas com uma curva especial:
                - Valores baixos (<=0.04045): divisao linear simples
                - Valores altos (>0.04045): funcao exponencial
                
                Por que 0.04045? Ponto onde as duas curvas se encontram suavemente.
                Por que 12.92? Inclinacao da parte linear (1/2.4 ≈ 0.4167).
                Por que 2.4? Gamma real do sRGB (nao exatamente 2.2).
                """
                if componente <= 0.04045:
                    # Regiao linear (sombras muito escuras)
                    return componente / 12.92
                else:
                    # Regiao exponencial (maior parte dos valores)
                    return pow((componente + 0.055) / 1.055, 2.4)
            
            # Aplica a remocao de gamma em cada canal
            r_linear = gamma_correction_reversa(r)
            g_linear = gamma_correction_reversa(g) 
            b_linear = gamma_correction_reversa(b)
            
            # =====================================================================
            # ETAPA 2: Linear RGB -> XYZ (Transformacao matricial)
            # =====================================================================
            # XYZ e um espaco "device-independent" baseado na CIE (1931)
            # Representa como os cones L,M,S do olho humano respondem a luz
            
            # Matriz de transformacao sRGB D65 -> XYZ (padrao internacional)
            # Valores derivados das cromaticidades do sRGB e illuminant D65
            # Cada linha corresponde a um componente XYZ
            
            # X: estimulo dos cones L (long wavelength - vermelhos)  
            X = r_linear * 0.4124564 + g_linear * 0.3575761 + b_linear * 0.1804375
            
            # Y: estimulo dos cones M (medium wavelength - verdes) + luminancia
            Y = r_linear * 0.2126729 + g_linear * 0.7151522 + b_linear * 0.0721750  
            
            # Z: estimulo dos cones S (short wavelength - azuis)
            Z = r_linear * 0.0193339 + g_linear * 0.1191920 + b_linear * 0.9503041
            
            # =====================================================================
            # ETAPA 2.5: Normalizacao com White Point D65 
            # =====================================================================
            # D65 = illuminant padrao (luz do dia a 6500K)
            # Normalizamos XYZ dividindo pelos valores do branco de referencia
            # Isso garante que branco puro (255,255,255) vire L=100 no Lab
            
            X_normalizado = X / 0.95047  # X do white point D65
            Y_normalizado = Y / 1.00000  # Y do white point D65 (sempre 1.0)
            Z_normalizado = Z / 1.08883  # Z do white point D65
            
            # =====================================================================
            # ETAPA 3: XYZ -> Lab (Funcoes nao-lineares perceptuais)
            # =====================================================================
            # Lab usa funcoes cubicas/lineares para modelar percepcao humana
            # O olho nao responde linearmente a luz - precisa de curvas especiais
            
            def xyz_para_lab_componente(valor_xyz_normalizado):
                """
                Aplica a funcao nao-linear XYZ->Lab baseada na percepcao humana.
                
                Por que 0.008856? Ponto onde curva cubica encontra reta linear.
                Por que 1/3? Curva cubica modela bem a percepcao de luminancia.
                Por que 7.787? Inclinacao da parte linear.
                Por que 16/116? Offset para conectar suavemente as duas curvas.
                
                Essas constantes foram determinadas experimentalmente pela CIE
                para melhor modelar como o olho humano percebe diferencas.
                """
                if valor_xyz_normalizado > 0.008856:
                    # Regiao cubica (maior parte dos valores)
                    return pow(valor_xyz_normalizado, 1/3)
                else:
                    # Regiao linear (sombras muito escuras)
                    return (7.787 * valor_xyz_normalizado + 16/116)
            
            # Aplica transformacao nao-linear em cada componente XYZ
            fx = xyz_para_lab_componente(X_normalizado)
            fy = xyz_para_lab_componente(Y_normalizado) 
            fz = xyz_para_lab_componente(Z_normalizado)
            
            # Calcula componentes finais do Lab
            # L: Lightness (luminosidade perceptual)
            L = 116 * fy - 16  # Range: [0-100] teorico
            
            # a: Eixo verde-vermelho (oponente)
            a = 500 * (fx - fy)  # Range: aproximadamente [-128 a +127]
            
            # b: Eixo azul-amarelo (oponente)  
            b_lab = 200 * (fy - fz)  # Range: aproximadamente [-128 a +127]
            
            # Por que 116, 500, 200? Constantes ajustadas para:
            # - L cobrir [0-100] para branco-preto
            # - a,b cobrirem faixas simetricas para cores oponentes
            # - Manter uniformidade perceptual
            
            # =====================================================================
            # ETAPA 4: Conversao para ranges uint8 convencionais 
            # =====================================================================
            # Precisamos mapear os ranges teoricos para [0-255] para armazenamento
            
            L_final = L * 255 / 100     # [0-100] -> [0-255]
            a_final = a + 128           # [-128,+127] -> [0-255] (centraliza em 128)  
            b_final = b_lab + 128       # [-128,+127] -> [0-255] (centraliza em 128)
            
            imagem_lab[y, x] = [L_final, a_final, b_final]

    return np.clip(imagem_lab, 0, 255).astype(np.uint8)
# cv_lib/espacos_cor/lab.py
"""
🧠 Conversões RGB ↔ Lab (L*a*b* - Perceptualmente Uniforme)

📚 **REFERÊNCIAS ACADÊMICAS:**
- **Szeliski, Computer Vision 2e, Eq.2.106-2.107, p.117**
  L* component definition e finite-slope approximation (δ=6/29)
- **Gonzalez & Woods, Digital Image Processing 4e, p.421**
  "Lab color space is colorimetric, perceptually uniform, and device independent"
- **CIE (1976). Colorimetry - Official Recommendations**
  Definição padrão internacional do espaço L*a*b*
- **MacAdams, D.L. (1942). Visual sensitivities to color differences**
  Base experimental para uniformidade perceptual
- **Fairchild, M.D. (2013). Color Appearance Models, 3rd Ed**

🧮 **DEFINIÇÃO MATEMÁTICA (Eq.2.106-2.107 Szeliski):**
```
L* = 116f(Y/Yn) - 16
a* = 500[f(X/Xn) - f(Y/Yn)]
b* = 200[f(Y/Yn) - f(Z/Zn)]

onde f(t) = { t^(1/3)           se t > δ³
            { t/(3δ²) + 2δ/3    caso contrário

δ = 6/29 ≈ 0.206896 (constante CIE)
```

💡 **COMPONENTES PERCEPTUAIS:**
- **L***: Lightness [0-100] - escala logarítmica de luminosidade
- **a***: Eixo verde(-) ↔ vermelho(+) [-128,+127] aproximadamente
- **b***: Eixo azul(-) ↔ amarelo(+) [-128,+127] aproximadamente

🎯 **APLICAÇÕES ACADÊMICAS CITADAS:**
- **Perceptual uniformity**: "Color differences are perceived uniformly" (G&W p.421)
- **Delta E calculations**: Medição objetiva de diferenças de cor
- **Device independence**: "Can represent accurately colors of any display" (G&W p.421)
- **Image compression**: Desacoplamento eficiente de intensidade e cor

🔄 **PROCESSO OBRIGATÓRIO (4 ETAPAS):**
1. **sRGB → Linear RGB**: Remove gamma correction (~2.2)
2. **Linear RGB → XYZ**: Transformação matricial (sRGB D65)
3. **XYZ → Lab**: Funções não-lineares baseadas na percepção
4. **Normalização**: Conversão para ranges uint8 convencionais
"""

import numpy as np
from ..utils.validacao import validar_imagem_rgb, garantir_uint8


def rgb_para_lab(imagem_rgb):
    """
    Converte RGB para espaço de cor Lab (L*a*b*) via XYZ.
    
    Args:
        imagem_rgb: Array RGB shape (altura, largura, 3)
        
    Returns:
        np.ndarray: Imagem Lab shape (altura, largura, 3) dtype uint8
        
    PROCESSO EM 3 ETAPAS OBRIGATÓRIAS:
        1. RGB (sRGB) → Linear RGB (remove gamma correction)
        2. Linear RGB → XYZ (transformação matricial)
        3. XYZ → Lab (funções não-lineares perceptuais)
        
    Por que 3 etapas?
        - sRGB tem gamma correction (~2.2) que precisa ser removida
        - XYZ é "device-independent" (independe do dispositivo) 
        - Lab aplica funções baseadas na percepção humana real
    """
    altura, largura, canais = validar_imagem_rgb(imagem_rgb, "imagem_rgb")
    imagem_lab = np.zeros((altura, largura, 3), dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            # Normaliza RGB de [0-255] para [0-1]
            r = imagem_rgb[y, x, 0] / 255.0
            g = imagem_rgb[y, x, 1] / 255.0  
            b = imagem_rgb[y, x, 2] / 255.0
            
            # =====================================================================
            # ETAPA 1: sRGB → Linear RGB (Remove gamma correction)
            # =====================================================================
            # Monitores aplicam "gamma correction" (~2.2) para otimizar percepção
            # Precisamos reverter isso para cálculos colorimétricos corretos
            
            r_linear = _remover_gamma_srgb(r)
            g_linear = _remover_gamma_srgb(g)
            b_linear = _remover_gamma_srgb(b)
            
            # =====================================================================
            # ETAPA 2: Linear RGB → XYZ (Transformação matricial)
            # =====================================================================
            # XYZ é um espaço "device-independent" baseado na CIE (1931)
            # Representa como os cones L,M,S do olho humano respondem à luz
            
            # Matriz de transformação sRGB D65 → XYZ (padrão internacional)
            # Valores derivados das cromaticidades do sRGB e illuminant D65
            X = r_linear * 0.4124564 + g_linear * 0.3575761 + b_linear * 0.1804375
            Y = r_linear * 0.2126729 + g_linear * 0.7151522 + b_linear * 0.0721750  
            Z = r_linear * 0.0193339 + g_linear * 0.1191920 + b_linear * 0.9503041
            
            # =====================================================================
            # ETAPA 2.5: Normalização com White Point D65 
            # =====================================================================
            # D65 = illuminant padrão (luz do dia a 6500K)
            # Normalizamos XYZ dividindo pelos valores do branco de referência
            X_normalizado = X / 0.95047  # X do white point D65
            Y_normalizado = Y / 1.00000  # Y do white point D65 (sempre 1.0)
            Z_normalizado = Z / 1.08883  # Z do white point D65
            
            # =====================================================================
            # ETAPA 3: XYZ → Lab (Funções não-lineares perceptuais)
            # =====================================================================
            # Lab usa funções cúbicas/lineares para modelar percepção humana
            
            fx = _xyz_para_lab_componente(X_normalizado)
            fy = _xyz_para_lab_componente(Y_normalizado) 
            fz = _xyz_para_lab_componente(Z_normalizado)
            
            # Calcula componentes finais do Lab
            L = 116 * fy - 16        # Lightness [0-100] teórico
            a = 500 * (fx - fy)      # Verde-Vermelho [-128,+127] aprox
            b_lab = 200 * (fy - fz)  # Azul-Amarelo [-128,+127] aprox
            
            # =====================================================================
            # ETAPA 4: Conversão para ranges uint8 convencionais 
            # =====================================================================
            L_final = L * 255 / 100     # [0-100] → [0-255]
            a_final = a + 128           # [-128,+127] → [0-255] (centraliza)  
            b_final = b_lab + 128       # [-128,+127] → [0-255] (centraliza)
            
            imagem_lab[y, x] = [L_final, a_final, b_final]

    return garantir_uint8(imagem_lab)


def lab_para_rgb(imagem_lab):
    """
    Converte imagem Lab de volta para RGB via XYZ.
    
    Args:
        imagem_lab: Array Lab shape (altura, largura, 3)
        
    Returns:
        np.ndarray: Imagem RGB shape (altura, largura, 3) dtype uint8
        
    PROCESSO INVERSO EM 3 ETAPAS:
        1. Lab → XYZ (funções inversas)
        2. XYZ → Linear RGB (matriz inversa)
        3. Linear RGB → sRGB (aplicar gamma correction)
        
    Nota: Conversão não é perfeitamente reversível devido a:
        - Arredondamentos em ponto flutuante
        - Clipping para ranges válidos  
        - Quantização uint8
        - Aproximações nas funções de percepção
    """
    altura, largura, _ = imagem_lab.shape
    imagem_rgb = np.zeros((altura, largura, 3), dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            # Desnormaliza componentes Lab
            L = imagem_lab[y, x, 0] * 100 / 255    # [0-255] → [0-100]
            a = imagem_lab[y, x, 1] - 128          # [0-255] → [-128,+127]
            b_lab = imagem_lab[y, x, 2] - 128      # [0-255] → [-128,+127]
            
            # =====================================================================
            # ETAPA 1: Lab → XYZ (Funções inversas)
            # =====================================================================
            fy = (L + 16) / 116
            fx = a / 500 + fy  
            fz = fy - b_lab / 200
            
            # Aplica funções inversas de percepção
            X_normalizado = _lab_para_xyz_componente(fx)
            Y_normalizado = _lab_para_xyz_componente(fy)
            Z_normalizado = _lab_para_xyz_componente(fz)
            
            # Desnormaliza com white point D65
            X = X_normalizado * 0.95047
            Y = Y_normalizado * 1.00000  
            Z = Z_normalizado * 1.08883
            
            # =====================================================================
            # ETAPA 2: XYZ → Linear RGB (Matriz inversa)
            # =====================================================================
            # Matriz inversa da transformação XYZ → sRGB D65
            r_linear = X * 3.2404542 + Y * -1.5371385 + Z * -0.4985314
            g_linear = X * -0.9692660 + Y * 1.8760108 + Z * 0.0415560
            b_linear = X * 0.0556434 + Y * -0.2040259 + Z * 1.0572252
            
            # =====================================================================
            # ETAPA 3: Linear RGB → sRGB (Aplicar gamma correction)
            # =====================================================================
            r = _aplicar_gamma_srgb(r_linear)
            g = _aplicar_gamma_srgb(g_linear)  
            b = _aplicar_gamma_srgb(b_linear)
            
            # Converte para [0-255]
            imagem_rgb[y, x] = [r * 255, g * 255, b * 255]

    return garantir_uint8(imagem_rgb)


def calcular_delta_e(cor1_lab, cor2_lab):
    """
    Calcula diferença perceptual Delta E entre duas cores Lab.
    
    Args:
        cor1_lab: tuple (L, a, b) primeira cor
        cor2_lab: tuple (L, a, b) segunda cor
        
    Returns:
        float: Delta E (diferença perceptual)
        
    Interpretação Delta E:
        - 0-1: Diferença imperceptível
        - 1-2: Diferença perceptível por observador treinado
        - 2-10: Diferença perceptível
        - 11-49: Cores parecem mais similares que diferentes
        - 100: Cores são opostas
        
    Aplicações:
        - Controle de qualidade em impressão
        - Matching de cores entre dispositivos
        - Avaliação de algoritmos de correção de cor
    """
    L1, a1, b1 = cor1_lab
    L2, a2, b2 = cor2_lab
    
    # Delta E CIE 1976 (fórmula euclidiana simples)
    delta_L = L1 - L2
    delta_a = a1 - a2
    delta_b = b1 - b2
    
    delta_e = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)
    return delta_e


def extrair_luminosidade(imagem_rgb):
    """
    Extrai apenas o canal L (Lightness) de uma imagem RGB.
    
    Args:
        imagem_rgb: Array RGB shape (altura, largura, 3)
        
    Returns:
        np.ndarray: Canal L shape (altura, largura) dtype uint8
        
    Diferença de rgb_para_cinza():
        - Lab L é perceptualmente uniforme
        - Melhor representa luminosidade percebida
        - Mais robusto para análises perceptuais
    """
    imagem_lab = rgb_para_lab(imagem_rgb)
    return imagem_lab[:, :, 0]  # Retorna apenas canal L


# =============================================================================
# FUNÇÕES AUXILIARES INTERNAS
# =============================================================================

def _remover_gamma_srgb(componente):
    """
    Remove gamma correction do sRGB para obter RGB linear.
    
    sRGB usa curva especial (não simples x^2.2):
    - Valores baixos (≤0.04045): divisão linear simples
    - Valores altos (>0.04045): função exponencial
    """
    if componente <= 0.04045:
        return componente / 12.92  # Região linear
    else:
        return pow((componente + 0.055) / 1.055, 2.4)  # Região exponencial


def _aplicar_gamma_srgb(componente_linear):
    """
    Aplica gamma correction do sRGB (inversa da função acima).
    """
    componente_linear = max(0, componente_linear)  # Evita valores negativos
    
    if componente_linear <= 0.0031308:
        return 12.92 * componente_linear  # Região linear
    else:
        return 1.055 * pow(componente_linear, 1.0/2.4) - 0.055  # Região exponencial


def _xyz_para_lab_componente(valor_xyz_normalizado):
    """
    Aplica função não-linear XYZ→Lab baseada na percepção humana.

    Implementação conforme CIE padrão com δ = 6/29:
    - Região cúbica: t^(1/3) para t > δ³
    - Região linear: t/(3δ²) + 2δ/3 para t ≤ δ³

    δ = 6/29 ≈ 0.206896 (constante CIE)
    δ³ ≈ 0.008856 (threshold para mudança de função)
    """
    delta = 6.0 / 29.0  # δ = 6/29 (constante CIE oficial)
    delta_cubo = delta**3  # δ³ ≈ 0.008856

    if valor_xyz_normalizado > delta_cubo:
        return pow(valor_xyz_normalizado, 1.0/3.0)  # Região cúbica
    else:
        # Aproximação linear: t/(3δ²) + 2δ/3
        return valor_xyz_normalizado / (3 * delta**2) + 2 * delta / 3


def _lab_para_xyz_componente(f_componente):
    """
    Aplica função inversa Lab→XYZ (inversa da função acima).

    Implementação conforme CIE padrão com δ = 6/29:
    - Região cúbica: f³ para f³ > δ³
    - Região linear: 3δ²(f - 2δ/3) para f³ ≤ δ³
    """
    delta = 6.0 / 29.0  # δ = 6/29 (constante CIE oficial)
    delta_cubo = delta**3  # δ³ ≈ 0.008856

    f_cubo = f_componente**3

    if f_cubo > delta_cubo:
        return f_cubo  # Região cúbica
    else:
        # Função linear inversa: 3δ²(f - 2δ/3)
        return 3 * delta**2 * (f_componente - 2 * delta / 3)


def obter_white_point_d65():
    """
    Retorna valores XYZ do white point D65 (illuminant padrão).
    
    Returns:
        tuple: (X, Y, Z) do branco de referência D65
        
    D65 = luz do dia a 6500K (padrão internacional)
    """
    return (0.95047, 1.00000, 1.08883)


def obter_matriz_srgb_para_xyz():
    """
    Retorna matriz de transformação sRGB→XYZ.
    
    Returns:
        np.ndarray: Matriz 3x3 para transformação matricial
        
    Útil para implementações vetorizadas usando NumPy.
    """
    return np.array([
        [0.4124564, 0.3575761, 0.1804375],  # X
        [0.2126729, 0.7151522, 0.0721750],  # Y  
        [0.0193339, 0.1191920, 0.9503041]   # Z
    ])


def obter_matriz_xyz_para_srgb():
    """
    Retorna matriz de transformação XYZ→sRGB (inversa).
    
    Returns:
        np.ndarray: Matriz 3x3 para transformação matricial inversa
    """
    return np.array([
        [3.2404542, -1.5371385, -0.4985314],  # R
        [-0.9692660, 1.8760108, 0.0415560],   # G
        [0.0556434, -0.2040259, 1.0572252]    # B
    ])
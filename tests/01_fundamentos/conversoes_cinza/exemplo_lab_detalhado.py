#!/usr/bin/env python3
"""
Exemplo detalhado do algoritmo RGB -> Lab com calculos passo a passo.

Este script demonstra EXATAMENTE como funciona o processo de 3 etapas:
RGB -> Linear RGB -> XYZ -> Lab
"""

import sys
from pathlib import Path

# Adiciona a raiz do projeto ao path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt

def lab_passo_a_passo(r, g, b, nome_cor=""):
    """
    Demonstra o algoritmo RGB->Lab passo a passo com explicacoes detalhadas.
    """
    print(f"\n🎨 CONVERTENDO {nome_cor.upper()} RGB({r}, {g}, {b}) -> Lab")
    print("=" * 80)
    
    # Normaliza para [0,1]
    r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
    print(f"📐 RGB normalizado: R={r_norm:.3f}, G={g_norm:.3f}, B={b_norm:.3f}")
    
    # ========================================================================
    # ETAPA 1: sRGB -> Linear RGB (Remove gamma correction)
    # ========================================================================
    print(f"\n🔄 ETAPA 1: sRGB -> Linear RGB (Remove Gamma Correction)")
    print("-" * 60)
    
    def gamma_reversa(c, nome_canal=""):
        print(f"   {nome_canal}: {c:.3f}", end="")
        if c <= 0.04045:
            linear = c / 12.92
            print(f" <= 0.04045, então divide por 12.92 = {linear:.6f}")
        else:
            linear = pow((c + 0.055) / 1.055, 2.4)
            print(f" > 0.04045, então ((c+0.055)/1.055)^2.4 = {linear:.6f}")
        return linear
    
    r_linear = gamma_reversa(r_norm, "R")
    g_linear = gamma_reversa(g_norm, "G") 
    b_linear = gamma_reversa(b_norm, "B")
    
    print(f"   📊 Linear RGB: ({r_linear:.6f}, {g_linear:.6f}, {b_linear:.6f})")
    
    # ========================================================================
    # ETAPA 2: Linear RGB -> XYZ (Transformação matricial)
    # ========================================================================
    print(f"\n🌈 ETAPA 2: Linear RGB -> XYZ (Transformação Matricial)")
    print("-" * 60)
    print("   Matriz sRGB D65 -> XYZ (padrão internacional):")
    print("   |X|   |0.4124564  0.3575761  0.1804375| |R_linear|")
    print("   |Y| = |0.2126729  0.7151522  0.0721750| |G_linear|")
    print("   |Z|   |0.0193339  0.1191920  0.9503041| |B_linear|")
    
    # Calcula XYZ
    X = r_linear * 0.4124564 + g_linear * 0.3575761 + b_linear * 0.1804375
    Y = r_linear * 0.2126729 + g_linear * 0.7151522 + b_linear * 0.0721750  
    Z = r_linear * 0.0193339 + g_linear * 0.1191920 + b_linear * 0.9503041
    
    print(f"\n   Cálculos detalhados:")
    print(f"   X = {r_linear:.6f}×0.4124564 + {g_linear:.6f}×0.3575761 + {b_linear:.6f}×0.1804375")
    print(f"     = {r_linear*0.4124564:.6f} + {g_linear*0.3575761:.6f} + {b_linear*0.1804375:.6f} = {X:.6f}")
    
    print(f"   Y = {r_linear:.6f}×0.2126729 + {g_linear:.6f}×0.7151522 + {b_linear:.6f}×0.0721750")  
    print(f"     = {r_linear*0.2126729:.6f} + {g_linear*0.7151522:.6f} + {b_linear*0.0721750:.6f} = {Y:.6f}")
    
    print(f"   Z = {r_linear:.6f}×0.0193339 + {g_linear:.6f}×0.1191920 + {b_linear:.6f}×0.9503041")
    print(f"     = {r_linear*0.0193339:.6f} + {g_linear*0.1191920:.6f} + {b_linear*0.9503041:.6f} = {Z:.6f}")
    
    print(f"\n   📊 XYZ bruto: ({X:.6f}, {Y:.6f}, {Z:.6f})")
    
    # ========================================================================
    # ETAPA 2.5: Normalização com White Point D65
    # ========================================================================
    print(f"\n⚪ ETAPA 2.5: Normalização com White Point D65")
    print("-" * 60)
    print("   White Point D65 (luz do dia 6500K): (0.95047, 1.00000, 1.08883)")
    
    X_norm = X / 0.95047
    Y_norm = Y / 1.00000  
    Z_norm = Z / 1.08883
    
    print(f"   X_norm = {X:.6f} / 0.95047 = {X_norm:.6f}")
    print(f"   Y_norm = {Y:.6f} / 1.00000 = {Y_norm:.6f}")
    print(f"   Z_norm = {Z:.6f} / 1.08883 = {Z_norm:.6f}")
    
    print(f"   📊 XYZ normalizado: ({X_norm:.6f}, {Y_norm:.6f}, {Z_norm:.6f})")
    
    # ========================================================================
    # ETAPA 3: XYZ -> Lab (Funções não-lineares perceptuais)
    # ========================================================================
    print(f"\n🧠 ETAPA 3: XYZ -> Lab (Funções Perceptuais)")
    print("-" * 60)
    print("   Aplicando função cúbica/linear baseada na percepção humana:")
    print("   f(t) = t^(1/3) se t > 0.008856, senão (7.787*t + 16/116)")
    
    def xyz_para_lab_func(t, nome=""):
        print(f"   {nome}: {t:.6f}", end="")
        if t > 0.008856:
            resultado = pow(t, 1/3)
            print(f" > 0.008856, então t^(1/3) = {resultado:.6f}")
        else:
            resultado = (7.787 * t + 16/116)
            print(f" <= 0.008856, então 7.787*t + 16/116 = {resultado:.6f}")
        return resultado
    
    fx = xyz_para_lab_func(X_norm, "f(X)")
    fy = xyz_para_lab_func(Y_norm, "f(Y)")
    fz = xyz_para_lab_func(Z_norm, "f(Z)")
    
    print(f"\n   Calculando componentes Lab:")
    
    # L (Lightness)
    L = 116 * fy - 16
    print(f"   L = 116 × f(Y) - 16 = 116 × {fy:.6f} - 16 = {L:.2f}")
    
    # a (Green-Red)
    a = 500 * (fx - fy)
    print(f"   a = 500 × (f(X) - f(Y)) = 500 × ({fx:.6f} - {fy:.6f}) = {a:.2f}")
    
    # b (Blue-Yellow)
    b_lab = 200 * (fy - fz)
    print(f"   b = 200 × (f(Y) - f(Z)) = 200 × ({fy:.6f} - {fz:.6f}) = {b_lab:.2f}")
    
    print(f"\n   📊 Lab teórico: L={L:.1f}, a={a:.1f}, b={b_lab:.1f}")
    
    # ========================================================================
    # ETAPA 4: Conversão para uint8 (armazenamento)
    # ========================================================================
    print(f"\n📦 ETAPA 4: Conversão para uint8 [0-255]")
    print("-" * 60)
    
    L_final = L * 255 / 100
    a_final = a + 128
    b_final = b_lab + 128
    
    print(f"   L_final = {L:.1f} × 255/100 = {L_final:.0f}")
    print(f"   a_final = {a:.1f} + 128 = {a_final:.0f}")
    print(f"   b_final = {b_lab:.1f} + 128 = {b_final:.0f}")
    
    # Clamp para [0-255]
    L_final = max(0, min(255, L_final))
    a_final = max(0, min(255, a_final))
    b_final = max(0, min(255, b_final))
    
    print(f"\n✅ RESULTADO FINAL: Lab({L_final:.0f}, {a_final:.0f}, {b_final:.0f})")
    
    return int(L_final), int(a_final), int(b_final)

def main():
    """Demonstra o algoritmo com cores conhecidas."""
    print("🧮 ALGORITMO RGB -> Lab: EXPLICAÇÃO MATEMÁTICA COMPLETA")
    print("\n🎯 PROCESSO EM 4 ETAPAS:")
    print("1. RGB -> Linear RGB (remove gamma correction)")
    print("2. Linear RGB -> XYZ (matriz de transformação)")
    print("3. XYZ -> Lab (funções perceptuais)")
    print("4. Lab -> uint8 (normalização para armazenamento)")
    
    # Cores para demonstrar o algoritmo
    cores_teste = [
        (255, 255, 255, "Branco Puro"),        # L≈100, a≈0, b≈0
        (0, 0, 0, "Preto Puro"),               # L≈0, a≈0, b≈0
        (255, 0, 0, "Vermelho Puro"),          # a > 0 (vermelho)
        (0, 255, 0, "Verde Puro"),             # a < 0 (verde)
        (0, 0, 255, "Azul Puro"),              # b < 0 (azul)
        (255, 255, 0, "Amarelo"),              # b > 0 (amarelo)
        (128, 64, 192, "Cor Intermediária"),   # Exemplo complexo
    ]
    
    resultados = []
    for r, g, b, nome in cores_teste:
        L, a, b_final = lab_passo_a_passo(r, g, b, nome)
        resultados.append((nome, r, g, b, L, a, b_final))
    
    # Tabela resumo
    print("\n" + "="*90)
    print("📋 TABELA RESUMO - RESULTADOS DAS CONVERSÕES RGB -> Lab")
    print("="*90)
    print(f"{'Cor':18} | {'RGB':12} | {'Lab':12} | Interpretação")
    print("-"*90)
    
    for nome, r, g, b, L, a, b_final in resultados:
        # Interpretação dos valores
        brilho = "escuro" if L < 85 else "médio" if L < 170 else "claro"
        cor_a = "verde" if a < 118 else "neutro" if a < 138 else "vermelho"
        cor_b = "azul" if b_final < 118 else "neutro" if b_final < 138 else "amarelo"
        
        interpretacao = f"{brilho}, {cor_a}, {cor_b}"
        
        print(f"{nome:18} | ({r:3},{g:3},{b:3}) | ({L:3},{a:3},{b_final:3}) | {interpretacao}")
    
    print("\n💡 CONCEITOS-CHAVE PARA ENTENDER:")
    print("• Gamma correction: monitores não são lineares, precisamos reverter")
    print("• XYZ: espaço 'device-independent' baseado na visão humana")
    print("• White Point D65: referência de luz branca (6500K)")
    print("• Funções cúbicas: modelam como percebemos diferenças de brilho")
    print("• L=0-100: preto a branco perceptual")
    print("• a: negativo=verde, positivo=vermelho")
    print("• b: negativo=azul, positivo=amarelo")
    print("• Lab é 'perceptualmente uniforme': distâncias = diferenças visuais")
    
    print("\n🔬 CONSTANTES IMPORTANTES:")
    print("• 0.04045, 12.92, 2.4: parâmetros da curva sRGB")
    print("• 0.008856, 7.787, 16/116: transição suave cúbica-linear")
    print("• 116, 500, 200: escalam Lab para ranges úteis")
    print("• (0.95047, 1.0, 1.08883): white point D65")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Exemplo detalhado do algoritmo RGB -> HSV com calculos passo a passo.

Este script demonstra EXATAMENTE como funciona o calculo do Hue (matiz)
com exemplos numericos concretos.
"""

import sys
from pathlib import Path

# Adiciona a raiz do projeto ao path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt

def hsv_passo_a_passo(r, g, b, nome_cor=""):
    """
    Demonstra o algoritmo RGB->HSV passo a passo com explicacoes.
    """
    print(f"\n🎨 CONVERTENDO {nome_cor.upper()} RGB({r}, {g}, {b}) -> HSV")
    print("=" * 60)
    
    # Normaliza para [0,1]
    r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
    print(f"📐 RGB normalizado: R={r_norm:.3f}, G={g_norm:.3f}, B={b_norm:.3f}")
    
    # Passo 1: Max, Min, Delta
    max_val = max(r_norm, g_norm, b_norm)
    min_val = min(r_norm, g_norm, b_norm)
    delta = max_val - min_val
    
    print(f"\n🔍 PASSO 1 - Analise dos valores:")
    print(f"   Max = {max_val:.3f}")
    print(f"   Min = {min_val:.3f}")  
    print(f"   Delta = {delta:.3f}")
    
    # Passo 2: VALUE (simples)
    v = max_val
    print(f"\n💡 PASSO 2 - VALUE (brilho):")
    print(f"   V = max_val = {v:.3f}")
    
    # Passo 3: SATURATION
    if max_val == 0:
        s = 0
        print(f"\n🎭 PASSO 3 - SATURATION:")
        print(f"   S = 0 (cor preta, sem saturacao)")
    else:
        s = delta / max_val
        print(f"\n🎭 PASSO 3 - SATURATION:")
        print(f"   S = delta/max = {delta:.3f}/{max_val:.3f} = {s:.3f}")
        print(f"   Interpretacao: {s*100:.1f}% de pureza da cor")
    
    # Passo 4: HUE (o mais complexo!)
    print(f"\n🌈 PASSO 4 - HUE (matiz) - O CALCULO COMPLEXO:")
    
    if delta == 0:
        h = 0
        print(f"   H = 0 (cor acinzentada, sem matiz definido)")
    else:
        # Determina qual cor e dominante
        if max_val == r_norm:
            componente_dominante = "VERMELHO"
            setor_base = "0° (setor 0 ou 5)"
            posicao = (g_norm - b_norm) / delta
            h_calculado = 60 * (posicao % 6)
            
            print(f"   🔴 {componente_dominante} e dominante!")
            print(f"   📍 Estamos no setor: {setor_base}")
            print(f"   📊 Posicao no setor = (G-B)/delta = ({g_norm:.3f}-{b_norm:.3f})/{delta:.3f} = {posicao:.3f}")
            print(f"   🔄 Aplicando % 6: {posicao:.3f} % 6 = {posicao % 6:.3f}")
            print(f"   🎯 H = 60° × {posicao % 6:.3f} = {h_calculado:.1f}°")
            
        elif max_val == g_norm:
            componente_dominante = "VERDE"  
            setor_base = "120° (setor 1)"
            posicao = (b_norm - r_norm) / delta
            offset = 2  # +2 setores = +120°
            h_calculado = 60 * (posicao + offset)
            
            print(f"   🟢 {componente_dominante} e dominante!")
            print(f"   📍 Estamos no setor: {setor_base}")
            print(f"   📊 Posicao no setor = (B-R)/delta = ({b_norm:.3f}-{r_norm:.3f})/{delta:.3f} = {posicao:.3f}")
            print(f"   ➕ Offset do setor = +{offset} (+{offset*60}°)")
            print(f"   🎯 H = 60° × ({posicao:.3f} + {offset}) = {h_calculado:.1f}°")
            
        elif max_val == b_norm:
            componente_dominante = "AZUL"
            setor_base = "240° (setor 2)"  
            posicao = (r_norm - g_norm) / delta
            offset = 4  # +4 setores = +240°
            h_calculado = 60 * (posicao + offset)
            
            print(f"   🔵 {componente_dominante} e dominante!")
            print(f"   📍 Estamos no setor: {setor_base}")
            print(f"   📊 Posicao no setor = (R-G)/delta = ({r_norm:.3f}-{g_norm:.3f})/{delta:.3f} = {posicao:.3f}")
            print(f"   ➕ Offset do setor = +{offset} (+{offset*60}°)")
            print(f"   🎯 H = 60° × ({posicao:.3f} + {offset}) = {h_calculado:.1f}°")
        
        h = h_calculado
    
    # Passo 5: Conversao para uint8
    h_final = h / 2      # [0-360°] -> [0-180] para OpenCV
    s_final = s * 255    # [0-1] -> [0-255]
    v_final = v * 255    # [0-1] -> [0-255]
    
    print(f"\n📦 PASSO 5 - Conversao para armazenamento uint8:")
    print(f"   H_final = {h:.1f}° ÷ 2 = {h_final:.1f} (OpenCV range)")
    print(f"   S_final = {s:.3f} × 255 = {s_final:.0f}")
    print(f"   V_final = {v:.3f} × 255 = {v_final:.0f}")
    
    print(f"\n✅ RESULTADO FINAL: HSV({h_final:.0f}, {s_final:.0f}, {v_final:.0f})")
    
    return int(h_final), int(s_final), int(v_final)


def main():
    """Demonstra o algoritmo com cores conhecidas."""
    print("🧮 ALGORITMO RGB -> HSV: EXPLICAÇÃO MATEMÁTICA COMPLETA")
    print("\nVamos converter algumas cores conhecidas e entender cada passo:")
    
    # Cores puras para demonstrar
    cores_teste = [
        (255, 0, 0, "Vermelho Puro"),      # Deve dar H≈0°
        (0, 255, 0, "Verde Puro"),         # Deve dar H≈120°  
        (0, 0, 255, "Azul Puro"),          # Deve dar H≈240°
        (255, 255, 0, "Amarelo"),          # Deve dar H≈60°
        (255, 0, 255, "Magenta"),          # Deve dar H≈300°
        (128, 128, 128, "Cinza Medio"),    # Deve dar H=0, S=0
        (255, 128, 0, "Laranja"),          # Exemplo intermediario
    ]
    
    resultados = []
    for r, g, b, nome in cores_teste:
        h, s, v = hsv_passo_a_passo(r, g, b, nome)
        resultados.append((nome, r, g, b, h, s, v))
    
    # Tabela resumo
    print("\n" + "="*80)
    print("📋 TABELA RESUMO - RESULTADOS DAS CONVERSÕES")
    print("="*80)
    print(f"{'Cor':15} | {'RGB':12} | {'HSV':12} | {'Hue°':>6} | Observação")
    print("-"*80)
    
    for nome, r, g, b, h, s, v in resultados:
        hue_real = h * 2  # Converte de volta para graus reais
        obs = ""
        if s < 50:
            obs = "Pouco saturada"
        elif hue_real < 30 or hue_real > 330:
            obs = "Região vermelha"
        elif 90 < hue_real < 150:
            obs = "Região verde"  
        elif 210 < hue_real < 270:
            obs = "Região azul"
            
        print(f"{nome:15} | ({r:3},{g:3},{b:3}) | ({h:3},{s:3},{v:3}) | {hue_real:5.0f}° | {obs}")
    
    print("\n💡 CONCEITOS-CHAVE PARA LEMBRAR:")
    print("• A roda HSV tem 6 setores de 60° cada")
    print("• O +2 e +4 são offsets para posicionar nos setores corretos") 
    print("• % 6 trata valores negativos (dá a volta na roda)")
    print("• Delta = 0 significa cor acinzentada (sem matiz)")
    print("• OpenCV usa H em [0-179] para caber em uint8")


if __name__ == "__main__":
    main()
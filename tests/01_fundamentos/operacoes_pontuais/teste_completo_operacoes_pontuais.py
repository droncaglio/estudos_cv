#!/usr/bin/env python3
"""
Teste completo das operacoes pontuais implementadas.

Este script demonstra todas as transformacoes pontuais: brilho, contraste, 
gama, normalizacao e operacoes entre imagens.
"""

import sys
from pathlib import Path

# Adiciona a raiz do projeto ao path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from cv_lib import processamento
from utils import visualization, datasets

def teste_brilho_contraste():
    """Testa ajustes de brilho e contraste."""
    print("🔆 Testando Brilho e Contraste")
    print("-" * 40)
    
    # Carrega imagem em escala de cinza
    imagem_rgb = datasets.carregar_imagem_teste('camera')
    imagem_cinza = processamento.rgb_para_cinza(imagem_rgb) if len(imagem_rgb.shape) == 3 else imagem_rgb
    
    # Diferentes combinacoes de brilho e contraste
    transformacoes = [
        (0, 1.0, "Original"),
        (50, 1.0, "Mais brilho (+50)"),
        (-50, 1.0, "Menos brilho (-50)"),
        (0, 1.5, "Mais contraste (1.5x)"),
        (0, 0.5, "Menos contraste (0.5x)"),
        (30, 1.3, "Brilho +30, Contraste 1.3x")
    ]
    
    imagens = []
    titulos = []
    
    for brilho, contraste, titulo in transformacoes:
        if brilho == 0 and contraste == 1.0:
            resultado = imagem_cinza
        else:
            resultado = processamento.ajustar_brilho_contraste(imagem_cinza, brilho, contraste)
        
        imagens.append(resultado)
        titulos.append(titulo)
        print(f"  {titulo}: min={resultado.min()}, max={resultado.max()}, media={resultado.mean():.1f}")
    
    # Visualiza resultados
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (img, titulo) in enumerate(zip(imagens, titulos)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(titulo, fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle("Transformacoes de Brilho e Contraste", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    

def teste_correcao_gama():
    """Testa correcao gama."""
    print("\n⚡ Testando Correcao Gama")
    print("-" * 40)
    
    # Carrega imagem
    imagem_rgb = datasets.carregar_imagem_teste('coins')
    imagem_cinza = processamento.rgb_para_cinza(imagem_rgb) if len(imagem_rgb.shape) == 3 else imagem_rgb
    
    # Diferentes valores de gama
    valores_gama = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    
    imagens = []
    titulos = []
    
    for gama in valores_gama:
        if gama == 1.0:
            resultado = imagem_cinza
            titulo = "Original (γ=1.0)"
        else:
            resultado = processamento.correcao_gama(imagem_cinza, gama=gama)
            titulo = f"γ={gama} ({'claro' if gama < 1 else 'escuro'})"
        
        imagens.append(resultado)
        titulos.append(titulo)
        print(f"  γ={gama}: min={resultado.min()}, max={resultado.max()}, media={resultado.mean():.1f}")
    
    # Visualiza resultados
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (img, titulo) in enumerate(zip(imagens, titulos)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(titulo, fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle("Correcao Gama - Diferentes Valores", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def teste_normalizacao():
    """Testa normalizacao de imagens."""
    print("\n📊 Testando Normalizacao")
    print("-" * 40)
    
    # Cria uma imagem com faixa dinamica limitada (simula imagem escura)
    imagem_original = datasets.carregar_imagem_teste('camera')
    if len(imagem_original.shape) == 3:
        imagem_original = processamento.rgb_para_cinza(imagem_original)
    
    # Simula imagem com baixo contraste (valores entre 80-120)
    imagem_baixo_contraste = np.clip(imagem_original * 0.2 + 80, 0, 255).astype(np.uint8)
    
    # Aplica normalizacao
    imagem_normalizada = processamento.normalizar_imagem(imagem_baixo_contraste)
    
    imagens = [imagem_original, imagem_baixo_contraste, imagem_normalizada]
    titulos = ["Original", "Baixo Contraste", "Normalizada"]
    
    for img, titulo in zip(imagens, titulos):
        print(f"  {titulo}: min={img.min()}, max={img.max()}, media={img.mean():.1f}")
    
    # Visualiza com histogramas
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for i, (img, titulo) in enumerate(zip(imagens, titulos)):
        # Imagem
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(titulo, fontsize=12)
        axes[0, i].axis('off')
        
        # Histograma
        axes[1, i].hist(img.flatten(), bins=50, alpha=0.7, color='blue')
        axes[1, i].set_title(f"Histograma - {titulo}", fontsize=10)
        axes[1, i].set_xlabel("Intensidade")
        axes[1, i].set_ylabel("Frequencia")
        axes[1, i].grid(True, alpha=0.3)
    
    plt.suptitle("Normalizacao e seus Efeitos no Histograma", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def teste_operacoes_entre_imagens():
    """Testa operacoes aritmeticas entre imagens."""
    print("\n🔢 Testando Operacoes Entre Imagens")
    print("-" * 40)
    
    # Carrega duas imagens diferentes
    img1 = datasets.carregar_imagem_teste('camera')
    img2 = datasets.carregar_imagem_teste('coins')
    
    # Converte para escala de cinza e redimensiona para mesmo tamanho
    if len(img1.shape) == 3:
        img1 = processamento.rgb_para_cinza(img1)
    if len(img2.shape) == 3:
        img2 = processamento.rgb_para_cinza(img2)
    
    # Redimensiona img2 para o tamanho de img1 (metodo simples)
    if img1.shape != img2.shape:
        # Usa slicing para fazer um crop simples
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]
    
    # Diferentes operacoes
    operacoes = [
        ('soma', 0.5, 0.5, "Soma (0.5 + 0.5)"),
        ('subtracao', 1.0, 1.0, "Subtracao (img1 - img2)"),
        ('multiplicacao', 1.0, 1.0, "Multiplicacao"),
        ('media_ponderada', 0.7, 0.3, "Media Ponderada (0.7 + 0.3)")
    ]
    
    # Imagens originais + resultados
    imagens = [img1, img2]
    titulos = ["Imagem 1 (Camera)", "Imagem 2 (Coins)"]
    
    for op, peso1, peso2, titulo in operacoes:
        try:
            resultado = processamento.operacao_entre_imagens(img1, img2, op, peso1, peso2)
            imagens.append(resultado)
            titulos.append(titulo)
            print(f"  {titulo}: min={resultado.min()}, max={resultado.max()}")
        except Exception as e:
            print(f"  Erro em {titulo}: {e}")
    
    # Visualiza resultados
    n_imagens = len(imagens)
    cols = 3
    rows = (n_imagens + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    if rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()
    
    for i, (img, titulo) in enumerate(zip(imagens, titulos)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(titulo, fontsize=10)
        axes[i].axis('off')
    
    # Remove axes extras
    for i in range(n_imagens, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Operacoes Aritmeticas Entre Imagens", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    """Funcao principal do teste."""
    print("🧪 Teste Completo: Operacoes Pontuais")
    print("=" * 50)
    
    # Executa todos os testes
    teste_brilho_contraste()
    teste_correcao_gama() 
    teste_normalizacao()
    teste_operacoes_entre_imagens()
    
    print("\n🎉 Todos os testes de operacoes pontuais concluidos!")
    print("\n💡 Conceitos importantes:")
    print("• Brilho/Contraste: Transformacao linear f(x) = a*x + b")
    print("• Gama: Transformacao nao-linear f(x) = c*x^γ")
    print("• Normalizacao: Expande faixa dinamica para usar [0-255] completo")
    print("• Operacoes entre imagens: Combinacao e comparacao de conteudo")


if __name__ == "__main__":
    main()
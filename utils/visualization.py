# utils/visualization.py

"""
Funcoes utilitarias para visualizacao de imagens e resultados.
Centralizacao de codigo comum de matplotlib para manter consistencia visual.
"""

import matplotlib.pyplot as plt
import numpy as np

def mostrar_comparacao(imagens, titulos, figsize=(15, 5), cmap_list=None):
    """
    Mostra multiplas imagens lado a lado para comparacao.
    
    Args:
        imagens: Lista de arrays numpy (imagens)
        titulos: Lista de strings (titulos)
        figsize: Tupla com tamanho da figura
        cmap_list: Lista de colormaps (None para RGB, 'gray' para cinza)
    """
    n_imagens = len(imagens)
    fig, axes = plt.subplots(1, n_imagens, figsize=figsize)
    
    if n_imagens == 1:
        axes = [axes]
    
    for i, (img, titulo) in enumerate(zip(imagens, titulos)):
        cmap = 'gray' if cmap_list is None else cmap_list[i]
        if len(img.shape) == 2:  # Imagem em escala de cinza
            cmap = 'gray'
            
        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(titulo, fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def mostrar_grid_conversoes(imagem_rgb, funcao_conversao, tipos_conversao, 
                           figsize=(16, 10)):
    """
    Mostra grid comparativo de diferentes tipos de conversao aplicados a uma imagem.
    
    Args:
        imagem_rgb: Imagem original RGB
        funcao_conversao: Funcao que aplica a conversao (ex: processamento.rgb_para_cinza)
        tipos_conversao: Lista de tipos de conversao a testar
        figsize: Tamanho da figura
    """
    n_tipos = len(tipos_conversao)
    cols = 4
    rows = (n_tipos + cols) // cols  # Calcula numero de linhas necessarias
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    # Mostra original
    axes[0].imshow(imagem_rgb)
    axes[0].set_title("Original RGB", fontweight='bold')
    axes[0].axis('off')
    
    # Aplica cada conversao
    for i, tipo in enumerate(tipos_conversao, 1):
        try:
            resultado = funcao_conversao(imagem_rgb, tipo=tipo)
            axes[i].imshow(resultado, cmap='gray')
            axes[i].set_title(f"{tipo}", fontsize=10)
            axes[i].axis('off')
            
            # Adiciona estatisticas basicas
            stats = f"min:{resultado.min()} max:{resultado.max()}"
            axes[i].text(0.02, 0.98, stats, transform=axes[i].transAxes, 
                        fontsize=8, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Erro:\n{str(e)}", 
                        transform=axes[i].transAxes, ha='center', va='center',
                        fontsize=10, color='red')
            axes[i].set_title(f"{tipo} (ERRO)", color='red')
            axes[i].axis('off')
    
    # Esconde axes extras se houver
    for i in range(n_tipos + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def salvar_resultado(imagem, caminho, dpi=150):
    """
    Salva uma imagem nos resultados com configuracoes padronizadas.
    
    Args:
        imagem: Array numpy da imagem
        caminho: Caminho onde salvar (incluir extensao)
        dpi: Resolucao da imagem salva
    """
    plt.figure(figsize=(10, 8))
    
    if len(imagem.shape) == 2:  # Escala de cinza
        plt.imshow(imagem, cmap='gray')
    else:  # RGB
        plt.imshow(imagem)
    
    plt.axis('off')
    plt.savefig(caminho, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Resultado salvo em: {caminho}")
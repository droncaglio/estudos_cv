# utils/datasets.py

"""
Funcoes para carregamento e gerenciamento de imagens de teste.
Centralizacao do acesso a datasets padrao e imagens customizadas.
"""

import os
import numpy as np
from skimage import data, io
from pathlib import Path

# Caminho base do projeto
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_PATH = PROJECT_ROOT / "assets" / "test_images"

def carregar_imagens_padrao():
    """
    Carrega conjunto de imagens padrao do scikit-image para testes.
    
    Returns:
        dict: Dicionario com nome -> imagem
    """
    imagens = {
        'chelsea': data.chelsea(),      # Gato colorido
        'camera': data.camera(),        # Cameraman (P&B clasica)
        'coins': data.coins(),          # Moedas (boa para segmentacao)
        'astronaut': data.astronaut(),  # Astronauta colorido
        'coffee': data.coffee(),        # Xicard de cafe
        'rocket': data.rocket(),        # Foguete (alta resolucao)
        'checkerboard': data.checkerboard()  # Tabuleiro (padroes geometricos)
    }
    return imagens

def carregar_imagem_teste(nome='chelsea'):
    """
    Carrega uma imagem de teste especifica.
    
    Args:
        nome: Nome da imagem ('chelsea', 'camera', 'coins', etc.)
        
    Returns:
        numpy.ndarray: Imagem carregada
    """
    imagens = carregar_imagens_padrao()
    
    if nome not in imagens:
        print(f"⚠️  Imagem '{nome}' nao encontrada. Opcoes disponiveis:")
        for key in imagens.keys():
            print(f"   - {key}")
        return imagens['chelsea']  # Padrao
    
    return imagens[nome]

def listar_imagens_customizadas():
    """
    Lista imagens customizadas na pasta assets/test_images/
    
    Returns:
        list: Lista de caminhos para imagens encontradas
    """
    if not ASSETS_PATH.exists():
        return []
    
    extensoes = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    imagens = []
    
    for ext in extensoes:
        imagens.extend(ASSETS_PATH.glob(f"*{ext}"))
        imagens.extend(ASSETS_PATH.glob(f"*{ext.upper()}"))
    
    return sorted(imagens)

def carregar_imagem_customizada(nome_arquivo):
    """
    Carrega imagem customizada da pasta assets/test_images/
    
    Args:
        nome_arquivo: Nome do arquivo (com ou sem caminho completo)
        
    Returns:
        numpy.ndarray: Imagem carregada ou None se nao encontrada
    """
    if isinstance(nome_arquivo, str):
        # Se ja eh um caminho completo
        if os.path.isabs(nome_arquivo):
            caminho = Path(nome_arquivo)
        else:
            # Se eh apenas o nome do arquivo
            caminho = ASSETS_PATH / nome_arquivo
    else:
        caminho = Path(nome_arquivo)
    
    if not caminho.exists():
        print(f"❌ Arquivo nao encontrado: {caminho}")
        print("📁 Imagens disponiveis:")
        for img in listar_imagens_customizadas():
            print(f"   - {img.name}")
        return None
    
    try:
        imagem = io.imread(caminho)
        print(f"✅ Carregada: {caminho.name} - Shape: {imagem.shape} - Tipo: {imagem.dtype}")
        return imagem
    except Exception as e:
        print(f"❌ Erro ao carregar {caminho}: {e}")
        return None

def info_imagem(imagem, nome=""):
    """
    Mostra informacoes detalhadas sobre uma imagem.
    
    Args:
        imagem: Array numpy da imagem
        nome: Nome opcional para identificacao
    """
    if nome:
        print(f"📊 Informacoes da imagem '{nome}':")
    else:
        print("📊 Informacoes da imagem:")
    
    print(f"   Shape: {imagem.shape}")
    print(f"   Tipo: {imagem.dtype}")
    print(f"   Min: {imagem.min()}, Max: {imagem.max()}")
    print(f"   Media: {imagem.mean():.2f}")
    
    if len(imagem.shape) == 3:
        print(f"   Canais: {imagem.shape[2]} (RGB)" if imagem.shape[2] == 3 else f"   Canais: {imagem.shape[2]}")
    else:
        print("   Formato: Escala de cinza")
    
    print(f"   Tamanho: {imagem.size} pixels ({imagem.nbytes} bytes)")
    print()

def criar_imagem_teste(tipo='gradiente', tamanho=(200, 200)):
    """
    Cria imagens sinteticas para teste de algoritmos.
    
    Args:
        tipo: Tipo de imagem ('gradiente', 'xadrez', 'circulo', 'ruido')
        tamanho: Tupla (altura, largura)
        
    Returns:
        numpy.ndarray: Imagem sintetica
    """
    h, w = tamanho
    
    if tipo == 'gradiente':
        # Gradiente horizontal
        imagem = np.linspace(0, 255, w, dtype=np.uint8)
        imagem = np.tile(imagem, (h, 1))
        
    elif tipo == 'xadrez':
        # Padrao xadrez
        imagem = np.zeros((h, w), dtype=np.uint8)
        quadrado = 20
        for i in range(0, h, quadrado):
            for j in range(0, w, quadrado):
                if ((i // quadrado) + (j // quadrado)) % 2 == 0:
                    imagem[i:i+quadrado, j:j+quadrado] = 255
                    
    elif tipo == 'circulo':
        # Circulo branco em fundo preto
        imagem = np.zeros((h, w), dtype=np.uint8)
        centro_y, centro_x = h // 2, w // 2
        raio = min(h, w) // 4
        
        for y in range(h):
            for x in range(w):
                if (y - centro_y)**2 + (x - centro_x)**2 <= raio**2:
                    imagem[y, x] = 255
                    
    elif tipo == 'ruido':
        # Ruido aleatorio
        imagem = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        
    else:
        raise ValueError(f"Tipo '{tipo}' nao reconhecido")
    
    return imagem
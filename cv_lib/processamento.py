# minha_cv_lib/processamento.py

import numpy as np

def rgb_para_cinza(imagem_rgb, tipo='luminancia'):
    """
    Converte uma imagem RGB para escala de cinza usando a fórmula de luminância ou média.
    
    Argumentos:
    imagem_rgb -- um array NumPy com shape (altura, largura, 3)
    tipo -- o tipo de conversão ('luminancia' ou 'media')
    """
    altura, largura, _ = imagem_rgb.shape
    imagem_cinza = np.zeros((altura, largura), dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            pixel = imagem_rgb[y, x]
            r, g, b = pixel[0], pixel[1], pixel[2]
            if tipo == 'luminancia':
                valor_cinza = 0.299 * r + 0.587 * g + 0.114 * b
            elif tipo == 'media':
                valor_cinza = (r + g + b) / 3
            imagem_cinza[y, x] = valor_cinza

    imagem_cinza = imagem_cinza.astype(np.uint8)

    return imagem_cinza
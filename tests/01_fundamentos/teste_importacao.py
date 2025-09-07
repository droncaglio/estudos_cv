#!/usr/bin/env python3
# Teste simples para verificar se a biblioteca está funcionando

import numpy as np
from cv_lib import processamento

# Criar uma imagem RGB de teste simples
imagem_teste = np.array([[[255, 0, 0], [0, 255, 0]], 
                        [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)

print("Imagem RGB de teste:")
print(imagem_teste)
print(f"Shape: {imagem_teste.shape}")

# Testar conversão para cinza
try:
    cinza_luminancia = processamento.rgb_para_cinza(imagem_teste, tipo='luminancia')
    cinza_media = processamento.rgb_para_cinza(imagem_teste, tipo='media')
    
    print("\n✅ Conversão luminância:")
    print(cinza_luminancia)
    
    print("\n✅ Conversão média:")
    print(cinza_media)
    
    print("\n🎉 Tudo funcionando!")
    
except Exception as e:
    print(f"\n❌ Erro: {e}")
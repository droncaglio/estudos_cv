#!/usr/bin/env python3
# Teste completo das diferentes conversoes RGB para cinza

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from cv_lib import processamento

# Carrega imagem de teste
imagem_rgb = data.chelsea()  # Gato colorido

# Lista todos os tipos de conversao
tipos = ['luminancia', 'bt709', 'media', 'desaturacao', 'canal_r', 'canal_g', 'canal_b']

# Cria subplots para comparar
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# Mostra original
axes[0].imshow(imagem_rgb)
axes[0].set_title("Original RGB")
axes[0].axis('off')

# Aplica cada conversao
for i, tipo in enumerate(tipos, 1):
    try:
        resultado = processamento.rgb_para_cinza(imagem_rgb, tipo=tipo)
        axes[i].imshow(resultado, cmap='gray')
        axes[i].set_title(f"{tipo}")
        axes[i].axis('off')
        print(f"✅ {tipo}: min={resultado.min()}, max={resultado.max()}")
    except Exception as e:
        print(f"❌ Erro em {tipo}: {e}")

plt.tight_layout()
plt.show()

print("\n📊 Comparacao das diferentes conversoes implementadas:")
print("• bt601/luminancia: Padrao classico, equilibra bem todos os canais")
print("• bt709: Padrao moderno HDTV, da mais peso ao verde")  
print("• media: Mais simples, pode dar resultados 'chapados'")
print("• desaturacao: Mantem mais detalhes em areas coloridas")
print("• canal_r/g/b: Util para analise especifica de cada cor")
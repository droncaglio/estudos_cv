#!/usr/bin/env python3
"""
Teste completo e organizado das conversoes RGB para escala de cinza.

Este script demonstra todas as conversoes implementadas na biblioteca cv_lib,
com visualizacao organizada e analise comparativa dos resultados.
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

def main():
    """Funcao principal do teste."""
    print("🧪 Teste Completo: Conversoes RGB para Escala de Cinza")
    print("=" * 60)
    
    # Carrega imagem de teste
    print("📂 Carregando imagem de teste...")
    imagem_rgb = datasets.carregar_imagem_teste('chelsea')
    datasets.info_imagem(imagem_rgb, 'Chelsea (gato)')
    
    # Lista todos os tipos de conversao implementados
    tipos_conversao = [
        'luminancia', 'bt709', 'media', 'desaturacao', 
        'canal_r', 'canal_g', 'canal_b'
    ]
    
    print(f"🔧 Testando {len(tipos_conversao)} tipos de conversao...")
    
    # Usa a funcao utilitaria para mostrar grid comparativo
    visualization.mostrar_grid_conversoes(
        imagem_rgb=imagem_rgb,
        funcao_conversao=processamento.rgb_para_cinza,
        tipos_conversao=tipos_conversao,
        figsize=(16, 10)
    )
    
    # Analise quantitativa
    print("\n📊 Analise Quantitativa dos Resultados:")
    print("-" * 50)
    
    resultados = {}
    for tipo in tipos_conversao:
        try:
            resultado = processamento.rgb_para_cinza(imagem_rgb, tipo=tipo)
            resultados[tipo] = resultado
            
            print(f"{tipo:12} | min: {resultado.min():3d} | max: {resultado.max():3d} | "
                  f"media: {resultado.mean():6.2f} | std: {resultado.std():6.2f}")
                  
        except Exception as e:
            print(f"{tipo:12} | ERRO: {e}")
    
    # Comparacao com implementacao do scikit-image (se disponivel)
    try:
        from skimage.color import rgb2gray
        sk_resultado = (rgb2gray(imagem_rgb) * 255).astype(np.uint8)
        
        print("\n🔍 Comparacao com scikit-image:")
        luminancia_resultado = resultados.get('luminancia')
        if luminancia_resultado is not None:
            diferenca = np.abs(luminancia_resultado.astype(float) - sk_resultado.astype(float))
            print(f"Diferenca media com scikit-image: {diferenca.mean():.2f}")
            print(f"Diferenca maxima: {diferenca.max():.2f}")
            
            # Se a diferenca for pequena, nossas implementacoes estao corretas!
            if diferenca.mean() < 1.0:
                print("✅ Nossa implementacao 'luminancia' esta correta!")
            else:
                print("⚠️  Diferenca significativa detectada")
    
    except ImportError:
        print("ℹ️  scikit-image nao disponivel para comparacao")
    
    # Salva alguns resultados para analise posterior
    print(f"\n💾 Salvando resultados em assets/results/...")
    results_dir = project_root / "assets" / "results"
    results_dir.mkdir(exist_ok=True)
    
    for tipo in ['luminancia', 'bt709', 'desaturacao']:
        if tipo in resultados:
            caminho = results_dir / f"conversao_{tipo}_chelsea.png"
            visualization.salvar_resultado(resultados[tipo], caminho)
    
    print("\n🎉 Teste concluido com sucesso!")
    print("\n💡 Observacoes importantes:")
    print("• bt601/luminancia: Equilibra bem os canais, padrao classico")
    print("• bt709: Mais peso no verde, melhor para displays modernos") 
    print("• desaturacao: Preserva contraste em areas coloridas")
    print("• canais individuais: Uteis para analises especificas")

if __name__ == "__main__":
    main()
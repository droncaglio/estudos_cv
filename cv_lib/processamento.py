# minha_cv_lib/processamento.py

import numpy as np

def rgb_para_cinza(imagem_rgb, tipo='luminancia'):
    """
    Converte uma imagem RGB para escala de cinza usando diferentes metodos.
    
    Argumentos:
    imagem_rgb -- um array NumPy com shape (altura, largura, 3)
    tipo -- o tipo de conversao:
        'luminancia' ou 'bt601': Formula padrao (0.299*R + 0.587*G + 0.114*B)
        'bt709': Formula HDTV (0.2126*R + 0.7152*G + 0.0722*B) 
        'media': Media aritmetica simples
        'desaturacao': (max + min) / 2
        'canal_r': Apenas canal vermelho
        'canal_g': Apenas canal verde  
        'canal_b': Apenas canal azul
    """
    altura, largura, _ = imagem_rgb.shape
    imagem_cinza = np.zeros((altura, largura), dtype=np.float64)

    for y in range(altura):
        for x in range(largura):
            pixel = imagem_rgb[y, x]
            r, g, b = pixel[0], pixel[1], pixel[2]
            
            if tipo in ['luminancia', 'bt601']:
                # ITU-R BT.601 (SDTV): Padrao classico baseado na sensibilidade do olho humano
                # Aplicacoes: Processamento geral, compatibilidade com sistemas antigos
                # O olho e mais sensivel ao verde (58.7%), depois vermelho (29.9%) e azul (11.4%)
                valor_cinza = 0.299 * r + 0.587 * g + 0.114 * b
                
            elif tipo == 'bt709':
                # ITU-R BT.709 (HDTV): Padrao moderno para TV digital e monitores
                # Aplicacoes: Processamento para displays modernos, conteudo HD/4K
                # Coefficientes ajustados para fosforos modernos (mais peso no verde: 71.52%)
                valor_cinza = 0.2126 * r + 0.7152 * g + 0.0722 * b
                
            elif tipo == 'media':
                # Media aritmetica simples: Trata todos os canais igualmente
                # Aplicacoes: Algoritmos simples, prototipagem rapida, quando nao ha preferencia de canal
                # Pode resultar em imagens "chapadas" pois ignora sensibilidade do olho humano
                valor_cinza = (r + g + b) / 3
                
            elif tipo == 'desaturacao':
                # Desaturacao: Media entre valor maximo e minimo dos canais RGB
                # Aplicacoes: Arte digital, quando se quer preservar contraste de cores saturadas
                # Mantem melhor os detalhes em areas muito coloridas comparado a luminancia
                valor_cinza = (max(r, g, b) + min(r, g, b)) / 2
                
            elif tipo == 'canal_r':
                # Canal vermelho isolado
                # Aplicacoes: Deteccao de sangue/vasos sanguineos, analise de vegetacao (contraste com clorofila),
                # fotografia infravermelha, deteccao de pele em imagens medicas
                valor_cinza = r
                
            elif tipo == 'canal_g':
                # Canal verde isolado  
                # Aplicacoes: Analise de vegetacao (clorofila), deteccao de plantas em agricultura,
                # melhor canal para deteccao de bordas (mais detalhado), visao noturna
                valor_cinza = g
                
            elif tipo == 'canal_b':
                # Canal azul isolado
                # Aplicacoes: Deteccao de agua, analise de ceu/atmosfera, deteccao de veias,
                # contraste em imagens medicas, analise de poluicao atmosferica
                valor_cinza = b
            else:
                raise ValueError(f"Tipo '{tipo}' nao reconhecido")
                
            imagem_cinza[y, x] = valor_cinza

    imagem_cinza = imagem_cinza.astype(np.uint8)

    return imagem_cinza
# cv_lib/utils/validacao.py
"""
🛡️ Funções de validação para entrada de dados

Este módulo contém funções auxiliares para validar tipos, shapes e ranges
de imagens e parâmetros, garantindo robustez nas operações.
"""

import numpy as np

def validar_imagem_rgb(imagem, nome_param="imagem"):
    """
    Valida se a entrada é uma imagem RGB válida.
    
    Args:
        imagem: Array a ser validado
        nome_param: Nome do parâmetro para mensagens de erro
        
    Returns:
        tuple: (altura, largura, canais) da imagem validada
        
    Raises:
        ValueError: Se a imagem não atender aos critérios
    """
    if not isinstance(imagem, np.ndarray):
        raise ValueError(f"{nome_param} deve ser um array NumPy")
    
    if len(imagem.shape) == 2:
        # Imagem em escala de cinza - adiciona dimensão de canal
        return imagem.shape[0], imagem.shape[1], 1
        
    if len(imagem.shape) != 3:
        raise ValueError(f"{nome_param} deve ter 2 ou 3 dimensões, mas tem {len(imagem.shape)}")
    
    altura, largura, canais = imagem.shape
    
    if canais != 3:
        raise ValueError(f"{nome_param} deve ter 3 canais (RGB), mas tem {canais}")
    
    if imagem.dtype not in [np.uint8, np.float32, np.float64]:
        raise ValueError(f"{nome_param} deve ser uint8, float32 ou float64, mas é {imagem.dtype}")
    
    return altura, largura, canais


def validar_parametro_numerico(valor, nome_param, min_val=None, max_val=None):
    """
    Valida parâmetros numéricos dentro de ranges específicos.
    
    Args:
        valor: Valor a ser validado
        nome_param: Nome do parâmetro para mensagens de erro
        min_val: Valor mínimo permitido (opcional)
        max_val: Valor máximo permitido (opcional)
        
    Raises:
        ValueError: Se o valor não atender aos critérios
    """
    if not isinstance(valor, (int, float, np.number)):
        raise ValueError(f"{nome_param} deve ser um número, mas é {type(valor)}")
    
    if min_val is not None and valor < min_val:
        raise ValueError(f"{nome_param} deve ser >= {min_val}, mas é {valor}")
    
    if max_val is not None and valor > max_val:
        raise ValueError(f"{nome_param} deve ser <= {max_val}, mas é {valor}")


def validar_shapes_compativeis(img1, img2, nome1="img1", nome2="img2"):
    """
    Valida se duas imagens têm shapes compatíveis para operações.
    
    Args:
        img1, img2: Arrays das imagens
        nome1, nome2: Nomes dos parâmetros para mensagens de erro
        
    Raises:
        ValueError: Se as shapes não forem compatíveis
    """
    if img1.shape != img2.shape:
        raise ValueError(f"{nome1}.shape {img1.shape} != {nome2}.shape {img2.shape}")


def garantir_uint8(imagem):
    """
    Garante que a imagem está no formato uint8 com valores [0-255].
    
    Args:
        imagem: Array da imagem
        
    Returns:
        np.ndarray: Imagem convertida para uint8
    """
    # Se já está em uint8, retorna cópia
    if imagem.dtype == np.uint8:
        return imagem.copy()
    
    # Clipa valores no range [0-255] e converte
    imagem_clipada = np.clip(imagem, 0, 255)
    return imagem_clipada.astype(np.uint8)


def log_operacao(nome_funcao, parametros=None):
    """
    Log educacional para debug de operações (opcional).
    
    Args:
        nome_funcao: Nome da função sendo executada
        parametros: Dicionário com parâmetros relevantes
    """
    if parametros:
        params_str = ", ".join([f"{k}={v}" for k, v in parametros.items()])
        print(f"🔧 Executando {nome_funcao}({params_str})")
    else:
        print(f"🔧 Executando {nome_funcao}()")
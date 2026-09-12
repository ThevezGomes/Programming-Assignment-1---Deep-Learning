"""
Módulo de inferência de segmentação de instâncias para o PA1.
Recebe o caminho de uma imagem qualquer, devolve a máscara de instâncias colorida e a contagem.
Roda sem retreinar carregando os pesos do checkpoint.
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    from .models import carregar_checkpoint
    from .postprocessing import inferir_imagem, colorir_mascara_instancias, decodificar_watershed_trilha_a
except ImportError:
    from models import carregar_checkpoint
    from postprocessing import inferir_imagem, colorir_mascara_instancias, decodificar_watershed_trilha_a


def carregar_modelo_inferencia(caminho_checkpoint="checkpoints/checkpoint.pt", device=None):
    """
    Carrega o modelo treinado a partir do checkpoint para inferência imediata sem retreinar.
    """
    return carregar_checkpoint(filepath=caminho_checkpoint, device=device)


def inferir(caminho_imagem, model=None, device=None, exibir=True, caminho_checkpoint="checkpoints/checkpoint.pt"):
    """
    Executa a inferência de segmentação de instâncias conforme o enunciado do PA1:
    Recebe o caminho de uma imagem qualquer, devolve a máscara de instâncias colorida e a contagem.

    Parâmetros:
      - caminho_imagem: Caminho da imagem (str) no disco.
      - model: Modelo UNetResNet carregado. Se None, carrega automaticamente do checkpoint.
      - device: Dispositivo onde executar (cuda, mps ou cpu).
      - exibir: Se True, plota a imagem de entrada e a máscara colorida com a contagem.
      - caminho_checkpoint: Caminho alternativo do checkpoint se model for None.

    Retorna:
      - mascara_colorida: Imagem RGB (H, W, 3) com cada instância em uma cor vibrante única (fundo preto).
      - contagem: Número inteiro de instâncias detectadas.
    """
    if model is None:
        model = carregar_modelo_inferencia(caminho_checkpoint=caminho_checkpoint, device=device)

    resultado = inferir_imagem(model, caminho_imagem, device=device)
    mascara_colorida = resultado['colored_mask']
    contagem = resultado['count']

    if exibir:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(resultado['image_rgb'])
        ax[0].set_title("Imagem de Entrada")
        ax[0].axis("off")

        ax[1].imshow(mascara_colorida)
        ax[1].set_title(f"Máscara de Instâncias (Contagem: {contagem})")
        ax[1].axis("off")

        plt.tight_layout()
        plt.show()

    return mascara_colorida, contagem

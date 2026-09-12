import os
import numpy as np
import cv2
import scipy.ndimage as ndi
from scipy.ndimage import label
import torch
import torch.nn.functional as F

try:
    from skimage.segmentation import watershed
except ImportError:
    watershed = None


def extract_instances_naive(pred_prob, threshold=0.5):
    """Extrai instâncias binarizando por limiar e aplicando componentes conexos de 8-conectividade (Item 2)."""
    binary_mask = pred_prob > threshold
    structure = np.ones((3, 3), dtype=int)
    labeled_mask, num_instances = label(binary_mask, structure=structure)
    return labeled_mask, num_instances


def decodificar_watershed_trilha_a(pred_probs, threshold_interior=0.35, threshold_fg=0.35, min_marker_size=1):
    """Decodifica as probabilidades de 3 classes em instâncias separadas usando o algoritmo Watershed."""
    if pred_probs.shape[0] == 3 and pred_probs.ndim == 3:
        p_bg = pred_probs[0]
        p_interior = pred_probs[1]
        p_border = pred_probs[2]
    elif pred_probs.shape[-1] == 3 and pred_probs.ndim == 3:
        p_bg = pred_probs[:, :, 0]
        p_interior = pred_probs[:, :, 1]
        p_border = pred_probs[:, :, 2]
    else:
        raise ValueError("pred_probs deve conter 3 canais de probabilidades.")

    # 1. Marcadores a partir da probabilidade de interior
    interior_binary = (p_interior > threshold_interior)
    markers, num_markers = label(interior_binary)

    if min_marker_size > 1 and num_markers > 0:
        sizes = ndi.sum(np.ones_like(markers), markers, range(1, num_markers + 1))
        small_mask = np.isin(markers, np.where(sizes < min_marker_size)[0] + 1)
        markers[small_mask] = 0
        markers, num_markers = label(markers > 0)

    if num_markers == 0:
        return np.zeros(p_bg.shape, dtype=np.int32), 0

    # 2. Máscara de primeiro plano para limitar a expansão do Watershed
    fg_mask = (p_interior + p_border > threshold_fg) | (markers > 0)

    # 3. Superfície topográfica física com vales nos centros e cristas nas fronteiras
    surface = -p_interior + p_border

    # 4. Inundação por Watershed com skimage ou fallback do OpenCV restrito à máscara
    if watershed is not None:
        labeled_instances = watershed(surface, markers=markers, mask=fg_mask)
    else:
        norm_surf = cv2.normalize(surface, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_3ch = cv2.cvtColor(norm_surf, cv2.COLOR_GRAY2BGR)
        cv2_markers = markers.copy().astype(np.int32)
        cv2.watershed(img_3ch, cv2_markers)
        labeled_instances = np.where((cv2_markers > 0) & fg_mask, cv2_markers, 0)

    unique_ids = np.unique(labeled_instances[labeled_instances > 0])
    final_mask = np.zeros_like(labeled_instances, dtype=np.int32)
    for new_id, old_id in enumerate(unique_ids, start=1):
        final_mask[labeled_instances == old_id] = new_id

    return final_mask, len(unique_ids)


def colorir_mascara_instancias(labeled_mask, seed=42):
    """
    Converte uma máscara rotulada com IDs inteiros de instâncias em uma imagem RGB colorida.
    O fundo (0) permanece preto, e cada instância recebe uma cor vibrante e distinta.
    """
    if labeled_mask.ndim != 2:
        raise ValueError("labeled_mask deve ser uma matriz 2D.")

    unique_ids = np.unique(labeled_mask)
    unique_ids = unique_ids[unique_ids > 0]

    h, w = labeled_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    if len(unique_ids) == 0:
        return colored

    rng = np.random.RandomState(seed)
    max_id = int(unique_ids.max())
    # Gera paleta de cores contrastantes (RGB entre 50 e 255 para evitar tons apagados)
    palette = rng.randint(50, 256, size=(max_id + 1, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # Background preto

    colored = palette[labeled_mask]
    return colored


def inferir_imagem(model, image_input, device=None, threshold_interior=0.35, threshold_fg=0.35, min_marker_size=1):
    """
    Executa a inferência completa em uma imagem (caminho no disco ou array numpy),
    retornando a máscara rotulada, contagem de instâncias, máscara colorida RGB e probabilidades das classes.

    Parâmetros:
      - model: Modelo PyTorch treinado (UNetResNet com 3 classes).
      - image_input: Caminho do arquivo de imagem (str) OU array numpy (H, W) / (H, W, 3) / (H, W, 4).
      - device: Dispositivo onde executar (CPU, CUDA ou MPS).
      - threshold_interior: Limiar para binarização dos núcleos/sementes de interior.
      - threshold_fg: Limiar para máscara de primeiro plano do Watershed.
      - min_marker_size: Tamanho mínimo em pixels para validar um marcador.

    Retorna dicionário contendo:
      - 'labeled_mask': matriz 2D com IDs inteiros de cada instância (0=fundo).
      - 'count': contagem inteira de instâncias detectadas.
      - 'colored_mask': imagem RGB (H, W, 3) colorida para visualização.
      - 'pred_probs': mapa de probabilidades (3, H, W).
      - 'image_rgb': imagem de entrada convertida para RGB uint8 (H, W, 3).
    """
    # 1. Carregamento e padronização da imagem
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Imagem não encontrada: {image_input}")
        img_bgr = cv2.imread(image_input, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Não foi possível decodificar a imagem: {image_input}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        img_rgb = image_input.copy()
        if img_rgb.ndim == 2:
            img_rgb = np.stack([img_rgb] * 3, axis=-1)
        elif img_rgb.ndim == 3 and img_rgb.shape[-1] == 4:
            img_rgb = img_rgb[:, :, :3]
        if img_rgb.dtype != np.uint8:
            if img_rgb.max() <= 1.0:
                img_rgb = (img_rgb * 255).astype(np.uint8)
            else:
                img_rgb = img_rgb.astype(np.uint8)
    else:
        raise TypeError("image_input deve ser string (caminho) ou np.ndarray.")

    if device is None:
        device = next(model.parameters()).device

    # 2. Inferência pelo modelo
    model.eval()
    tensor_img = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor_img = tensor_img.to(device)

    with torch.no_grad():
        logits = model(tensor_img)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (3, H, W)

    # 3. Decodificação por Watershed
    labeled_mask, count = decodificar_watershed_trilha_a(
        probs,
        threshold_interior=threshold_interior,
        threshold_fg=threshold_fg,
        min_marker_size=min_marker_size
    )

    # 4. Colorização da máscara
    colored_mask = colorir_mascara_instancias(labeled_mask)

    return {
        "labeled_mask": labeled_mask,
        "count": count,
        "colored_mask": colored_mask,
        "pred_probs": probs,
        "image_rgb": img_rgb
    }

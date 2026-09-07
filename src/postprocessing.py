import numpy as np
import cv2
import scipy.ndimage as ndi
from scipy.ndimage import label

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

    # 1. Marcadores a partir da probabilidade de interior sem atenuação de blur
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

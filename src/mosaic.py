import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    from .postprocessing import decodificar_watershed_trilha_a, extract_instances_naive
    from .metrics import calculate_instance_iou_matrix, match_instances_hungarian, match_instances_greedy
except ImportError:
    from postprocessing import decodificar_watershed_trilha_a, extract_instances_naive
    from metrics import calculate_instance_iou_matrix, match_instances_hungarian, match_instances_greedy


class DisjointSetUnion:
    """Estrutura Disjoint Set Union (DSU / Union-Find) para agrupamento de instâncias entre tiles."""
    def __init__(self):
        self.parent = {}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
            return i
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def criar_mosaico_imagens(images, masks, grid_shape=(2, 2)):
    """
    Parte 4 (Item 1): Monta uma imagem grande (mosaico de várias imagens do dataset)
    e reindexa as máscaras de instâncias para garantir unicidade global de cada núcleo.
    """
    rows, cols = grid_shape
    assert len(images) >= rows * cols, f"Imagens insuficientes ({len(images)}) para a grade {grid_shape}"
    h, w, c = images[0].shape
    mosaic_img = np.zeros((h * rows, w * cols, c), dtype=images[0].dtype)
    mosaic_mask = np.zeros((h * rows, w * cols), dtype=np.int32)

    current_max_id = 0
    idx = 0
    for r in range(rows):
        for c_idx in range(cols):
            img_curr = images[idx]
            mask_curr = masks[idx]

            y_start = r * h
            x_start = c_idx * w

            mosaic_img[y_start:y_start + h, x_start:x_start + w] = img_curr

            u_ids = np.unique(mask_curr[mask_curr > 0])
            reindexed_mask = np.zeros_like(mask_curr, dtype=np.int32)
            for new_offset, old_id in enumerate(u_ids, start=1):
                reindexed_mask[mask_curr == old_id] = current_max_id + new_offset
            current_max_id += len(u_ids)

            mosaic_mask[y_start:y_start + h, x_start:x_start + w] = reindexed_mask
            idx += 1

    return mosaic_img, mosaic_mask


def inferencia_mosaico_tiles(model, mosaic_img, tile_size=128, stride=96, device="cpu",
                             threshold_interior=0.35, threshold_fg=0.35, min_marker_size=1):
    """
    Parte 4 (Item 2): Executa inferência em tiles com sobreposição (slide 83 da aula).
    - Agrega média de probabilidades semânticas nos patches sobrepostos ('average the results');
    - Extrai instâncias locais por tile via Watershed (Trilha A) ou limiar ingênuo;
    - Monta o mosaico antes da correção considerando a parte interna de cada patch ('consider the inner part'),
      evidenciando a fratura de objetos que cruzam as fronteiras.
    """
    model.eval()
    H, W = mosaic_img.shape[:2]

    y_starts = list(range(0, H - tile_size + 1, stride))
    if y_starts[-1] + tile_size < H:
        y_starts.append(H - tile_size)
    x_starts = list(range(0, W - tile_size + 1, stride))
    if x_starts[-1] + tile_size < W:
        x_starts.append(W - tile_size)

    y_starts = sorted(list(set(y_starts)))
    x_starts = sorted(list(set(x_starts)))

    dummy_in = torch.zeros(1, 3, tile_size, tile_size, device=device)
    with torch.no_grad():
        dummy_out = model(dummy_in)
    out_channels = dummy_out.shape[1]

    semantic_prob_mosaic = np.zeros((out_channels, H, W), dtype=np.float32)
    weight_mosaic = np.zeros((H, W), dtype=np.float32)

    tiles_data = []
    tile_idx = 0

    with torch.no_grad():
        for y in y_starts:
            for x in x_starts:
                patch = mosaic_img[y:y + tile_size, x:x + tile_size]
                patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1).unsqueeze(0).contiguous().to(device) / 255.0
                logits = model(patch_tensor)

                if out_channels == 3:
                    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    labeled_mask, num_instances = decodificar_watershed_trilha_a(
                        probs, threshold_interior=threshold_interior, threshold_fg=threshold_fg, min_marker_size=min_marker_size
                    )
                else:
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                    if probs.ndim == 2:
                        labeled_mask, num_instances = extract_instances_naive(probs, threshold=0.5)
                        probs = np.expand_dims(probs, 0)
                    else:
                        labeled_mask, num_instances = extract_instances_naive(probs[0], threshold=0.5)

                semantic_prob_mosaic[:, y:y + tile_size, x:x + tile_size] += probs
                weight_mosaic[y:y + tile_size, x:x + tile_size] += 1.0

                tiles_data.append({
                    'tile_id': tile_idx,
                    'y': y, 'x': x,
                    'local_mask': labeled_mask,
                    'num_instances': num_instances,
                    'probs': probs,
                    'center': (y + tile_size / 2.0, x + tile_size / 2.0)
                })
                tile_idx += 1

    semantic_prob_mosaic /= np.maximum(weight_mosaic, 1e-6)

    pred_mosaic_sem_fusao = np.zeros((H, W), dtype=np.int32)
    tile_id_offsets = {}
    global_counter = 0
    for t in tiles_data:
        tile_id_offsets[t['tile_id']] = global_counter
        global_counter += t['num_instances']

    for r in range(H):
        for c in range(W):
            best_t = None
            min_d2 = 1e9
            for t in tiles_data:
                if t['y'] <= r < t['y'] + tile_size and t['x'] <= c < t['x'] + tile_size:
                    cy, cx = t['center']
                    d2 = (r - cy) ** 2 + (c - cx) ** 2
                    if d2 < min_d2:
                        min_d2 = d2
                        best_t = t
            if best_t is not None:
                loc_id = best_t['local_mask'][r - best_t['y'], c - best_t['x']]
                if loc_id > 0:
                    pred_mosaic_sem_fusao[r, c] = tile_id_offsets[best_t['tile_id']] + loc_id

    u_sem = np.unique(pred_mosaic_sem_fusao[pred_mosaic_sem_fusao > 0])
    pred_sem_compact = np.zeros_like(pred_mosaic_sem_fusao)
    for new_id, old_id in enumerate(u_sem, start=1):
        pred_sem_compact[pred_mosaic_sem_fusao == old_id] = new_id

    return pred_sem_compact, semantic_prob_mosaic, tiles_data


def fundir_instancias_tiles(tiles_data, image_shape, tile_size=128, min_overlap_iou=0.20, min_relative_inter=0.30):
    """
    Parte 4 (Item 4): Algoritmo de correção por fusão de instâncias entre tiles sobrepostos.
    - Avalia a concordância espacial de cada par de instâncias na faixa de sobreposição (overlap strip);
    - Agrupa os fragmentos correspondentes ao mesmo objeto biológico via Disjoint Set Union (Union-Find);
    - Reconstrói a máscara global contínua unificando as metades cortadas sob o mesmo ID global.
    """
    H, W = image_shape[:2]
    dsu = DisjointSetUnion()

    num_tiles = len(tiles_data)
    for i in range(num_tiles):
        for j in range(i + 1, num_tiles):
            tA = tiles_data[i]
            tB = tiles_data[j]

            y_min = max(tA['y'], tB['y'])
            y_max = min(tA['y'] + tile_size, tB['y'] + tile_size)
            x_min = max(tA['x'], tB['x'])
            x_max = min(tA['x'] + tile_size, tB['x'] + tile_size)

            if y_max > y_min and x_max > x_min:
                subA = tA['local_mask'][y_min - tA['y']:y_max - tA['y'], x_min - tA['x']:x_max - tA['x']]
                subB = tB['local_mask'][y_min - tB['y']:y_max - tB['y'], x_min - tB['x']:x_max - tB['x']]

                uA = np.unique(subA[subA > 0])
                uB = np.unique(subB[subB > 0])

                for instA in uA:
                    maskA = (subA == instA)
                    areaA = maskA.sum()
                    for instB in uB:
                        maskB = (subB == instB)
                        inter = np.logical_and(maskA, maskB).sum()
                        if inter > 0:
                            areaB = maskB.sum()
                            union = areaA + areaB - inter
                            iou = inter / union
                            rel_inter = inter / min(areaA, areaB)

                            if iou >= min_overlap_iou or rel_inter >= min_relative_inter:
                                dsu.union((tA['tile_id'], instA), (tB['tile_id'], instB))

    root_to_id = {}
    curr_id = 1
    for t in tiles_data:
        for inst in range(1, t['num_instances'] + 1):
            r = dsu.find((t['tile_id'], inst))
            if r not in root_to_id:
                root_to_id[r] = curr_id
                curr_id += 1

    pred_mosaic_com_fusao = np.zeros((H, W), dtype=np.int32)
    for t in tiles_data:
        for r_loc in range(tile_size):
            for c_loc in range(tile_size):
                inst = t['local_mask'][r_loc, c_loc]
                if inst > 0:
                    gy = t['y'] + r_loc
                    gx = t['x'] + c_loc
                    gid = root_to_id[dsu.find((t['tile_id'], inst))]
                    pred_mosaic_com_fusao[gy, gx] = gid

    return pred_mosaic_com_fusao


def avaliar_e_comparar_mosaico(mosaic_gt, pred_sem_fusao, pred_com_fusao,
                               iou_thresholds=np.arange(0.50, 1.00, 0.05), matching_method="hungarian"):
    """
    Parte 4 (Item 4): Avalia e compara quantitativamente o mAP@[.50:.95] e o erro absoluto de contagem
    no mosaico completo antes e depois da correção por fusão de instâncias.
    """
    num_gt = len(np.unique(mosaic_gt[mosaic_gt > 0]))
    num_sem = len(np.unique(pred_sem_fusao[pred_sem_fusao > 0]))
    num_com = len(np.unique(pred_com_fusao[pred_com_fusao > 0]))

    iou_mat_sem = calculate_instance_iou_matrix(mosaic_gt, num_gt, pred_sem_fusao, num_sem)
    iou_mat_com = calculate_instance_iou_matrix(mosaic_gt, num_gt, pred_com_fusao, num_com)

    aps_sem = []
    aps_com = []

    match_fn = match_instances_hungarian if matching_method == "hungarian" else match_instances_greedy

    for t in iou_thresholds:
        tp_sem = match_fn(iou_mat_sem, t)
        denom_sem = num_gt + num_sem - tp_sem
        aps_sem.append(tp_sem / denom_sem if denom_sem > 0 else 0.0)

        tp_com = match_fn(iou_mat_com, t)
        denom_com = num_gt + num_com - tp_com
        aps_com.append(tp_com / denom_com if denom_com > 0 else 0.0)

    aps_sem = np.array(aps_sem, dtype=np.float32)
    aps_com = np.array(aps_com, dtype=np.float32)

    map_sem = float(np.mean(aps_sem))
    map_com = float(np.mean(aps_com))

    err_sem = abs(num_sem - num_gt)
    err_com = abs(num_com - num_gt)

    print(f"\n=======================================================================")
    print(f"      COMPARAÇÃO NO MOSAICO: ANTES VS DEPOIS DA FUSÃO DE INSTÂNCIAS     ")
    print(f"=======================================================================")
    print(f"{'Métrica':<35} | {'Antes da Fusão':<16} | {'Depois da Fusão':<16} | {'Variação':<10}")
    print("-" * 75)
    print(f"{'mAP@[.50:.95]':<35} | {map_sem:<16.4f} | {map_com:<16.4f} | {map_com - map_sem:+10.4f}")
    print(f"{'AP @ IoU=0.50':<35} | {aps_sem[0]:<16.4f} | {aps_com[0]:<16.4f} | {aps_com[0] - aps_sem[0]:+10.4f}")
    print(f"{'AP @ IoU=0.75':<35} | {aps_sem[5]:<16.4f} | {aps_com[5]:<16.4f} | {aps_com[5] - aps_sem[5]:+10.4f}")
    print(f"{'Número de Instâncias Preditas':<35} | {num_sem:<16d} | {num_com:<16d} | {num_com - num_sem:+10d}")
    print(f"{'Instâncias no Ground Truth':<35} | {num_gt:<16d} | {num_gt:<16d} | {'--':<10}")
    print(f"{'Erro Absoluto de Contagem':<35} | {err_sem:<16d} | {err_com:<16d} | {err_com - err_sem:+10d}")
    print("=" * 75)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(iou_thresholds, aps_sem, 'r-o', linewidth=2, label=f'Antes da Fusão (mAP = {map_sem:.4f})')
    plt.plot(iou_thresholds, aps_com, 'b-s', linewidth=2, label=f'Depois da Fusão (mAP = {map_com:.4f})')
    plt.title('Precisão Média (AP) por Limiar de IoU')
    plt.xlabel('Limiar de IoU')
    plt.ylabel('AP')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    categories = ['mAP@[.50:.95] (x100)', 'Erro de Contagem']
    antes_vals = [map_sem * 100, err_sem]
    depois_vals = [map_com * 100, err_com]

    x_bar = np.arange(len(categories))
    bar_width = 0.35
    plt.bar(x_bar - bar_width / 2, antes_vals, bar_width, label='Antes da Fusão', color='crimson', alpha=0.8)
    plt.bar(x_bar + bar_width / 2, depois_vals, bar_width, label='Depois da Fusão', color='royalblue', alpha=0.8)
    plt.xticks(x_bar, categories)
    plt.ylabel('Valor')
    plt.title('Impacto da Correção por Fusão')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    return {
        "mAP_sem": map_sem, "mAP_com": map_com,
        "err_sem": err_sem, "err_com": err_com,
        "aps_sem": aps_sem, "aps_com": aps_com,
        "num_gt": num_gt, "num_sem": num_sem, "num_com": num_com
    }

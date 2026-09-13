import numpy as np
import matplotlib.pyplot as plt
import torch

try:
    from .metrics import (
        calculate_dice_coefficient, calculate_iou,
        evaluate_instance_level, evaluate_instance_level_trilha_a
    )
    from .postprocessing import extract_instances_naive, decodificar_watershed_trilha_a
except ImportError:
    from metrics import (
        calculate_dice_coefficient, calculate_iou,
        evaluate_instance_level, evaluate_instance_level_trilha_a
    )
    from postprocessing import extract_instances_naive, decodificar_watershed_trilha_a


def plot_dataset_samples(X_train, y_train, X_val, y_val, X_test, y_test):
    """Exibe amostras dos conjuntos de Treino, Validação e Teste."""
    plt.figure(figsize=(12, 6))

    samples = [
        ("Treino 1", X_train[0], y_train[0]),
        ("Treino 2", X_train[1], y_train[1]),
        ("Treino 3", X_train[2], y_train[2]),
        ("Val 1", X_val[0], y_val[0]),
        ("Test 1", X_test[0], y_test[0]),
    ]

    for i, (title, img, mask) in enumerate(samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(f'Img ({title})')
        plt.axis('off')

        plt.subplot(2, 5, i + 6)
        plt.imshow(mask > 0, cmap='gray')
        plt.title(f'Máscara ({title})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_predictions(model, X_test, y_test, device, num_samples=5, threshold=0.5):
    """Visualiza as predições do modelo nas amostras do conjunto de teste comparando com o Ground Truth."""
    model.eval()

    with torch.no_grad():
        for i in range(min(num_samples, len(X_test))):
            img = X_test[i]
            mask = y_test[i]

            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).contiguous().to(device) / 255.0

            output = model(img_tensor)
            output_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            output_mask_binary = (output_mask > threshold).astype(np.uint8) * 255

            y_t = (mask > 0).astype(np.float32)
            y_p = (output_mask > threshold).astype(np.float32)
            dice_val = calculate_dice_coefficient(y_t, y_p)
            iou_val = calculate_iou(y_t, y_p)

            plt.figure(figsize=(10, 3.5))
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title(f'Imagem Teste {i+1}')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(mask > 0, cmap='gray')
            plt.title('Máscara Real (Ground Truth)')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(output_mask_binary, cmap='gray')
            plt.title(f'Predição (Dice: {dice_val:.3f}, IoU: {iou_val:.3f})')
            plt.axis('off')

            plt.tight_layout()
            plt.show()


def plot_naive_instance_extraction(model, X_test, y_test, device, num_samples=3, threshold=0.5):
    """Exibe a extração de instâncias usando o Método Ingênuo vs Ground Truth Real."""
    model.eval()

    with torch.no_grad():
        for i in range(min(num_samples, len(X_test))):
            img = X_test[i]
            gt_labeled = y_test[i]

            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).contiguous().to(device) / 255.0
            output = model(img_tensor)
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()

            labeled_mask, num_instances = extract_instances_naive(pred_prob, threshold=threshold)
            num_gt_instances = len(np.unique(gt_labeled[gt_labeled > 0]))

            plt.figure(figsize=(12, 3.5))
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title(f'Imagem {i+1}')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(gt_labeled, cmap='nipy_spectral')
            plt.title(f'GT Instâncias ({num_gt_instances} objetos)')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(labeled_mask, cmap='nipy_spectral')
            plt.title(f'Instâncias Ingênuas ({num_instances} detectadas)')
            plt.axis('off')

            plt.tight_layout()
            plt.show()


def plot_quantify_failure(results, dataset_name="Reais"):
    """Item 5: Plota gráficos de mAP e Erro de Contagem vs. Densidade de Objetos para quantificar o fracasso."""
    num_gt = results["num_gt_list"]
    mAPs = results["mAP_list"]
    count_errors = results["count_error_list"]
    method = results["matching_method"].capitalize()

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(num_gt, mAPs, alpha=0.6, color='crimson', edgecolors='k', label='Amostras')

    if len(num_gt) > 1 and len(np.unique(num_gt)) > 1:
        degree = 2 if len(np.unique(num_gt)) > 2 else 1
        z = np.polyfit(num_gt, mAPs, degree)
        p = np.poly1d(z)
        x_trend = np.linspace(num_gt.min(), num_gt.max(), 100)
        plt.plot(x_trend, p(x_trend), "b--", linewidth=2, label="Tendência")

    plt.title(f'Quantificação do Fracasso: mAP vs Densidade ({dataset_name} - {method})')
    plt.xlabel('Densidade de Objetos na Imagem (Num GT)')
    plt.ylabel('mAP@[.50:.95]')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(num_gt, count_errors, alpha=0.6, color='darkorange', edgecolors='k', label='Amostras')

    if len(num_gt) > 1 and len(np.unique(num_gt)) > 1:
        z_err = np.polyfit(num_gt, count_errors, 1)
        p_err = np.poly1d(z_err)
        x_trend = np.linspace(num_gt.min(), num_gt.max(), 100)
        plt.plot(x_trend, p_err(x_trend), "r--", linewidth=2, label="Tendência")

    plt.title(f'Erro de Contagem vs Densidade ({dataset_name} - {method})')
    plt.xlabel('Densidade de Objetos na Imagem (Num GT)')
    plt.ylabel('Erro Absoluto de Contagem (|Pred - GT|)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_trilha_a_samples(X, y_3c, num_samples=3):
    """Exibe amostras das imagens originais e dos alvos de 3 classes gerados."""
    plt.figure(figsize=(4 * num_samples, 4))
    for i in range(min(num_samples, len(X))):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(X[i])
        plt.title(f'Imagem {i+1}')
        plt.axis('off')

        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(y_3c[i], cmap='viridis', vmin=0, vmax=2)
        plt.title('Alvo 3C (0:Bg, 1:Core, 2:Bord)')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_trilha_a_predictions(model, X_test, y_test_gt, device, num_samples=3,
                             threshold_interior=0.35, threshold_fg=0.35, min_marker_size=1):
    """Visualiza as predições da Trilha A: Imagem, GT, Probabilidade Interior, Probabilidade Fronteira e Watershed."""
    model.eval()

    with torch.no_grad():
        for i in range(min(num_samples, len(X_test))):
            img = X_test[i]
            gt_labeled = y_test_gt[i]

            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).contiguous().to(device) / 255.0
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            labeled_mask, num_instances = decodificar_watershed_trilha_a(
                probs, threshold_interior=threshold_interior, threshold_fg=threshold_fg, min_marker_size=min_marker_size
            )
            num_gt_instances = len(np.unique(gt_labeled[gt_labeled > 0]))

            plt.figure(figsize=(16, 3.5))
            plt.subplot(1, 5, 1)
            plt.imshow(img)
            plt.title(f'Imagem {i+1}')
            plt.axis('off')

            plt.subplot(1, 5, 2)
            plt.imshow(gt_labeled, cmap='nipy_spectral')
            plt.title(f'GT ({num_gt_instances} objetos)')
            plt.axis('off')

            plt.subplot(1, 5, 3)
            plt.imshow(probs[1], cmap='hot', vmin=0, vmax=1)
            plt.title('Prob. Interior (Seeds)')
            plt.axis('off')

            plt.subplot(1, 5, 4)
            plt.imshow(probs[2], cmap='viridis', vmin=0, vmax=1)
            plt.title('Prob. Fronteira')
            plt.axis('off')

            plt.subplot(1, 5, 5)
            plt.imshow(labeled_mask, cmap='nipy_spectral')
            plt.title(f'Watershed ({num_instances} det.)')
            plt.axis('off')

            plt.tight_layout()
            plt.show()


def plot_objeto_fronteira_tiles(mosaic_img, mosaic_gt, pred_sem_fusao, pred_com_fusao,
                                tiles_data, tile_size=128, margin=20, corrigido=True):
    """
    Parte 4 (Item 3): Mostra detalhadamente o que acontece com um objeto que cai na fronteira entre dois tiles.
    Exibe a imagem geral com os contornos dos tiles, e um zoom na fronteira comparando:
    Ground Truth, Predição no Tile A, Predição no Tile B, Mosaico Sem Fusão (cortado) e Mosaico Com Fusão (unificado).
    """
    H, W = mosaic_gt.shape

    unique_gt = np.unique(mosaic_gt[mosaic_gt > 0])
    best_uid = None
    max_split = 1
    for uid in unique_gt:
        mask_u = (mosaic_gt == uid)
        preds_in_u = np.unique(pred_sem_fusao[mask_u])
        preds_in_u = preds_in_u[preds_in_u > 0]
        if len(preds_in_u) > max_split:
            max_split = len(preds_in_u)
            best_uid = uid

    if best_uid is None:
        min_dist_border = 1e9
        for uid in unique_gt:
            coords = np.argwhere(mosaic_gt == uid)
            cy, cx = coords.mean(axis=0)
            for t in tiles_data:
                d = min(abs(cy - t['y']), abs(cy - (t['y'] + tile_size)),
                        abs(cx - t['x']), abs(cx - (t['x'] + tile_size)))
                if d < min_dist_border:
                    min_dist_border = d
                    best_uid = uid

    coords = np.argwhere(mosaic_gt == best_uid)
    y_min_roi = max(0, coords[:, 0].min() - margin)
    y_max_roi = min(H, coords[:, 0].max() + margin + 1)
    x_min_roi = max(0, coords[:, 1].min() - margin)
    x_max_roi = min(W, coords[:, 1].max() + margin + 1)

    cy, cx = coords.mean(axis=0)

    touching_tiles = []
    for t in tiles_data:
        if t['y'] <= cy <= t['y'] + tile_size and t['x'] <= cx <= t['x'] + tile_size:
            touching_tiles.append(t)

    tA = touching_tiles[0] if len(touching_tiles) > 0 else tiles_data[0]
    tB = touching_tiles[1] if len(touching_tiles) > 1 else (tiles_data[1] if len(tiles_data) > 1 else tiles_data[0])

    def get_tile_crop(t):
        full_tile_mask = np.zeros((H, W), dtype=np.int32)
        full_tile_mask[t['y']:t['y'] + tile_size, t['x']:t['x'] + tile_size] = t['local_mask']
        return full_tile_mask[y_min_roi:y_max_roi, x_min_roi:x_max_roi]

    crop_img = mosaic_img[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    crop_gt = mosaic_gt[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    crop_tA = get_tile_crop(tA)
    crop_tB = get_tile_crop(tB)
    crop_sem = pred_sem_fusao[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    crop_com = pred_com_fusao[y_min_roi:y_max_roi, x_min_roi:x_max_roi]

    ids_sem = np.unique(crop_sem[crop_gt == best_uid])
    ids_sem = ids_sem[ids_sem > 0]
    ids_com = np.unique(crop_com[crop_gt == best_uid])
    ids_com = ids_com[ids_com > 0]

    plt.figure(figsize=(18, 7))

    plt.subplot(2, 3, 1)
    plt.imshow(mosaic_img)
    plt.plot([x_min_roi, x_max_roi, x_max_roi, x_min_roi, x_min_roi],
             [y_min_roi, y_min_roi, y_max_roi, y_max_roi, y_min_roi], 'r-', linewidth=2.5, label='ROI Fronteira')
    for t in tiles_data:
        plt.plot([t['x'], t['x'] + tile_size, t['x'] + tile_size, t['x'], t['x']],
                 [t['y'], t['y'], t['y'] + tile_size, t['y'] + tile_size, t['y']],
                 'w--', alpha=0.4, linewidth=1)
    plt.title(f'Mosaico Completo ({H}x{W}) e Grid de Tiles')
    plt.axis('off')
    plt.legend(loc='lower right')

    plt.subplot(2, 3, 2)
    plt.imshow(crop_img)
    plt.imshow(np.ma.masked_where(crop_gt == 0, crop_gt), cmap='spring', alpha=0.6)
    plt.title(f'Ground Truth (Núcleo #{best_uid}: Unificado)')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(crop_img)
    plt.imshow(np.ma.masked_where(crop_tA == 0, crop_tA), cmap='cool', alpha=0.6)
    plt.title(f'Predição Tile {tA["tile_id"]} (Local)')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(crop_img)
    plt.imshow(np.ma.masked_where(crop_tB == 0, crop_tB), cmap='winter', alpha=0.6)
    plt.title(f'Predição Tile {tB["tile_id"]} (Local)')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(crop_img)
    plt.imshow(np.ma.masked_where(crop_sem == 0, crop_sem), cmap='nipy_spectral', alpha=0.65)
    plt.title(f'Sem Fusão: CORTADO! ({len(ids_sem)} IDs: {ids_sem})')
    plt.axis('off')

    if corrigido:
        plt.subplot(2, 3, 6)
        plt.imshow(crop_img)
        plt.imshow(np.ma.masked_where(crop_com == 0, crop_com), cmap='nipy_spectral', alpha=0.65)
        plt.title(f'Com Fusão: UNIFICADO! (1 ID: {ids_com})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_mosaico_completo(mosaic_img, mosaic_gt, pred_sem_fusao, pred_com_fusao):
    """Exibe visualmente o mosaico completo: Imagem, Ground Truth, Predição Sem Fusão e Predição Com Fusão."""
    num_gt = len(np.unique(mosaic_gt[mosaic_gt > 0]))
    num_sem = len(np.unique(pred_sem_fusao[pred_sem_fusao > 0]))
    num_com = len(np.unique(pred_com_fusao[pred_com_fusao > 0]))

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(mosaic_img)
    plt.title(f'Mosaico de Entrada ({mosaic_img.shape[0]}x{mosaic_img.shape[1]})')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(mosaic_gt, cmap='nipy_spectral')
    plt.title(f'Ground Truth ({num_gt} núcleos)')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(pred_sem_fusao, cmap='nipy_spectral')
    plt.title(f'Sem Fusão ({num_sem} det. - Fraturas)')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(pred_com_fusao, cmap='nipy_spectral')
    plt.title(f'Com Fusão ({num_com} det. - Unificado)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def avaliar_e_plotar_comparativo_instancias(
    res_real,
    res_synth=None,
    model_synth=None,
    X_synth=None,
    y_synth=None,
    device=None,
    is_trilha_a=False,
    threshold_interior=0.35,
    threshold_fg=0.35,
    threshold_baseline=0.5,
    matching_method="hungarian",
    titulo_metodo="Avaliação por Instâncias",
    nome_real="Reais (DSB2018)",
    nome_synth="Sintéticos (Elipses)"
):
    """
    Avalia o modelo no dataset sintético (se não pré-computado) e gera:
    1) Tabela quantitativa comparativa formatada das métricas no console (Reais vs. Sintéticos).
    2) Painel com dois histogramas comparativos do Erro Absoluto de Contagem por Imagem (|N_pred - N_gt|).

    Pode ser reutilizada tanto para a Baseline (Parte 1) quanto para a Trilha A (Parte 2).
    """
    # 1. Se res_synth não foi fornecido pronto, computa usando o avaliador adequado
    if res_synth is None:
        if model_synth is None or X_synth is None or y_synth is None or device is None:
            raise ValueError(
                "Para avaliar o dataset sintético, forneça 'res_synth' ou ('model_synth', 'X_synth', 'y_synth', 'device')."
            )

        if is_trilha_a:
            res_synth = evaluate_instance_level_trilha_a(
                model=model_synth,
                X=X_synth,
                y=y_synth,
                device=device,
                threshold_interior=threshold_interior,
                threshold_fg=threshold_fg,
                matching_method=matching_method,
                dataset_name=nome_synth
            )
        else:
            res_synth = evaluate_instance_level(
                model=model_synth,
                X=X_synth,
                y=y_synth,
                device=device,
                threshold=threshold_baseline,
                matching_method=matching_method,
                dataset_name=nome_synth
            )

    # 2. Exibe tabela comparativa no console
    print(f"\n{'='*75}")
    print(f"   COMPARAÇÃO QUANTITATIVA: REAIS vs. SINTÉTICOS")
    print(f"   Configuração: {titulo_metodo}")
    print(f"{'='*75}")
    print(f"{'Métrica':<35} | {nome_real:<18} | {nome_synth:<18}")
    print(f"{'-'*75}")
    print(f"{'mAP@[.50:.95]':<35} | {res_real['mean_mAP']:<18.4f} | {res_synth['mean_mAP']:<18.4f}")
    print(f"{'Erro Médio Contagem (abs)':<35} | {res_real['mean_count_error']:<18.2f} | {res_synth['mean_count_error']:<18.2f}")

    if 'mean_aps_per_threshold' in res_real and 'mean_aps_per_threshold' in res_synth:
        ap50_r = res_real['mean_aps_per_threshold'][0]
        ap50_s = res_synth['mean_aps_per_threshold'][0]
        print(f"{'AP @ IoU=0.50':<35} | {ap50_r:<18.4f} | {ap50_s:<18.4f}")
        if len(res_real['mean_aps_per_threshold']) > 5:
            ap75_r = res_real['mean_aps_per_threshold'][5]
            ap75_s = res_synth['mean_aps_per_threshold'][5]
            print(f"{'AP @ IoU=0.75':<35} | {ap75_r:<18.4f} | {ap75_s:<18.4f}")

    print(f"{'='*75}\n")

    # 3. Histograma Comparativo do Erro Absoluto de Contagem
    errors_real = res_real["count_error_list"]
    errors_synth = res_synth["count_error_list"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Subplot 1: Dataset Real ---
    max_r = int(np.max(errors_real)) if len(errors_real) > 0 else 10
    bins_r = np.arange(0, max_r + 2) - 0.5
    axes[0].hist(
        errors_real,
        bins=bins_r,
        color="royalblue",
        edgecolor="black",
        alpha=0.8,
        rwidth=0.85
    )
    axes[0].axvline(
        res_real["mean_count_error"],
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f'Média = {res_real["mean_count_error"]:.2f}'
    )
    axes[0].axvline(
        np.median(errors_real),
        color="darkgreen",
        linestyle=":",
        linewidth=2,
        label=f'Mediana = {np.median(errors_real):.1f}'
    )
    axes[0].set_title(
        f"Erro de Contagem — {nome_real}\n({titulo_metodo})",
        fontsize=12,
        fontweight="bold"
    )
    axes[0].set_xlabel("Erro Absoluto (|N_pred - N_gt|)", fontsize=11)
    axes[0].set_ylabel("Frequência (Nº de Imagens)", fontsize=11)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=10)

    # --- Subplot 2: Dataset Sintético ---
    max_s = int(np.max(errors_synth)) if len(errors_synth) > 0 else 10
    bins_s = np.arange(0, max_s + 2) - 0.5
    axes[1].hist(
        errors_synth,
        bins=bins_s,
        color="seagreen",
        edgecolor="black",
        alpha=0.8,
        rwidth=0.85
    )
    axes[1].axvline(
        res_synth["mean_count_error"],
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f'Média = {res_synth["mean_count_error"]:.2f}'
    )
    axes[1].axvline(
        np.median(errors_synth),
        color="darkgreen",
        linestyle=":",
        linewidth=2,
        label=f'Mediana = {np.median(errors_synth):.1f}'
    )
    axes[1].set_title(
        f"Erro de Contagem — {nome_synth}\n({titulo_metodo})",
        fontsize=12,
        fontweight="bold"
    )
    axes[1].set_xlabel("Erro Absoluto (|N_pred - N_gt|)", fontsize=11)
    axes[1].set_ylabel("Frequência (Nº de Imagens)", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.show()


import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

try:
    from .postprocessing import extract_instances_naive, decodificar_watershed_trilha_a
except ImportError:
    from postprocessing import extract_instances_naive, decodificar_watershed_trilha_a


def calculate_dice_coefficient(y_true, y_pred):
    """Calcula o Coeficiente Dice entre duas máscaras binárias."""
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)


def calculate_iou(y_true, y_pred):
    """Calcula a Interseção sobre União (IoU) entre duas máscaras binárias."""
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-6)


def evaluate_model(model, X, y, device, threshold=0.5, batch_size=8):
    """Avalia o modelo em um conjunto de dados calculando Mean Dice Coefficient e Mean IoU semânticos."""
    model.eval()
    dice_list = []
    iou_list = []
    num_samples = X.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))

    with torch.no_grad():
        for b in range(num_batches):
            batch_img = X[b * batch_size:(b + 1) * batch_size]
            batch_mask = y[b * batch_size:(b + 1) * batch_size]

            img_tensor = torch.from_numpy(batch_img).float().permute(0, 3, 1, 2).contiguous().to(device) / 255.0
            outputs = model(img_tensor)
            preds = (torch.sigmoid(outputs) > threshold).cpu().numpy().squeeze(1)

            for i in range(len(batch_img)):
                y_t = (batch_mask[i] > 0).astype(np.float32)
                y_p = preds[i].astype(np.float32)
                dice_list.append(calculate_dice_coefficient(y_t, y_p))
                iou_list.append(calculate_iou(y_t, y_p))

    mean_dice = np.mean(dice_list)
    mean_iou = np.mean(iou_list)
    return mean_dice, mean_iou


def calculate_instance_iou_matrix(gt_labeled, num_gt, pred_labeled, num_pred):
    """Calcula a matriz de IoU entre todas as instâncias do GT e da Predição."""
    iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)
    if num_gt == 0 or num_pred == 0:
        return iou_matrix

    unique_gt = np.unique(gt_labeled[gt_labeled > 0])
    for i_idx, gt_id in enumerate(unique_gt):
        gt_mask = (gt_labeled == gt_id)
        area_gt = gt_mask.sum()
        if area_gt == 0:
            continue
        for j in range(1, num_pred + 1):
            pred_mask = (pred_labeled == j)
            inter = np.logical_and(gt_mask, pred_mask).sum()
            if inter == 0:
                continue
            union = area_gt + pred_mask.sum() - inter
            iou_matrix[i_idx, j - 1] = inter / union if union > 0 else 0.0

    return iou_matrix


def match_instances_greedy(iou_matrix, iou_thresh):
    """Casamento Guloso por IoU Decrescente."""
    num_gt, num_pred = iou_matrix.shape
    matched_gt = set()
    matched_pred = set()

    pairs = []
    for i in range(num_gt):
        for j in range(num_pred):
            if iou_matrix[i, j] >= iou_thresh:
                pairs.append((iou_matrix[i, j], i, j))

    pairs.sort(key=lambda x: x[0], reverse=True)

    tp = 0
    for iou_val, i, j in pairs:
        if i not in matched_gt and j not in matched_pred:
            matched_gt.add(i)
            matched_pred.add(j)
            tp += 1

    return tp


def match_instances_hungarian(iou_matrix, iou_thresh):
    """Casamento Global Otimizado via Algoritmo Húngaro (Munkres)."""
    num_gt, num_pred = iou_matrix.shape
    if num_gt == 0 or num_pred == 0:
        return 0

    cost_matrix = 1.0 - iou_matrix
    gt_ind, pred_ind = linear_sum_assignment(cost_matrix)

    tp = 0
    for i, j in zip(gt_ind, pred_ind):
        if iou_matrix[i, j] >= iou_thresh:
            tp += 1

    return tp


def evaluate_instance_metrics_single(pred_prob, mask_gt, threshold=0.5, iou_thresholds=np.arange(0.50, 1.00, 0.05), matching_method="greedy"):
    """Avalia uma única imagem em nível de instâncias calculando mAP@[.50:.95] e erro de contagem."""
    pred_labeled, num_pred = extract_instances_naive(pred_prob, threshold=threshold)

    gt_labeled = mask_gt.astype(np.int32)
    unique_gt = np.unique(gt_labeled[gt_labeled > 0])
    num_gt = len(unique_gt)

    count_error = abs(num_pred - num_gt)

    if num_gt == 0 and num_pred == 0:
        aps = np.ones(len(iou_thresholds), dtype=np.float32)
        return 1.0, count_error, num_gt, num_pred, aps
    elif num_gt == 0 or num_pred == 0:
        aps = np.zeros(len(iou_thresholds), dtype=np.float32)
        return 0.0, count_error, num_gt, num_pred, aps

    iou_matrix = calculate_instance_iou_matrix(gt_labeled, num_gt, pred_labeled, num_pred)

    aps = []
    for t in iou_thresholds:
        if matching_method == "greedy":
            tp = match_instances_greedy(iou_matrix, t)
        elif matching_method == "hungarian":
            tp = match_instances_hungarian(iou_matrix, t)
        else:
            raise ValueError(f"Método de matching desconhecido: {matching_method}")

        fp = num_pred - tp
        fn = num_gt - tp
        denom = tp + fp + fn
        ap_t = tp / denom if denom > 0 else 0.0
        aps.append(ap_t)

    aps = np.array(aps, dtype=np.float32)
    mAP = float(np.mean(aps))
    return mAP, count_error, num_gt, num_pred, aps


def evaluate_model_instances(model, X, y, device, threshold=0.5, iou_thresholds=np.arange(0.50, 1.00, 0.05), matching_method="greedy"):
    """Avalia o modelo em todo o conjunto de teste em nível de instâncias."""
    model.eval()
    mAP_list = []
    count_error_list = []
    num_gt_list = []
    num_pred_list = []
    aps_matrix = []

    t0 = time.time()
    with torch.no_grad():
        for i in range(len(X)):
            img = X[i]
            mask = y[i]

            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).contiguous().to(device) / 255.0
            output = model(img_tensor)
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()

            mAP, count_err, num_gt, num_pred, aps = evaluate_instance_metrics_single(
                pred_prob, mask, threshold=threshold, iou_thresholds=iou_thresholds, matching_method=matching_method
            )

            mAP_list.append(mAP)
            count_error_list.append(count_err)
            num_gt_list.append(num_gt)
            num_pred_list.append(num_pred)
            aps_matrix.append(aps)

    elapsed_time = time.time() - t0
    mean_aps_per_threshold = np.mean(aps_matrix, axis=0)

    results = {
        "mean_mAP": float(np.mean(mAP_list)),
        "mean_count_error": float(np.mean(count_error_list)),
        "elapsed_time": elapsed_time,
        "mAP_list": np.array(mAP_list),
        "count_error_list": np.array(count_error_list),
        "num_gt_list": np.array(num_gt_list),
        "num_pred_list": np.array(num_pred_list),
        "iou_thresholds": iou_thresholds,
        "mean_aps_per_threshold": mean_aps_per_threshold,
        "matching_method": matching_method,
    }

    return results


def evaluate_instance_level(model, X, y, device, threshold=0.5, matching_method="hungarian", dataset_name="Reais"):
    """Item 3: Avalia o AP para cada limiar de IoU (0.50 a 0.95), o mAP@[.50:.95] e o erro de contagem."""
    iou_thresholds = np.arange(0.50, 1.00, 0.05)
    results = evaluate_model_instances(
        model, X, y, device, threshold=threshold, iou_thresholds=iou_thresholds, matching_method=matching_method
    )

    print(f"\n--- [Item 3] Avaliação por Instâncias ({dataset_name}) ---")
    print("Precisão Média (AP) para cada Limiar de IoU (0.50 a 0.95, passo 0.05):")
    for idx, t in enumerate(iou_thresholds):
        ap_t = results['mean_aps_per_threshold'][idx]
        print(f"  IoU = {t:.2f} : AP = {ap_t:.4f}")

    print(f"\n--> mAP@[.50:.95] Final: {results['mean_mAP']:.4f}")
    print(f"--> Erro Absoluto Médio de Contagem por Imagem: {results['mean_count_error']:.2f} objetos")

    return results


def compare_matching_methods(model, X, y, device, threshold=0.5, dataset_name="Reais"):
    """Item 4: Documenta e compara a regra de Matching Guloso vs. Algoritmo Húngaro (Munkres)."""
    iou_thresholds = np.arange(0.50, 1.00, 0.05)
    res_greedy = evaluate_model_instances(model, X, y, device, threshold=threshold, iou_thresholds=iou_thresholds, matching_method="greedy")
    res_hungarian = evaluate_model_instances(model, X, y, device, threshold=threshold, iou_thresholds=iou_thresholds, matching_method="hungarian")

    print(f"\n--- [Item 4] Comparação da Regra de Matching ({dataset_name}) ---")
    print(f"Guloso (Greedy):    mAP@[.50:.95] = {res_greedy['mean_mAP']:.4f} | Erro Médio Contagem = {res_greedy['mean_count_error']:.2f}")
    print(f"Húngaro (Hungarian): mAP@[.50:.95] = {res_hungarian['mean_mAP']:.4f} | Erro Médio Contagem = {res_hungarian['mean_count_error']:.2f}")
    print("Regra de matching documentada e explicitada: Algoritmo Húngaro (Hungarian)")

    return res_greedy, res_hungarian


def evaluate_instance_metrics_single_trilha_a(pred_probs, mask_gt_instances,
                                             threshold_interior=0.35, threshold_fg=0.35,
                                             min_marker_size=1,
                                             iou_thresholds=np.arange(0.50, 1.00, 0.05),
                                             matching_method="hungarian"):
    """Avalia uma única imagem da Trilha A calculando mAP@[.50:.95] e erro de contagem via Watershed."""
    pred_labeled, num_pred = decodificar_watershed_trilha_a(
        pred_probs, threshold_interior=threshold_interior, threshold_fg=threshold_fg, min_marker_size=min_marker_size
    )

    gt_labeled = mask_gt_instances.astype(np.int32)
    unique_gt = np.unique(gt_labeled[gt_labeled > 0])
    num_gt = len(unique_gt)

    count_error = abs(num_pred - num_gt)

    if num_gt == 0 and num_pred == 0:
        aps = np.ones(len(iou_thresholds), dtype=np.float32)
        return 1.0, count_error, num_gt, num_pred, aps
    elif num_gt == 0 or num_pred == 0:
        aps = np.zeros(len(iou_thresholds), dtype=np.float32)
        return 0.0, count_error, num_gt, num_pred, aps

    iou_matrix = calculate_instance_iou_matrix(gt_labeled, num_gt, pred_labeled, num_pred)

    aps = []
    for t in iou_thresholds:
        if matching_method == "greedy":
            tp = match_instances_greedy(iou_matrix, t)
        elif matching_method == "hungarian":
            tp = match_instances_hungarian(iou_matrix, t)
        else:
            raise ValueError(f"Método de matching desconhecido: {matching_method}")

        fp = num_pred - tp
        fn = num_gt - tp
        denom = tp + fp + fn
        ap_t = tp / denom if denom > 0 else 0.0
        aps.append(ap_t)

    aps = np.array(aps, dtype=np.float32)
    mAP = float(np.mean(aps))
    return mAP, count_error, num_gt, num_pred, aps


def evaluate_model_instances_trilha_a(model, X, y_gt_instances, device,
                                     threshold_interior=0.35, threshold_fg=0.35,
                                     min_marker_size=1,
                                     iou_thresholds=np.arange(0.50, 1.00, 0.05),
                                     matching_method="hungarian"):
    """Avalia o modelo da Trilha A em todo o conjunto de teste em nível de instâncias."""
    model.eval()
    mAP_list = []
    count_error_list = []
    num_gt_list = []
    num_pred_list = []
    aps_matrix = []

    t0 = time.time()
    with torch.no_grad():
        for i in range(len(X)):
            img = X[i]
            mask_gt = y_gt_instances[i]

            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).contiguous().to(device) / 255.0
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            mAP, count_err, num_gt, num_pred, aps = evaluate_instance_metrics_single_trilha_a(
                probs, mask_gt, threshold_interior=threshold_interior, threshold_fg=threshold_fg,
                min_marker_size=min_marker_size, iou_thresholds=iou_thresholds, matching_method=matching_method
            )

            mAP_list.append(mAP)
            count_error_list.append(count_err)
            num_gt_list.append(num_gt)
            num_pred_list.append(num_pred)
            aps_matrix.append(aps)

    elapsed_time = time.time() - t0
    mean_aps_per_threshold = np.mean(aps_matrix, axis=0)

    results = {
        "mean_mAP": float(np.mean(mAP_list)),
        "mean_count_error": float(np.mean(count_error_list)),
        "elapsed_time": elapsed_time,
        "mAP_list": np.array(mAP_list),
        "count_error_list": np.array(count_error_list),
        "num_gt_list": np.array(num_gt_list),
        "num_pred_list": np.array(num_pred_list),
        "iou_thresholds": iou_thresholds,
        "mean_aps_per_threshold": mean_aps_per_threshold,
        "matching_method": matching_method,
    }

    return results


def evaluate_instance_level_trilha_a(model, X, y, device, threshold_interior=0.35, threshold_fg=0.35,
                                    min_marker_size=1, matching_method="hungarian", dataset_name="Reais (DSB2018)"):
    """Parte 2: Avalia o AP para cada limiar de IoU (0.50 a 0.95), o mAP@[.50:.95] e o erro de contagem via Watershed."""
    iou_thresholds = np.arange(0.50, 1.00, 0.05)
    results = evaluate_model_instances_trilha_a(
        model, X, y, device, threshold_interior=threshold_interior, threshold_fg=threshold_fg,
        min_marker_size=min_marker_size, iou_thresholds=iou_thresholds, matching_method=matching_method
    )

    print(f"\n--- [Parte 2 - Trilha A] Avaliação por Instâncias ({dataset_name}) ---")
    print("Precisão Média (AP) para cada Limiar de IoU (0.50 a 0.95, passo 0.05):")
    for idx, t in enumerate(iou_thresholds):
        ap_t = results['mean_aps_per_threshold'][idx]
        print(f"  IoU = {t:.2f} : AP = {ap_t:.4f}")

    print(f"\n--> mAP@[.50:.95] Trilha A: {results['mean_mAP']:.4f}")
    print(f"--> Erro Absoluto Médio de Contagem por Imagem: {results['mean_count_error']:.2f} objetos")

    return results


def compare_baseline_vs_trilha_a(results_baseline, results_trilha_a, dataset_name="Reais (DSB2018)"):
    """Compara lado a lado a Baseline da Parte 1 e a Trilha A da Parte 2."""
    import matplotlib.pyplot as plt

    print(f"\n========================================================")
    print(f"   COMPARAÇÃO: BASELINE (Parte 1) vs TRILHA A (Parte 2)   ")
    print(f"                   Dataset: {dataset_name}                ")
    print(f"========================================================")
    print(f"{'Métrica':<35} | {'Baseline (Ingênua)':<20} | {'Trilha A (Watershed)':<20}")
    print("-" * 80)
    print(f"{'mAP@[.50:.95]':<35} | {results_baseline['mean_mAP']:<20.4f} | {results_trilha_a['mean_mAP']:<20.4f}")
    print(f"{'Erro Médio de Contagem (abs)':<35} | {results_baseline['mean_count_error']:<20.2f} | {results_trilha_a['mean_count_error']:<20.2f}")
    print(f"{'AP @ IoU=0.50':<35} | {results_baseline['mean_aps_per_threshold'][0]:<20.4f} | {results_trilha_a['mean_aps_per_threshold'][0]:<20.4f}")
    print(f"{'AP @ IoU=0.75':<35} | {results_baseline['mean_aps_per_threshold'][5]:<20.4f} | {results_trilha_a['mean_aps_per_threshold'][5]:<20.4f}")
    print("=" * 80)

    num_gt = results_baseline["num_gt_list"]
    mAPs_b = results_baseline["mAP_list"]
    mAPs_a = results_trilha_a["mAP_list"]
    err_b = results_baseline["count_error_list"]
    err_a = results_trilha_a["count_error_list"]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(num_gt, mAPs_b, alpha=0.4, color='crimson', label='Baseline (Amostras)')
    plt.scatter(num_gt, mAPs_a, alpha=0.4, color='teal', label='Trilha A (Amostras)')

    if len(num_gt) > 1 and len(np.unique(num_gt)) > 1:
        z_b = np.polyfit(num_gt, mAPs_b, 1)
        p_b = np.poly1d(z_b)
        z_a = np.polyfit(num_gt, mAPs_a, 1)
        p_a = np.poly1d(z_a)
        x_trend = np.linspace(num_gt.min(), num_gt.max(), 100)
        plt.plot(x_trend, p_b(x_trend), "r--", linewidth=2.5, label='Tendência Baseline')
        plt.plot(x_trend, p_a(x_trend), "b-", linewidth=2.5, label='Tendência Trilha A')

    plt.title(f'Comparação: mAP@[.50:.95] vs Densidade')
    plt.xlabel('Densidade de Objetos na Imagem (Num GT)')
    plt.ylabel('mAP@[.50:.95]')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(num_gt, err_b, alpha=0.4, color='darkorange', label='Baseline (Amostras)')
    plt.scatter(num_gt, err_a, alpha=0.4, color='dodgerblue', label='Trilha A (Amostras)')

    if len(num_gt) > 1 and len(np.unique(num_gt)) > 1:
        z_err_b = np.polyfit(num_gt, err_b, 1)
        p_err_b = np.poly1d(z_err_b)
        z_err_a = np.polyfit(num_gt, err_a, 1)
        p_err_a = np.poly1d(z_err_a)
        x_trend = np.linspace(num_gt.min(), num_gt.max(), 100)
        plt.plot(x_trend, p_err_b(x_trend), "r--", linewidth=2.5, label='Tendência Baseline')
        plt.plot(x_trend, p_err_a(x_trend), "b-", linewidth=2.5, label='Tendência Trilha A')

    plt.title(f'Comparação: Erro de Contagem vs Densidade')
    plt.xlabel('Densidade de Objetos na Imagem (Num GT)')
    plt.ylabel('Erro Absoluto de Contagem (|Pred - GT|)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()


def imprimir_tabela_ablação(resultados_dict, titulo="Ablações"):
    """Exibe os resultados da ablação em formato tabular com média e desvio padrão."""
    print(f"\n{'='*75}")
    print(f"   TABELA DE ABLAÇÃO: {titulo.upper()} (2 SEEDS: MÉDIA ± DESVIO)")
    print(f"{'='*75}")
    print(f"{'Configuração':<35} | {'mAP@[.50:.95]':<18} | {'Erro Médio Contagem':<18}")
    print("-" * 75)
    for nome, r in resultados_dict.items():
        map_str = f"{r['mAP_mean']:.4f} ± {r['mAP_std']:.4f}"
        err_str = f"{r['err_mean']:.2f} ± {r['err_std']:.2f}"
        print(f"{nome:<35} | {map_str:<18} | {err_str:<18}")
    print("=" * 75)

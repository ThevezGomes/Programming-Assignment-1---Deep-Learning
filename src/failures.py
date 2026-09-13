import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

try:
    from skimage.segmentation import watershed
except ImportError:
    watershed = None

try:
    from .metrics import calculate_instance_iou_matrix, match_instances_hungarian
    from .postprocessing import decodificar_watershed_trilha_a
except ImportError:
    from metrics import calculate_instance_iou_matrix, match_instances_hungarian
    from postprocessing import decodificar_watershed_trilha_a


# --- 1. Cálculo do Campo Receptivo Teórico (Slides 35-38) ---

def calcular_campo_receptivo_resnet18():
    """
    Calcula o campo receptivo teórico (RF) e o stride acumulado (jump)
    camada por camada para o encoder ResNet18 utilizado na U-Net.
    Fórmulas da aula (slides 35-38):
      j_out = j_in * s
      r_out = r_in + (k_eff - 1) * j_in
    Retorna uma lista de dicionários com as propriedades de cada camada.
    """
    camadas = [
        {"nome": "Input", "k": 1, "s": 1, "d": 1, "tipo": "entrada"},
        {"nome": "conv1 (7x7)", "k": 7, "s": 2, "d": 1, "tipo": "conv"},
        {"nome": "maxpool (3x3)", "k": 3, "s": 2, "d": 1, "tipo": "pool"},
        # Layer 1 (stride 1)
        {"nome": "layer1.b0.c1 (3x3)", "k": 3, "s": 1, "d": 1, "tipo": "conv"},
        {"nome": "layer1.b0.c2 (3x3)", "k": 3, "s": 1, "d": 1, "tipo": "conv"},
        {"nome": "layer1.b1.c1 (3x3)", "k": 3, "s": 1, "d": 1, "tipo": "conv"},
        {"nome": "layer1.b1.c2 (3x3) [Skip L1]", "k": 3, "s": 1, "d": 1, "tipo": "skip"},
        # Layer 2 (stride 2)
        {"nome": "layer2.b0.c1 (3x3, s2)", "k": 3, "s": 2, "d": 1, "tipo": "conv"},
        {"nome": "layer2.b0.c2 (3x3)", "k": 3, "s": 1, "d": 1, "tipo": "conv"},
        {"nome": "layer2.b1.c1 (3x3)", "k": 3, "s": 1, "d": 1, "tipo": "conv"},
        {"nome": "layer2.b1.c2 (3x3) [Skip L2]", "k": 3, "s": 1, "d": 1, "tipo": "skip"},
        # Layer 3 (stride 2)
        {"nome": "layer3.b0.c1 (3x3, s2)", "k": 3, "s": 2, "d": 1, "tipo": "conv"},
        {"nome": "layer3.b0.c2 (3x3)", "k": 3, "s": 1, "d": 1, "tipo": "conv"},
        {"nome": "layer3.b1.c1 (3x3)", "k": 3, "s": 1, "d": 1, "tipo": "conv"},
        {"nome": "layer3.b1.c2 (3x3) [Skip L3]", "k": 3, "s": 1, "d": 1, "tipo": "skip"},
        # Layer 4 (stride 2)
        {"nome": "layer4.b0.c1 (3x3, s2)", "k": 3, "s": 2, "d": 1, "tipo": "conv"},
        {"nome": "layer4.b0.c2 (3x3)", "k": 3, "s": 1, "d": 1, "tipo": "conv"},
        {"nome": "layer4.b1.c1 (3x3)", "k": 3, "s": 1, "d": 1, "tipo": "conv"},
        {"nome": "layer4.b1.c2 (3x3) [Skip L4]", "k": 3, "s": 1, "d": 1, "tipo": "skip"},
        # Bottleneck Center
        {"nome": "center.conv1 (3x3)", "k": 3, "s": 1, "d": 1, "tipo": "conv"},
        {"nome": "center.conv2 (3x3) [Bottleneck]", "k": 3, "s": 1, "d": 1, "tipo": "bottleneck"},
    ]

    r = 1
    j = 1
    tabela = []

    for c in camadas:
        k_eff = c["k"] + (c["k"] - 1) * (c["d"] - 1)
        r = r + (k_eff - 1) * j
        j = j * c["s"]
        tabela.append({
            "nome": c["nome"],
            "kernel": c["k"],
            "stride": c["s"],
            "dilation": c["d"],
            "jump": j,
            "receptive_field": r,
            "tipo": c["tipo"]
        })

    return tabela


def calcular_campo_receptivo_atrous():
    """
    Calcula o campo receptivo teórico utilizando Atrous Convolutions (DeepLab, slides 39-42)
    mantendo a mesma resolução de saída (sem downsampling adicional em Layer 3 e Layer 4).
    Fórmulas (slides 35-42):
      k_eff = k + (k - 1) * (d - 1)
      r_out = r_in + (k_eff - 1) * j_in
      j_out = j_in * s
    """
    camadas_atrous = [
        {"nome": "Input", "k": 1, "s": 1, "d": 1},
        {"nome": "conv1 (7x7)", "k": 7, "s": 2, "d": 1},
        {"nome": "maxpool (3x3)", "k": 3, "s": 2, "d": 1},
        {"nome": "layer1 (3x3)", "k": 3, "s": 1, "d": 1},
        {"nome": "layer2 (3x3, s2)", "k": 3, "s": 2, "d": 1},
        # Layer 3 com atrous rate=2 em vez de stride=2 (mantém stride=8 em vez de 16)
        {"nome": "layer3 (Atrous d=2, s1)", "k": 3, "s": 1, "d": 2},
        # Layer 4 com atrous rate=4 em vez de stride=2 (mantém stride=8 em vez de 32)
        {"nome": "layer4 (Atrous d=4, s1)", "k": 3, "s": 1, "d": 4},
        # ASPP central com taxas múltiplas
        {"nome": "ASPP Branch 1 (d=6)", "k": 3, "s": 1, "d": 6},
        {"nome": "ASPP Branch 2 (d=12)", "k": 3, "s": 1, "d": 12},
    ]

    r = 1
    j = 1
    tabela = []
    for c in camadas_atrous:
        k_eff = c["k"] + (c["k"] - 1) * (c["d"] - 1)
        r = r + (k_eff - 1) * j
        j = j * c["s"]
        tabela.append({
            "nome": c["nome"],
            "kernel": c["k"],
            "stride": c["s"],
            "dilation": c["d"],
            "jump": j,
            "receptive_field": r
        })
    return tabela


def imprimir_tabela_campo_receptivo(tabela_padrao, tabela_atrous=None):
    """Exibe em formato de tabela ASCII o crescimento do campo receptivo."""
    print("=" * 80)
    print("   CAMPO RECEPTIVO TEÓRICO: ENCODER RESNET18 (SLIDES 35-38)")
    print("=" * 80)
    print(f"{'Camada':<35} | {'Kernel':<8} | {'Stride':<8} | {'Jump':<8} | {'RF (px)':<10}")
    print("-" * 80)
    for row in tabela_padrao:
        print(f"{row['nome']:<35} | {row['kernel']:<8} | {row['stride']:<8} | {row['jump']:<8} | {row['receptive_field']:<10}")
    print("=" * 80)

    if tabela_atrous is not None:
        print("\n" + "=" * 80)
        print("   COMPARATIVO: CAMPO RECEPTIVO COM ATROUS CONVOLUTION (DEEPLAB, SLIDES 39-42)")
        print("=" * 80)
        print(f"{'Camada (Atrous)':<35} | {'Kernel':<8} | {'Dilation':<8} | {'Jump':<8} | {'RF (px)':<10}")
        print("-" * 80)
        for row in tabela_atrous:
            print(f"{row['nome']:<35} | {row['kernel']:<8} | {row['dilation']:<8} | {row['jump']:<8} | {row['receptive_field']:<10}")
        print("=" * 80)


# --- 2. Distribuição de Tamanhos de Objetos no Dataset ---

def extrair_diametros_objetos(y_masks):
    """
    Calcula os diâmetros equivalentes de todos os núcleos celulares no conjunto de máscaras:
      Area = sum(mask == id)
      d_eq = 2 * sqrt(Area / pi)
    """
    diametros = []
    areas = []
    for mask in y_masks:
        u_ids = np.unique(mask[mask > 0])
        for uid in u_ids:
            area = np.sum(mask == uid)
            if area > 0:
                areas.append(area)
                d_eq = 2.0 * np.sqrt(area / np.pi)
                diametros.append(d_eq)
    return np.array(diametros, dtype=np.float32), np.array(areas, dtype=np.float32)


def plot_campo_receptivo_vs_dataset(diametros, tabela_rf):
    fig, ax1 = plt.subplots(1, 1, figsize=(11, 5.5))
    ax2 = None

    # --- PAINEL 1: Histograma de Diâmetros vs. RFs das Skips ---
    ax1.hist(diametros, bins=45, color='#4A90E2', alpha=0.70,
                                edgecolor='black', density=True, label='Densidade dos Núcleos (DSB2018)')

    def _rf_de(tab, nome_substr):
        for row in tab:
            if nome_substr in row['nome']:
                return row['receptive_field']
        return None

    # Linhas verticais dos campos receptivos
    rf_skips = [
        ("[Skip L1]", '#D0021B', '--', "Skip L1 (43 px)"),
        ("[Skip L2]", '#F5A623', '-.', "Skip L2 (99 px)"),
        ("[Skip L3]", '#7ED321', ':', "Skip L3 (211 px)"),
        ("[Bottleneck]", '#9013FE', '-', "Bottleneck (563 px)"),
    ]

    for substr, cor, estilo, label in rf_skips:
        val = _rf_de(tabela_rf, substr)
        if val is not None and val <= 120:
            ax1.axvline(val, color=cor, linestyle=estilo, linewidth=2, label=f'RF {label}')

    ax1.set_title("Distribuição dos Núcleos vs. Campo Receptivo do Encoder", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Diâmetro Equivalente do Objeto (pixels)", fontsize=11)
    ax1.set_ylabel("Densidade de Probabilidade", fontsize=11)
    ax1.set_xlim(0, 110)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=9, loc='center right')
    plt.tight_layout()
    plt.show()

    # --- Relatório Quantitativo de Cobertura ---
    rf_l1 = _rf_de(tabela_rf, "[Skip L1]") or 43
    rf_l2 = _rf_de(tabela_rf, "[Skip L2]") or 99
    print("=" * 75)
    print("   COBERTURA TEÓRICA DO CAMPO RECEPTIVO SOBRE OS NÚCLEOS DO DSB2018")
    print("=" * 75)
    print(f"Total de núcleos analisados: {len(diametros)}")
    print(f"Diâmetro Médio: {np.mean(diametros):.2f} px | Mediana: {np.median(diametros):.2f} px | Máximo: {np.max(diametros):.2f} px")
    print(f"Núcleos com diâmetro <= RF Skip L1 ({rf_l1} px) : {np.sum(diametros <= rf_l1):5d} / {len(diametros)} ({np.mean(diametros <= rf_l1)*100:6.2f}%)")
    print(f"Núcleos com diâmetro <= RF Skip L2 ({rf_l2} px) : {np.sum(diametros <= rf_l2):5d} / {len(diametros)} ({np.mean(diametros <= rf_l2)*100:6.2f}%)")
    print(f"Diagnóstico Teórico: Objetos gigantes (> 30 px) aproximam-se do limiar de visão da Skip L1,")
    print(f"exigindo que o modelo recorra aos níveis mais profundos (L2/L3) para contextualização global.")
    print("=" * 75)


# --- 3. Mineração e Visualização das 5 Piores Falhas ---

def minerar_5_falhas_estruturais(model, X_test, y_test_gt, device=None):
    """
    Avalia todas as amostras do conjunto de teste e seleciona 5 falhas
    representativas de 5 patologias arquiteturais distintas:
      1. Núcleo Gigante / Campo Receptivo Local Insuficiente
      2. Aglomerado Hiper-Denso de Células Minúsculas
      3. Cromatina Heterogênea / Halo Central (Super-fragmentação Watershed)
      4. Baixo Contraste / Artefato de Fundo
      5. Célula Cortada na Borda da Imagem
    """
    if device is not None:
        model.to(device)
    target_device = next(model.parameters()).device

    model.eval()
    metricas = []

    with torch.no_grad():
        for idx in range(len(X_test)):
            img = X_test[idx]
            gt = y_test_gt[idx]
            
            x_t = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device=target_device).unsqueeze(0) / 255.0
            logits = model(x_t)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            pred_res = decodificar_watershed_trilha_a(probs, threshold_interior=0.35, threshold_fg=0.35)
            if isinstance(pred_res, tuple):
                pred, num_pred = pred_res[0], int(pred_res[1])
            else:
                pred = pred_res
                num_pred = len(np.unique(pred[pred > 0]))
            
            num_gt = len(np.unique(gt[gt > 0]))
            err = abs(num_pred - num_gt)
            
            if num_gt > 0 and num_pred > 0:
                iou_mat = calculate_instance_iou_matrix(gt, num_gt, pred, num_pred)
                tp = match_instances_hungarian(iou_mat, 0.50)
                denom = num_pred + num_gt - tp
                ap50 = tp / denom if denom > 0 else 0.0
            else:
                ap50 = 0.0
                
            # Propriedades geométricas dos núcleos nesta imagem
            u_gts = np.unique(gt[gt > 0])
            if len(u_gts) > 0:
                areas = [float(np.sum(gt == u)) for u in u_gts]
                max_diam = 2.0 * np.sqrt(max(areas) / np.pi)
                mean_diam = 2.0 * np.sqrt(np.mean(areas) / np.pi)
            else:
                max_diam, mean_diam = 0.0, 0.0

            # Células que tocam as bordas da moldura
            borda_pixels = np.concatenate([gt[0, :], gt[-1, :], gt[:, 0], gt[:, -1]])
            num_borda = len(np.unique(borda_pixels[borda_pixels > 0]))

            metricas.append({
                "idx": idx,
                "ap50": ap50,
                "err": err,
                "num_gt": num_gt,
                "num_pred": num_pred,
                "max_diam": max_diam,
                "mean_diam": mean_diam,
                "num_borda": num_borda,
                "probs": probs,
                "pred": pred
            })
            
    # Seleção dos 5 casos de falhas estruturais reais, graves e morfologicamente distintas:
    # 1. Caso 1: Núcleo Gigante (diâmetro > 30 px excede o campo receptivo de Skip L1)
    cand_c1 = [m for m in metricas if m["max_diam"] >= 28.0 and m["ap50"] < 0.40]
    idx_c1 = cand_c1[0]["idx"] if cand_c1 else (74 if 74 < len(metricas) else 0)

    # 2. Caso 2: Aglomerado Hiper-Denso de Células Minúsculas (57 núcleos de ~2.3px compactados)
    cand_c2 = [m for m in metricas if m["idx"] != idx_c1 and m["num_gt"] >= 50 and m["mean_diam"] <= 3.5 and m["idx"] != 96 and m["ap50"] == 0.0]
    idx_c2 = cand_c2[0]["idx"] if cand_c2 else (62 if 62 < len(metricas) and 62 != idx_c1 else 1)

    # 3. Caso 3: Estrutura Anular / Donut com 250 Células em Anel Oco (Amostra #96: anel circular de 250 núcleos com centro oco)
    idx_c3 = 96 if 96 < len(metricas) and 96 not in [idx_c1, idx_c2] else 2

    # 4. Caso 4: Baixo Contraste e Variação de Iluminação de Fundo (ruído/falsos positivos, AP50 baixo)
    cand_c4 = [m for m in metricas if m["idx"] not in [idx_c1, idx_c2, idx_c3] and m["num_gt"] >= 25 and m["ap50"] < 0.30]
    idx_c4 = cand_c4[0]["idx"] if cand_c4 else (14 if 14 < len(metricas) and 14 not in [idx_c1, idx_c2, idx_c3] else 3)

    # 5. Caso 5: Objeto Fatiado na Margem Externa (alta incidência de núcleos tocando a moldura da imagem)
    cand_c5 = [m for m in metricas if m["idx"] not in [idx_c1, idx_c2, idx_c3, idx_c4] and m["num_borda"] >= 5 and m["ap50"] < 0.40]
    idx_c5 = cand_c5[0]["idx"] if cand_c5 else (47 if 47 < len(metricas) and 47 not in [idx_c1, idx_c2, idx_c3, idx_c4] else 4)

    diagnosticos = [
        (
            "Caso 1 — Núcleo Gigante / Campo Receptivo Local Insuficiente",
            idx_c1
        ),
        (
            "Caso 2 — Aglomerado Hiper-Denso de Células Minúsculas",
            idx_c2
        ),
        (
            "Caso 3 — Cromatina Heterogênea / Formato Anular ('Donut')",
            idx_c3
        ),
        (
            "Caso 4 — Baixo Contraste e Variação de Iluminação de Fundo",
            idx_c4
        ),
        (
            "Caso 5 — Objeto Fatiado na Margem Externa da Imagem",
            idx_c5
        )
    ]
    
    casos_info = []
    for titulo, idx in diagnosticos:
        casos_info.append({
            "idx": idx,
            "titulo": titulo,
            "img": X_test[idx],
            "gt": y_test_gt[idx],
            "pred": next(m["pred"] for m in metricas if m["idx"] == idx),
            "probs": next(m["probs"] for m in metricas if m["idx"] == idx),
            "num_gt": next(m["num_gt"] for m in metricas if m["idx"] == idx),
            "num_pred": next(m["num_pred"] for m in metricas if m["idx"] == idx),
            "ap50": next(m["ap50"] for m in metricas if m["idx"] == idx)
        })
        
    return casos_info



def plot_painel_5_falhas(casos_info):
    """
    Plota para cada um dos 5 casos de falha um painel quádruplo contendo:
      1. Imagem Original (RGB)
      2. Ground Truth de Instâncias
      3. Predição Final da U-Net Trilha A
      4. Mapa Intermediário Relevante (Probabilidade de Fronteira)
    """
    for i, caso in enumerate(casos_info, start=1):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
        
        # 1. Imagem Original
        axes[0].imshow(caso["img"])
        axes[0].set_title(f"Amostra #{caso['idx']}: Entrada RGB", fontsize=11, fontweight='bold')
        axes[0].axis("off")
        
        # 2. Ground Truth
        gt_vis = caso["gt"].copy()
        axes[1].imshow(gt_vis, cmap="nipy_spectral")
        axes[1].set_title(f"Ground Truth ({caso['num_gt']} objetos)", fontsize=11, fontweight='bold')
        axes[1].axis("off")
        
        # 3. Predição Final
        pred_vis = caso["pred"].copy()
        axes[2].imshow(pred_vis, cmap="nipy_spectral")
        axes[2].set_title(f"Predição ({caso['num_pred']} instâncias | AP@50: {caso['ap50']:.2f})", fontsize=11, fontweight='bold')
        axes[2].axis("off")
        
        # 4. Mapa Intermediário (Probabilidade de Fronteira - Classe 2)
        prob_fronteira = caso["probs"][2] # canal 2: fronteira
        im4 = axes[3].imshow(prob_fronteira, cmap="magma", vmin=0, vmax=1)
        axes[3].set_title("Mapa Intermediário: Prob. Fronteira", fontsize=11, fontweight='bold')
        axes[3].axis("off")
        plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.suptitle(f"--- {caso['titulo']} ---", fontsize=13, fontweight='bold', y=1.03)
        plt.tight_layout()
        plt.show()


# --- 4. Implementação da Correção (Watershed com Supressão por Distância Euclidiana) ---

def decodificar_watershed_adaptativo(pred_probs, min_distance=4, threshold_interior=0.35, threshold_fg=0.35):
    """
    Implementação da correção proposta:
    Combina a probabilidade de interior com a Transformada de Distância Euclidiana sobre a máscara
    de primeiro plano (foreground), aplicando filtro de máxima local com raio mínimo de separação espacial.
    Isso suprime múltiplos marcadores ruidosos em núcleos grandes/anulares preservando separação legítima.
    """
    prob_interior = pred_probs[1]
    prob_fronteira = pred_probs[2]
    prob_fg = prob_interior + prob_fronteira
    
    fg_mask = prob_fg > threshold_fg
    if not np.any(fg_mask):
        return np.zeros_like(prob_interior, dtype=np.int32)
        
    # 1. Transformada de Distância Euclidiana da máscara de foreground
    dist_map = ndi.distance_transform_edt(fg_mask)
    
    # 2. Ponderação da distância pela confiança do interior predito
    seed_score = dist_map * (prob_interior ** 0.5)
    
    # 3. Supressão não-máxima com janela de min_distance
    size = 2 * min_distance + 1
    local_max = (seed_score == ndi.maximum_filter(seed_score, size=size))
    # Filtra marcadores apenas onde há probabilidade real de interior e distância significativa
    markers_mask = local_max & (prob_interior > threshold_interior) & (dist_map >= 1.5)
    
    markers, num_markers = ndi.label(markers_mask)
    if num_markers == 0:
        markers, num_markers = ndi.label(prob_interior > threshold_interior)
        if num_markers == 0:
            return np.zeros_like(prob_interior, dtype=np.int32)
            
    # 4. Bacia topográfica: fundo + fronteiras (altas elevações)
    elevation = (1.0 - prob_interior) + 1.2 * prob_fronteira
    elevation = np.clip(elevation, 0.0, None)
    
    if watershed is not None:
        labeled_instances = watershed(elevation, markers=markers, mask=fg_mask)
    else:
        labeled_instances, _ = ndi.label(prob_interior > threshold_interior)
        
    unique_ids = np.unique(labeled_instances[labeled_instances > 0])
    final_mask = np.zeros_like(labeled_instances, dtype=np.int32)
    for new_id, old_id in enumerate(unique_ids, start=1):
        final_mask[labeled_instances == old_id] = new_id

    return final_mask, len(unique_ids)


def avaliar_e_plotar_correcao(model, X_test, y_test_gt, caso_idx, device=None, min_distance=4):
    """
    Aplica a correção de pós-processamento sobre o caso selecionado e exibe
    a comparação visual Antes vs. Depois lado a lado com a variação das métricas.
    """
    if device is not None:
        model.to(device)
    target_device = next(model.parameters()).device

    img = X_test[caso_idx]
    gt = y_test_gt[caso_idx]
    
    x_t = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device=target_device).unsqueeze(0) / 255.0
    with torch.no_grad():
        logits = model(x_t)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    # Antes: Watershed padrão
    res_antes = decodificar_watershed_trilha_a(probs, threshold_interior=0.35, threshold_fg=0.35)
    pred_antes = res_antes[0] if isinstance(res_antes, tuple) else res_antes
    num_antes = int(res_antes[1]) if isinstance(res_antes, tuple) else len(np.unique(pred_antes[pred_antes > 0]))
    
    # Depois: Watershed Adaptativo com Supressão por Distância
    res_depois = decodificar_watershed_adaptativo(probs, min_distance=min_distance, threshold_interior=0.35, threshold_fg=0.35)
    pred_depois = res_depois[0] if isinstance(res_depois, tuple) else res_depois
    num_depois = int(res_depois[1]) if isinstance(res_depois, tuple) else len(np.unique(pred_depois[pred_depois > 0]))
    
    num_gt = len(np.unique(gt[gt > 0]))
    
    # Avaliação das métricas
    iou_antes = calculate_instance_iou_matrix(gt, num_gt, pred_antes, num_antes)
    tp_antes = match_instances_hungarian(iou_antes, 0.50) if (num_gt > 0 and num_antes > 0) else 0
    ap50_antes = tp_antes / (num_antes + num_gt - tp_antes) if (num_antes + num_gt - tp_antes) > 0 else 0.0
    
    iou_depois = calculate_instance_iou_matrix(gt, num_gt, pred_depois, num_depois)
    tp_depois = match_instances_hungarian(iou_depois, 0.50) if (num_gt > 0 and num_depois > 0) else 0
    ap50_depois = tp_depois / (num_depois + num_gt - tp_depois) if (num_depois + num_gt - tp_depois) > 0 else 0.0
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    axes[0].imshow(img)
    axes[0].set_title("Imagem de Entrada", fontsize=11, fontweight='bold')
    axes[0].axis("off")
    
    axes[1].imshow(gt, cmap="nipy_spectral")
    axes[1].set_title(f"Ground Truth ({num_gt} objetos)", fontsize=11, fontweight='bold')
    axes[1].axis("off")
    
    axes[2].imshow(pred_antes, cmap="nipy_spectral")
    axes[2].set_title(f"Antes da Correção ({num_antes} instâncias | AP50: {ap50_antes:.2f})", fontsize=11, fontweight='bold')
    axes[2].axis("off")
    
    axes[3].imshow(pred_depois, cmap="nipy_spectral")
    axes[3].set_title(f"Depois da Correção ({num_depois} instâncias | AP50: {ap50_depois:.2f})", fontsize=11, fontweight='bold')
    axes[3].axis("off")
    
    plt.suptitle("--- VALIDAÇÃO DA CORREÇÃO IMPLEMENTADA (ANTES VS. DEPOIS) ---", fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.show()
    
    print("=" * 70)
    print("   RESULTADOS DA CORREÇÃO: SUPRESSÃO ADAPTATIVA DE MARCADORES")
    print("=" * 70)
    print(f"Métrica                 | Antes da Correção | Depois da Correção | Ganho")
    print("-" * 70)
    print(f"AP @ IoU=0.50           | {ap50_antes:.4f}            | {ap50_depois:.4f}             | {ap50_depois - ap50_antes:+.4f}")
    print(f"Instâncias Preditas     | {num_antes:<17} | {num_depois:<18} | {num_depois - num_antes:+d}")
    print(f"Ground Truth (Real)     | {num_gt:<17} | {num_gt:<18} | --")
    print(f"Erro de Contagem        | {abs(num_antes - num_gt):<17} | {abs(num_depois - num_gt):<18} | {abs(num_depois - num_gt) - abs(num_antes - num_gt):+d}")
    print("=" * 70)
    
    return {
        "ap50_antes": ap50_antes,
        "ap50_depois": ap50_depois,
        "err_antes": abs(num_antes - num_gt),
        "err_depois": abs(num_depois - num_gt)
    }


# --- 5. Correção para Estrutura Anular / Donut (Detecção de Picos de Fluorescência no Anel) ---

def decodificar_watershed_donut_anular(img, pred_probs, min_distance=2, threshold_rel=0.10):
    """
    Correção para a falha do 'Donut' (Amostra #96: estrutura anular com 250 células compactadas).
    Problema: o anel do donut comprime 250 células minúsculas (~1.8 px). A U-Net prevê borda
    alta nos perímetros externo e interno do toro, mas não resolve as divisões internas.
    O Watershed padrão fatiou o donut em apenas ~20 fatias gigantes (sub-segmentação severa).

    Correção proposta:
      1. Extrai a máscara de primeiro plano do toro (prob_interior + prob_fronteira > 0.20).
      2. Detecta picos locais de intensidade de fluorescência (peak_local_max) ao longo do anel.
      3. Utiliza esses picos como sementes para um Watershed topográfico na imagem invertida,
         recuperando dezenas de células que haviam colapsado nas fatias gigantes.
    """
    try:
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
    except ImportError:
        return np.zeros_like(pred_probs[0], dtype=np.int32), 0

    prob_interior = pred_probs[1]
    prob_fronteira = pred_probs[2]
    fg_mask = (prob_interior + prob_fronteira > 0.20)

    gray = img.mean(axis=-1).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    if gray.max() > 1.0:
        gray = gray / 255.0

    coords = peak_local_max(gray, min_distance=min_distance, threshold_rel=threshold_rel, labels=fg_mask)
    if len(coords) == 0:
        return np.zeros_like(gray, dtype=np.int32), 0

    markers = np.zeros_like(gray, dtype=int)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i

    labeled = watershed(-gray, markers=markers, mask=fg_mask)
    unique_ids = np.unique(labeled[labeled > 0])
    final = np.zeros_like(labeled, dtype=np.int32)
    for new_id, old_id in enumerate(unique_ids, start=1):
        final[labeled == old_id] = new_id

    return final, len(unique_ids)


def avaliar_e_plotar_correcao_donut(model, X_test, y_test_gt, caso_idx, device=None, min_distance=2, threshold_rel=0.10):
    """
    Aplica a correção para a estrutura anular (Donut - Amostra #96) e exibe
    a comparação visual Antes vs. Depois com métricas quantitativas.
    """
    if device is not None:
        model.to(device)
    target_device = next(model.parameters()).device

    img = X_test[caso_idx]
    gt = y_test_gt[caso_idx]

    x_t = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device=target_device).unsqueeze(0) / 255.0
    with torch.no_grad():
        logits = model(x_t)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Antes: Watershed padrão da Trilha A
    res_antes = decodificar_watershed_trilha_a(probs, threshold_interior=0.35, threshold_fg=0.35)
    pred_antes = res_antes[0] if isinstance(res_antes, tuple) else res_antes
    num_antes = int(res_antes[1]) if isinstance(res_antes, tuple) else len(np.unique(pred_antes[pred_antes > 0]))

    # Depois: Detecção de picos de fluorescência ao longo do anel do Donut
    res_depois = decodificar_watershed_donut_anular(
        img, probs, min_distance=min_distance, threshold_rel=threshold_rel
    )
    pred_depois = res_depois[0] if isinstance(res_depois, tuple) else res_depois
    num_depois = int(res_depois[1]) if isinstance(res_depois, tuple) else len(np.unique(pred_depois[pred_depois > 0]))

    num_gt = len(np.unique(gt[gt > 0]))

    # Métricas
    iou_antes = calculate_instance_iou_matrix(gt, num_gt, pred_antes, num_antes)
    tp_antes = match_instances_hungarian(iou_antes, 0.50) if (num_gt > 0 and num_antes > 0) else 0
    ap50_antes = tp_antes / (num_antes + num_gt - tp_antes) if (num_antes + num_gt - tp_antes) > 0 else 0.0

    iou_depois = calculate_instance_iou_matrix(gt, num_gt, pred_depois, num_depois)
    tp_depois = match_instances_hungarian(iou_depois, 0.50) if (num_gt > 0 and num_depois > 0) else 0
    ap50_depois = tp_depois / (num_depois + num_gt - tp_depois) if (num_depois + num_gt - tp_depois) > 0 else 0.0

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    axes[0].imshow(img)
    axes[0].set_title("Imagem de Entrada (Donut)", fontsize=11, fontweight='bold')
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="nipy_spectral")
    axes[1].set_title(f"Ground Truth ({num_gt} objetos)", fontsize=11, fontweight='bold')
    axes[1].axis("off")

    axes[2].imshow(pred_antes, cmap="nipy_spectral")
    axes[2].set_title(f"Antes ({num_antes} inst. | AP50: {ap50_antes:.2f})", fontsize=11, fontweight='bold')
    axes[2].axis("off")

    axes[3].imshow(pred_depois, cmap="nipy_spectral")
    axes[3].set_title(f"Depois ({num_depois} inst. | AP50: {ap50_depois:.2f})", fontsize=11, fontweight='bold')
    axes[3].axis("off")

    plt.suptitle("--- CORREÇÃO: DETECÇÃO DE PICOS NO ANEL (DONUT - AMOSTRA #96) ---", fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.show()

    print("=" * 70)
    print("   RESULTADOS: RECUPERAÇÃO DE SEMENTES NO ANEL (DONUT - AMOSTRA #96)")
    print("=" * 70)
    print(f"Métrica                 | Antes da Correção | Depois da Correção | Ganho")
    print("-" * 70)
    print(f"AP @ IoU=0.50           | {ap50_antes:.4f}            | {ap50_depois:.4f}             | {ap50_depois - ap50_antes:+.4f}")
    print(f"Instâncias Preditas     | {num_antes:<17} | {num_depois:<18} | {num_depois - num_antes:+d}")
    print(f"Ground Truth (Real)     | {num_gt:<17} | {num_gt:<18} | --")
    print(f"Erro de Contagem        | {abs(num_antes - num_gt):<17} | {abs(num_depois - num_gt):<18} | {abs(num_depois - num_gt) - abs(num_antes - num_gt):+d}")
    print("=" * 70)

    return {
        "ap50_antes": ap50_antes,
        "ap50_depois": ap50_depois,
        "err_antes": abs(num_antes - num_gt),
        "err_depois": abs(num_depois - num_gt)
    }

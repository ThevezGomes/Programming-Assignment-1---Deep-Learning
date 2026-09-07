import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    from .data import identificar_modalidades, calcular_pesos_classes_trilha_a
    from .models import create_segmentation_head, UNetResNet
    from .training import train_model_trilha_a
    from .metrics import evaluate_model_instances_trilha_a
    from .postprocessing import decodificar_watershed_trilha_a
    from .losses import FocalLossMultiClass
except ImportError:
    from data import identificar_modalidades, calcular_pesos_classes_trilha_a
    from models import create_segmentation_head, UNetResNet
    from training import train_model_trilha_a
    from metrics import evaluate_model_instances_trilha_a
    from postprocessing import decodificar_watershed_trilha_a
    from losses import FocalLossMultiClass


def filtrar_por_modalidade(images, masks, targets_3c=None, modalidade="fluorescence", excluir=False):
    """
    Filtra subconjuntos de imagens e anotações por modalidade de microscopia.

    Parâmetros:
      images: np.ndarray (N, H, W, 3)
      masks: np.ndarray (N, H, W) com IDs de instâncias
      targets_3c: np.ndarray opcional (N, H, W) com alvos da Trilha A
      modalidade: string ('fluorescence', 'brightfield_color', 'brightfield_gray')
      excluir: bool. Se True, seleciona todas as amostras EXCETO a modalidade especificada.
               Se False, seleciona EXCLUSIVAMENTE a modalidade especificada.

    Retorna:
      (images_filt, masks_filt) se targets_3c for None
      (images_filt, masks_filt, targets_3c_filt) se targets_3c for fornecido
    """
    mods = identificar_modalidades(images)
    if excluir:
        idx = np.where(mods != modalidade)[0]
    else:
        idx = np.where(mods == modalidade)[0]

    if targets_3c is not None:
        return images[idx], masks[idx], targets_3c[idx]
    return images[idx], masks[idx]


def treinar_modelo_modalidade(X_train, y_train_3c, X_val, y_val_3c, device,
                             num_epochs=15, batch_size=16, learning_rate=0.0005,
                             random_seed=42):
    """
    Instancia e treina uma U-Net (Trilha A) exclusivamente no subconjunto de dados fornecido,
    recalculando dinamicamente os pesos balanceados de classes.

    Parâmetros:
      X_train: imagens de treino da modalidade (N_tr, H, W, 3)
      y_train_3c: alvos de 3 classes de treino (N_tr, H, W)
      X_val: imagens de validação da modalidade (N_val, H, W, 3)
      y_val_3c: alvos de 3 classes de validação (N_val, H, W)
      device: dispositivo de execução (cuda, mps ou cpu)
      num_epochs: número de épocas
      batch_size: tamanho do batch
      learning_rate: taxa de aprendizado
      random_seed: semente determinística

    Retorna:
      (model, class_weights)
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    class_weights = calcular_pesos_classes_trilha_a(y_train_3c)
    head = create_segmentation_head(in_channels=32, out_channels=3, head_type="conv1x1")
    model = UNetResNet(head=head, pretrained=True, freeze_backbone=False)

    print(f"Iniciando treinamento na modalidade selecionada ({X_train.shape[0]} amostras de treino)...")
    train_model_trilha_a(
        model=model,
        X_train=X_train,
        y_train_3c=y_train_3c,
        X_val=X_val,
        y_val_3c=y_val_3c,
        device=device,
        class_weights=class_weights,
        gamma=0.0,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    return model, class_weights


def avaliar_teste_estresse_modalidade(model_stress, model_full, X_test, y_test_gt, device,
                                     mod_treinada="fluorescence", mod_estresse="brightfield_color",
                                     threshold_interior=0.35, threshold_fg=0.35, matching_method="hungarian"):
    """
    Executa o Teste de Estresse da Parte 6:
    Avalia o modelo treinado sem a modalidade de teste (model_stress) sob domain shift extremo
    (mod_estresse) e compara com o controle in-domain e com o modelo treinado em todas as modalidades (model_full).

    Retorna um dicionário estruturado com as métricas de instâncias.
    """
    mods_test = identificar_modalidades(X_test)
    idx_in = np.where(mods_test == mod_treinada)[0]
    idx_out = np.where(mods_test == mod_estresse)[0]

    assert len(idx_out) > 0, f"Nenhuma amostra encontrada para a modalidade de estresse: {mod_estresse}"
    assert len(idx_in) > 0, f"Nenhuma amostra encontrada para a modalidade in-domain: {mod_treinada}"

    print(f"Avaliando Teste de Estresse:")
    print(f"  - In-Domain ({mod_treinada}): {len(idx_in)} amostras de teste")
    print(f"  - Out-of-Domain / Estresse ({mod_estresse}): {len(idx_out)} amostras de teste")

    # 1. Modelo Completo na modalidade de estresse (Referência Generalista)
    print("\n[1/3] Avaliando Modelo Completo (M_all) na modalidade de estresse...")
    res_full_out = evaluate_model_instances_trilha_a(
        model=model_full,
        X=X_test[idx_out],
        y_gt_instances=y_test_gt[idx_out],
        device=device,
        threshold_interior=threshold_interior,
        threshold_fg=threshold_fg,
        matching_method=matching_method
    )

    # 2. Modelo de Estresse no controle In-Domain (fluorescence)
    print("\n[2/3] Avaliando Modelo de Estresse (M_stress) no In-Domain...")
    res_stress_in = evaluate_model_instances_trilha_a(
        model=model_stress,
        X=X_test[idx_in],
        y_gt_instances=y_test_gt[idx_in],
        device=device,
        threshold_interior=threshold_interior,
        threshold_fg=threshold_fg,
        matching_method=matching_method
    )

    # 3. Modelo de Estresse na modalidade nunca vista (brightfield_color)
    print("\n[3/3] Avaliando Modelo de Estresse (M_stress) na modalidade NÃO VISTA (Estresse)...")
    res_stress_out = evaluate_model_instances_trilha_a(
        model=model_stress,
        X=X_test[idx_out],
        y_gt_instances=y_test_gt[idx_out],
        device=device,
        threshold_interior=threshold_interior,
        threshold_fg=threshold_fg,
        matching_method=matching_method
    )

    # Extrai AP@0.50 e AP@0.75
    idx_50 = 0
    idx_75 = 5

    def extrair_resumo(res):
        num_preds = res.get("num_pred_list", [])
        num_gts = res.get("num_gt_list", [])
        return {
            "mAP": float(res["mean_mAP"]),
            "AP50": float(res["mean_aps_per_threshold"][idx_50]),
            "AP75": float(res["mean_aps_per_threshold"][idx_75]),
            "mean_count_error": float(res["mean_count_error"]),
            "mean_pred_count": float(np.mean(num_preds)) if len(num_preds) > 0 else 0.0,
            "mean_gt_count": float(np.mean(num_gts)) if len(num_gts) > 0 else 0.0,
            "raw": res
        }

    return {
        "mod_treinada": mod_treinada,
        "mod_estresse": mod_estresse,
        "n_samples_in": len(idx_in),
        "n_samples_out": len(idx_out),
        "full_on_stress": extrair_resumo(res_full_out),
        "stress_on_indomain": extrair_resumo(res_stress_in),
        "stress_on_stress": extrair_resumo(res_stress_out)
    }


def imprimir_tabela_estresse_modalidade(res_dict):
    """
    Imprime tabela comparativa demonstrando o impacto quantitativo da mudança de modalidade.
    """
    mod_tr = res_dict["mod_treinada"]
    mod_st = res_dict["mod_estresse"]

    f_out = res_dict["full_on_stress"]
    s_in = res_dict["stress_on_indomain"]
    s_out = res_dict["stress_on_stress"]

    map_ref = f_out["mAP"]
    map_stress = s_out["mAP"]
    delta_map = ((map_stress - map_ref) / (map_ref + 1e-6)) * 100.0

    print("=" * 96)
    print(" " * 20 + f"TABELA DE ESTRESSE: MUDANÇA DE MODALIDADE ({mod_tr} -> {mod_st})")
    print("=" * 96)
    headers = f"{'Configuração Experimental':<44} | {'mAP@[.5:.95]':<12} | {'AP@0.50':<8} | {'AP@0.75':<8} | {'Erro Cont.':<10}"
    print(headers)
    print("-" * 96)

    linha1 = f"1. M_all (Treinado c/ Todas)  -> Teste {mod_st:<11} | {f_out['mAP']:<12.4f} | {f_out['AP50']:<8.4f} | {f_out['AP75']:<8.4f} | {f_out['mean_count_error']:<10.2f}"
    linha2 = f"2. M_stress (Treino {mod_tr}) -> Teste {mod_tr:<11} | {s_in['mAP']:<12.4f} | {s_in['AP50']:<8.4f} | {s_in['AP75']:<8.4f} | {s_in['mean_count_error']:<10.2f}"
    linha3 = f"3. M_stress (Treino {mod_tr}) -> Teste {mod_st:<11} | {s_out['mAP']:<12.4f} | {s_out['AP50']:<8.4f} | {s_out['AP75']:<8.4f} | {s_out['mean_count_error']:<10.2f}"

    print(linha1)
    print(linha2)
    print(linha3)
    print("=" * 96)
    print(f"-> Degradação relativa de mAP no teste de estresse (Linha 3 vs Linha 1): {delta_map:+.2f}%")
    print(f"-> Aumento no erro absoluto de contagem: de {f_out['mean_count_error']:.2f} para {s_out['mean_count_error']:.2f} núcleos por imagem")
    print("=" * 96)


def plot_comparacao_estresse_modalidade(model_stress, model_full, X_test, y_test_gt, device,
                                       mod_estresse="brightfield_color", num_samples=3,
                                       threshold_interior=0.35, threshold_fg=0.35):
    """
    Renderiza um painel comparativo visual mostrando a resposta de ambos os modelos na modalidade omitida.
    Exibe: Imagem RGB, Ground Truth, Predição M_all, Predição M_stress, Probabilidade Interior e Probabilidade Fronteira.
    """
    mods_test = identificar_modalidades(X_test)
    idx_out = np.where(mods_test == mod_estresse)[0]

    assert len(idx_out) > 0, f"Nenhuma amostra encontrada para a modalidade: {mod_estresse}"
    chosen_indices = idx_out[:num_samples]

    model_full.eval()
    model_stress.eval()

    fig, axes = plt.subplots(num_samples, 6, figsize=(22, 3.8 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    with torch.no_grad():
        for row_idx, sample_idx in enumerate(chosen_indices):
            img_rgb = X_test[sample_idx]
            gt_mask = y_test_gt[sample_idx]

            inp_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

            logits_full = model_full(inp_tensor)
            probs_full = F.softmax(logits_full, dim=1).squeeze(0).cpu().numpy()
            pred_inst_full, _ = decodificar_watershed_trilha_a(
                probs_full,
                threshold_interior=threshold_interior, threshold_fg=threshold_fg
            )

            logits_stress = model_stress(inp_tensor)
            probs_stress = F.softmax(logits_stress, dim=1).squeeze(0).cpu().numpy()
            pred_inst_stress, _ = decodificar_watershed_trilha_a(
                probs_stress,
                threshold_interior=threshold_interior, threshold_fg=threshold_fg
            )

            num_gt = len(np.unique(gt_mask[gt_mask > 0]))
            num_pred_full = len(np.unique(pred_inst_full[pred_inst_full > 0]))
            num_pred_stress = len(np.unique(pred_inst_stress[pred_inst_stress > 0]))

            axes[row_idx, 0].imshow(img_rgb)
            axes[row_idx, 0].set_title(f"Amostra #{sample_idx} ({mod_estresse})\nEntrada RGB (Campo Claro)", fontsize=11)
            axes[row_idx, 0].axis('off')

            axes[row_idx, 1].imshow(gt_mask, cmap='nipy_spectral', interpolation='nearest')
            axes[row_idx, 1].set_title(f"Ground Truth\n({num_gt} núcleos)", fontsize=11)
            axes[row_idx, 1].axis('off')

            axes[row_idx, 2].imshow(pred_inst_full, cmap='nipy_spectral', interpolation='nearest')
            axes[row_idx, 2].set_title(f"M_all (Treinado c/ Todas)\n({num_pred_full} núcleos)", fontsize=11)
            axes[row_idx, 2].axis('off')

            axes[row_idx, 3].imshow(pred_inst_stress, cmap='nipy_spectral', interpolation='nearest')
            axes[row_idx, 3].set_title(f"M_stress (SEM {mod_estresse})\n({num_pred_stress} núcleos detectados)", fontsize=11, color='crimson', fontweight='bold')
            axes[row_idx, 3].axis('off')

            im_p_int = axes[row_idx, 4].imshow(probs_stress[1], cmap='magma', vmin=0, vmax=1)
            axes[row_idx, 4].set_title("Prob. Interior (M_stress)\n(Mapa de Sementes)", fontsize=11)
            axes[row_idx, 4].axis('off')
            plt.colorbar(im_p_int, ax=axes[row_idx, 4], fraction=0.046, pad=0.04)

            im_p_bnd = axes[row_idx, 5].imshow(probs_stress[2], cmap='viridis', vmin=0, vmax=1)
            axes[row_idx, 5].set_title("Prob. Fronteira (M_stress)\n(Barreiras Watershed)", fontsize=11)
            axes[row_idx, 5].axis('off')
            plt.colorbar(im_p_bnd, ax=axes[row_idx, 5], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


# =========================================================================
# SEÇÃO EXTRA — MITIGAÇÃO E CORREÇÃO DO TESTE DE ESTRESSE
# =========================================================================

def normalizar_cor_polaridade(images):
    """
    Normaliza imagens de microscopia para o espaço canônico de contraste (fundo escuro e núcleos claros).
    Se a imagem for de campo claro (fundo médio nos cantos > 100), inverte a polaridade fotométrica (255 - I)
    e equaliza o contraste dos canais (estiramento de percentis 2% a 98%).
    Se já for fluorescência, preserva a imagem original.
    """
    is_single = (images.ndim == 3)
    imgs = np.expand_dims(images, 0) if is_single else images
    normalized = []

    for img in imgs:
        corners = np.concatenate([img[:8, :8], img[-8:, :8], img[:8, -8:], img[-8:, -8:]])
        bg = float(np.mean(corners))
        if bg > 100.0:
            inv = 255.0 - img.astype(np.float32)
            p_low = np.percentile(inv, 2.0, axis=(0, 1), keepdims=True)
            p_high = np.percentile(inv, 98.0, axis=(0, 1), keepdims=True)
            norm_img = np.clip((inv - p_low) / (p_high - p_low + 1e-5) * 255.0, 0, 255).astype(np.uint8)
            normalized.append(norm_img)
        else:
            normalized.append(img.copy())

    res = np.array(normalized)
    return res[0] if is_single else res


def treinar_modelo_estresse_com_augmentation(X_train, y_train_3c, X_val, y_val_3c, device,
                                             prob_inversao=0.5, num_epochs=15, batch_size=16,
                                             learning_rate=0.0005, random_seed=42):
    """
    Treina uma U-Net (Trilha A) no subconjunto de fluorescência aplicando Data Augmentation online
    com Inversão Estocástica de polaridade fotométrica (p=prob_inversao) e variações de iluminação,
    induzindo o encoder a aprender invariância ao contraste claro/escuro.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    class_weights = calcular_pesos_classes_trilha_a(y_train_3c)
    head = create_segmentation_head(in_channels=32, out_channels=3, head_type="conv1x1")
    model = UNetResNet(head=head, pretrained=True, freeze_backbone=False)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = FocalLossMultiClass(weight=class_weights, gamma=0.0)

    num_train = X_train.shape[0]
    num_val = X_val.shape[0]

    print(f"Treinando U-Net com Data Augmentation de Inversão Estocástica (p={prob_inversao:.2f}) em {num_train} amostras...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = int(np.ceil(num_train / batch_size))

        perm = np.random.permutation(num_train)
        X_shuff = X_train[perm]
        y_shuff = y_train_3c[perm]

        for i in range(train_batches):
            batch_imgs = X_shuff[i * batch_size:(i + 1) * batch_size].copy()
            batch_masks = y_shuff[i * batch_size:(i + 1) * batch_size]

            # Inversão estocástica online por imagem no batch
            for b_idx in range(len(batch_imgs)):
                if np.random.rand() < prob_inversao:
                    batch_imgs[b_idx] = 255 - batch_imgs[b_idx]
                if np.random.rand() < 0.25:
                    alpha = np.random.uniform(0.8, 1.2)
                    beta = np.random.uniform(-15, 15)
                    batch_imgs[b_idx] = np.clip(batch_imgs[b_idx] * alpha + beta, 0, 255).astype(np.uint8)

            batch_imgs_tensor = torch.from_numpy(batch_imgs).float().permute(0, 3, 1, 2).contiguous().to(device) / 255.0
            batch_masks_tensor = torch.from_numpy(batch_masks).long().contiguous().to(device)

            optimizer.zero_grad()
            outputs = model(batch_imgs_tensor)
            loss = criterion(outputs, batch_masks_tensor)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_imgs)

        train_loss /= num_train

        model.eval()
        val_loss = 0.0
        val_batches = int(np.ceil(num_val / batch_size))
        with torch.no_grad():
            for i in range(val_batches):
                batch_imgs = X_val[i * batch_size:(i + 1) * batch_size]
                batch_masks = y_val_3c[i * batch_size:(i + 1) * batch_size]

                batch_imgs_tensor = torch.from_numpy(batch_imgs).float().permute(0, 3, 1, 2).contiguous().to(device) / 255.0
                batch_masks_tensor = torch.from_numpy(batch_masks).long().contiguous().to(device)

                outputs = model(batch_imgs_tensor)
                loss = criterion(outputs, batch_masks_tensor)
                val_loss += loss.item() * len(batch_imgs)

        val_loss /= num_val
        if (epoch + 1) % 3 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch + 1:2d}/{num_epochs:2d}] | Train Loss (Aug): {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model, class_weights


def avaliar_comparacao_correcoes(model_stress_baseline, model_stress_augmented, model_full,
                                 X_test, y_test_gt, device, mod_estresse="brightfield_color",
                                 threshold_interior=0.35, threshold_fg=0.35, matching_method="hungarian"):
    """
    Avalia e compara as estratégias de correção do estresse de modalidade no subconjunto de campo claro:
      1. Sem Correção (M_stress original sobre imagens brutas)
      2. Correção 1: Normalização de Cor/Polaridade (M_stress original sobre imagens normalizadas)
      3. Correção 2: Inversão Estocástica (M_aug sobre imagens brutas)
      4. Correção 3: Combinada (M_aug sobre imagens normalizadas)
      5. Referência: Modelo Completo (M_all treinado com todas as modalidades)
    """
    mods_test = identificar_modalidades(X_test)
    idx_out = np.where(mods_test == mod_estresse)[0]
    assert len(idx_out) > 0, f"Nenhuma amostra encontrada para: {mod_estresse}"

    X_stress = X_test[idx_out]
    y_stress = y_test_gt[idx_out]
    X_stress_norm = normalizar_cor_polaridade(X_stress)

    idx_50 = 0
    idx_75 = 5

    def extrair_resumo(res):
        num_preds = res.get("num_pred_list", [])
        num_gts = res.get("num_gt_list", [])
        return {
            "mAP": float(res["mean_mAP"]),
            "AP50": float(res["mean_aps_per_threshold"][idx_50]),
            "AP75": float(res["mean_aps_per_threshold"][idx_75]),
            "mean_count_error": float(res["mean_count_error"]),
            "mean_pred_count": float(np.mean(num_preds)) if len(num_preds) > 0 else 0.0,
            "mean_gt_count": float(np.mean(num_gts)) if len(num_gts) > 0 else 0.0,
            "raw": res
        }

    print(f"\n[1/5] Avaliando Sem Correção (M_stress sobre entrada bruta)...")
    res_sem_corr = evaluate_model_instances_trilha_a(
        model=model_stress_baseline, X=X_stress, y_gt_instances=y_stress,
        device=device, threshold_interior=threshold_interior, threshold_fg=threshold_fg,
        matching_method=matching_method
    )

    print(f"[2/5] Avaliando Correção 1: Normalização de Cor (M_stress sobre entrada normalizada)...")
    res_norm = evaluate_model_instances_trilha_a(
        model=model_stress_baseline, X=X_stress_norm, y_gt_instances=y_stress,
        device=device, threshold_interior=threshold_interior, threshold_fg=threshold_fg,
        matching_method=matching_method
    )

    print(f"[3/5] Avaliando Correção 2: Inversão Estocástica (M_aug sobre entrada bruta)...")
    res_aug = evaluate_model_instances_trilha_a(
        model=model_stress_augmented, X=X_stress, y_gt_instances=y_stress,
        device=device, threshold_interior=threshold_interior, threshold_fg=threshold_fg,
        matching_method=matching_method
    )

    print(f"[4/5] Avaliando Correção 3: Combinada (M_aug sobre entrada normalizada)...")
    res_comb = evaluate_model_instances_trilha_a(
        model=model_stress_augmented, X=X_stress_norm, y_gt_instances=y_stress,
        device=device, threshold_interior=threshold_interior, threshold_fg=threshold_fg,
        matching_method=matching_method
    )

    print(f"[5/5] Avaliando Referência: Modelo Completo (M_all)...")
    res_full = evaluate_model_instances_trilha_a(
        model=model_full, X=X_stress, y_gt_instances=y_stress,
        device=device, threshold_interior=threshold_interior, threshold_fg=threshold_fg,
        matching_method=matching_method
    )

    return {
        "mod_estresse": mod_estresse,
        "n_samples": len(idx_out),
        "sem_correcao": extrair_resumo(res_sem_corr),
        "norm_cor": extrair_resumo(res_norm),
        "inversao_estocastica": extrair_resumo(res_aug),
        "combinada": extrair_resumo(res_comb),
        "modelo_completo": extrair_resumo(res_full)
    }


def imprimir_tabela_correcoes_estresse(res_dict):
    """
    Imprime tabela comparativa das abordagens de correção do teste de estresse de modalidade.
    """
    mod_st = res_dict["mod_estresse"]
    c_none = res_dict["sem_correcao"]
    c_norm = res_dict["norm_cor"]
    c_aug = res_dict["inversao_estocastica"]
    c_comb = res_dict["combinada"]
    c_full = res_dict["modelo_completo"]

    base_map = c_none["mAP"]

    print("=" * 102)
    print(" " * 20 + f"TABELA DE CORREÇÃO: TESTE DE ESTRESSE ({mod_st})")
    print("=" * 102)
    headers = f"{'Abordagem / Estratégia':<48} | {'mAP@[.5:.95]':<12} | {'AP@0.50':<8} | {'AP@0.75':<8} | {'Erro Cont.':<10} | {'Ganho mAP':<10}"
    print(headers)
    print("-" * 102)

    def format_row(nome, dados):
        delta = dados["mAP"] - base_map
        return f"{nome:<48} | {dados['mAP']:<12.4f} | {dados['AP50']:<8.4f} | {dados['AP75']:<8.4f} | {dados['mean_count_error']:<10.2f} | {delta:+10.4f}"

    print(format_row("1. M_stress Sem Correção (Colapso Original)", c_none))
    print(format_row("2. Correção 1: Normalização de Cor (Test-Time)", c_norm))
    print(format_row("3. Correção 2: Inversão Estocástica (Data Aug)", c_aug))
    print(format_row("4. Correção 3: Combinada (Aug + Normalização)", c_comb))
    print(format_row("5. Referência: M_all (Treinado c/ Todas)", c_full))
    print("=" * 102)
    print(f"-> Ganho absoluto de mAP pela Normalização de Cor: {c_norm['mAP'] - base_map:+.4f}")
    print(f"-> Ganho absoluto de mAP pela Inversão Estocástica: {c_aug['mAP'] - base_map:+.4f}")
    print(f"-> Redução do erro de contagem: de {c_none['mean_count_error']:.2f} para {c_aug['mean_count_error']:.2f} núcleos por imagem")
    print("=" * 102)


def plot_comparacao_correcoes_estresse(model_stress_baseline, model_stress_augmented, model_full,
                                       X_test, y_test_gt, device, mod_estresse="brightfield_color",
                                       num_samples=3, threshold_interior=0.35, threshold_fg=0.35):
    """
    Renderiza painel comparativo visual mostrando o resgate da segmentação pelas técnicas de correção.
    Exibe: Imagem Original, Ground Truth, Sem Correção, Correção 1 (Normalização), Correção 2 (Inversão Estocástica) e Referência M_all.
    """
    mods_test = identificar_modalidades(X_test)
    idx_out = np.where(mods_test == mod_estresse)[0]
    chosen_indices = idx_out[:num_samples]

    model_stress_baseline.eval()
    model_stress_augmented.eval()
    model_full.eval()

    fig, axes = plt.subplots(num_samples, 6, figsize=(22, 3.8 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    with torch.no_grad():
        for row_idx, sample_idx in enumerate(chosen_indices):
            img_rgb = X_test[sample_idx]
            gt_mask = y_test_gt[sample_idx]
            img_norm = normalizar_cor_polaridade(img_rgb)

            # Tensores
            t_orig = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
            t_norm = torch.from_numpy(img_norm).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

            # 1. Sem Correção (M_stress sobre t_orig)
            l_orig = model_stress_baseline(t_orig)
            p_orig = F.softmax(l_orig, dim=1).squeeze(0).cpu().numpy()
            pred_orig, _ = decodificar_watershed_trilha_a(p_orig, threshold_interior, threshold_fg)

            # 2. Correção 1: Normalização de Cor (M_stress sobre t_norm)
            l_norm = model_stress_baseline(t_norm)
            p_norm = F.softmax(l_norm, dim=1).squeeze(0).cpu().numpy()
            pred_norm, _ = decodificar_watershed_trilha_a(p_norm, threshold_interior, threshold_fg)

            # 3. Correção 2: Inversão Estocástica (M_aug sobre t_orig)
            l_aug = model_stress_augmented(t_orig)
            p_aug = F.softmax(l_aug, dim=1).squeeze(0).cpu().numpy()
            pred_aug, _ = decodificar_watershed_trilha_a(p_aug, threshold_interior, threshold_fg)

            # 4. Referência M_all (sobre t_orig)
            l_full = model_full(t_orig)
            p_full = F.softmax(l_full, dim=1).squeeze(0).cpu().numpy()
            pred_full, _ = decodificar_watershed_trilha_a(p_full, threshold_interior, threshold_fg)

            num_gt = len(np.unique(gt_mask[gt_mask > 0]))
            num_orig = len(np.unique(pred_orig[pred_orig > 0]))
            num_norm = len(np.unique(pred_norm[pred_norm > 0]))
            num_aug = len(np.unique(pred_aug[pred_aug > 0]))
            num_full = len(np.unique(pred_full[pred_full > 0]))

            # Coluna 1: Imagem Original
            axes[row_idx, 0].imshow(img_rgb)
            axes[row_idx, 0].set_title(f"Amostra #{sample_idx}\nEntrada RGB (Campo Claro)", fontsize=11)
            axes[row_idx, 0].axis('off')

            # Coluna 2: Ground Truth
            axes[row_idx, 1].imshow(gt_mask, cmap='nipy_spectral', interpolation='nearest')
            axes[row_idx, 1].set_title(f"Ground Truth\n({num_gt} núcleos)", fontsize=11)
            axes[row_idx, 1].axis('off')

            # Coluna 3: Sem Correção (M_stress)
            axes[row_idx, 2].imshow(pred_orig, cmap='nipy_spectral', interpolation='nearest')
            axes[row_idx, 2].set_title(f"Sem Correção (M_stress)\n({num_orig} núcleos - Colapso)", fontsize=11, color='crimson', fontweight='bold')
            axes[row_idx, 2].axis('off')

            # Coluna 4: Correção 1: Normalização de Cor
            axes[row_idx, 3].imshow(pred_norm, cmap='nipy_spectral', interpolation='nearest')
            axes[row_idx, 3].set_title(f"Corr. 1: Normalização Cor\n({num_norm} núcleos)", fontsize=11, color='darkgreen', fontweight='bold')
            axes[row_idx, 3].axis('off')

            # Coluna 5: Correção 2: Inversão Estocástica
            axes[row_idx, 4].imshow(pred_aug, cmap='nipy_spectral', interpolation='nearest')
            axes[row_idx, 4].set_title(f"Corr. 2: Inversão Estocástica\n({num_aug} núcleos)", fontsize=11, color='navy', fontweight='bold')
            axes[row_idx, 4].axis('off')

            # Coluna 6: Referência: M_all
            axes[row_idx, 5].imshow(pred_full, cmap='nipy_spectral', interpolation='nearest')
            axes[row_idx, 5].set_title(f"Referência: M_all\n({num_full} núcleos)", fontsize=11)
            axes[row_idx, 5].axis('off')

    plt.tight_layout()
    plt.show()


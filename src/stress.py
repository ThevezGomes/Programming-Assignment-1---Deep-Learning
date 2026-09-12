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
except ImportError:
    from data import identificar_modalidades, calcular_pesos_classes_trilha_a
    from models import create_segmentation_head, UNetResNet
    from training import train_model_trilha_a
    from metrics import evaluate_model_instances_trilha_a
    from postprocessing import decodificar_watershed_trilha_a


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




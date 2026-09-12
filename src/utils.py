import torch

def get_device():
    """Retorna o dispositivo acelerado disponível (MPS, CUDA ou CPU)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# Reexportação de todos os módulos especializados para compatibilidade retroativa total (Facade)
try:
    from .models import (
        create_segmentation_head, DecoderBlock, UNetResNet, SegNet,
        salvar_checkpoint, carregar_checkpoint, criar_unet_res18, criar_segnet
    )
    from .inference import inferir, carregar_modelo_inferencia
    from .losses import FocalLossMultiClass
    from .postprocessing import extract_instances_naive, decodificar_watershed_trilha_a, colorir_mascara_instancias, inferir_imagem
    from .data import (
        generate_ellipse, gerar_dataset_elipses, carregar_dataset_real,
        identificar_modalidades, split_dataset, gerar_alvos_trilha_a, calcular_pesos_classes_trilha_a
    )
    from .metrics import (
        calculate_dice_coefficient, calculate_iou, evaluate_model,
        calculate_instance_iou_matrix, match_instances_greedy, match_instances_hungarian,
        evaluate_instance_metrics_single, evaluate_model_instances, evaluate_instance_level,
        compare_matching_methods, evaluate_instance_metrics_single_trilha_a,
        evaluate_model_instances_trilha_a, evaluate_instance_level_trilha_a,
        compare_baseline_vs_trilha_a, imprimir_tabela_ablação
    )
    from .training import train_model, train_model_trilha_a, train_model_trilha_a_twophase, rodar_ablação_seeds
    from .mosaic import (
        DisjointSetUnion, criar_mosaico_imagens, inferencia_mosaico_tiles,
        fundir_instancias_tiles, avaliar_e_comparar_mosaico
    )
    from .visualization import (
        plot_dataset_samples, plot_predictions, plot_naive_instance_extraction,
        plot_quantify_failure, plot_trilha_a_samples, plot_trilha_a_predictions,
        plot_objeto_fronteira_tiles, plot_mosaico_completo
    )
    from .failures import (
        calcular_campo_receptivo_resnet18, calcular_campo_receptivo_atrous,
        imprimir_tabela_campo_receptivo, extrair_diametros_objetos,
        plot_campo_receptivo_vs_dataset, minerar_5_falhas_estruturais,
        plot_painel_5_falhas, decodificar_watershed_adaptativo, avaliar_e_plotar_correcao
    )
    from .stress import (
        filtrar_por_modalidade, treinar_modelo_modalidade,
        avaliar_teste_estresse_modalidade, imprimir_tabela_estresse_modalidade,
        plot_comparacao_estresse_modalidade, normalizar_cor_polaridade,
        treinar_modelo_estresse_com_augmentation, avaliar_comparacao_correcoes,
        imprimir_tabela_correcoes_estresse, plot_comparacao_correcoes_estresse
    )
except ImportError:
    from models import (
        create_segmentation_head, DecoderBlock, UNetResNet, SegNet,
        salvar_checkpoint, carregar_checkpoint, criar_unet_res18, criar_segnet
    )
    from inference import inferir, carregar_modelo_inferencia
    from losses import FocalLossMultiClass
    from postprocessing import extract_instances_naive, decodificar_watershed_trilha_a, colorir_mascara_instancias, inferir_imagem
    from data import (
        generate_ellipse, gerar_dataset_elipses, carregar_dataset_real,
        identificar_modalidades, split_dataset, gerar_alvos_trilha_a, calcular_pesos_classes_trilha_a
    )
    from metrics import (
        calculate_dice_coefficient, calculate_iou, evaluate_model,
        calculate_instance_iou_matrix, match_instances_greedy, match_instances_hungarian,
        evaluate_instance_metrics_single, evaluate_model_instances, evaluate_instance_level,
        compare_matching_methods, evaluate_instance_metrics_single_trilha_a,
        evaluate_model_instances_trilha_a, evaluate_instance_level_trilha_a,
        compare_baseline_vs_trilha_a, imprimir_tabela_ablação
    )
    from training import train_model, train_model_trilha_a, train_model_trilha_a_twophase, rodar_ablação_seeds
    from mosaic import (
        DisjointSetUnion, criar_mosaico_imagens, inferencia_mosaico_tiles,
        fundir_instancias_tiles, avaliar_e_comparar_mosaico
    )
    from visualization import (
        plot_dataset_samples, plot_predictions, plot_naive_instance_extraction,
        plot_quantify_failure, plot_trilha_a_samples, plot_trilha_a_predictions,
        plot_objeto_fronteira_tiles, plot_mosaico_completo
    )
    from failures import (
        calcular_campo_receptivo_resnet18, calcular_campo_receptivo_atrous,
        imprimir_tabela_campo_receptivo, extrair_diametros_objetos,
        plot_campo_receptivo_vs_dataset, minerar_5_falhas_estruturais,
        plot_painel_5_falhas, decodificar_watershed_adaptativo, avaliar_e_plotar_correcao
    )
    from stress import (
        filtrar_por_modalidade, treinar_modelo_modalidade,
        avaliar_teste_estresse_modalidade, imprimir_tabela_estresse_modalidade,
        plot_comparacao_estresse_modalidade, normalizar_cor_polaridade,
        treinar_modelo_estresse_com_augmentation, avaliar_comparacao_correcoes,
        imprimir_tabela_correcoes_estresse, plot_comparacao_correcoes_estresse
    )



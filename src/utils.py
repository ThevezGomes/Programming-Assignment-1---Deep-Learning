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
        gerar_dataset_elipses, carregar_dataset_real,
        identificar_modalidades, split_dataset, gerar_alvos_trilha_a, calcular_pesos_classes_trilha_a
    )
    from .metrics import (
        calculate_dice_coefficient, calculate_iou, evaluate_model,
        calculate_instance_iou_matrix, match_instances_greedy, match_instances_hungarian,
        evaluate_instance_metrics_single, evaluate_model_instances, evaluate_instance_level,
        evaluate_instance_metrics_single_trilha_a,
        evaluate_model_instances_trilha_a, evaluate_instance_level_trilha_a,
        compare_baseline_vs_trilha_a, imprimir_tabela_ablação
    )
    from .training import (
        train_model, train_model_trilha_a, train_model_trilha_a_twophase, rodar_ablação_seeds,
        treinar_e_avaliar_sintetico_baseline, treinar_e_avaliar_sintetico_trilha_a
    )
    from .mosaic import (
        DisjointSetUnion, criar_mosaico_imagens, inferencia_mosaico_tiles,
        fundir_instancias_tiles, avaliar_e_comparar_mosaico
    )
    from .visualization import (
        plot_dataset_samples, plot_predictions, plot_naive_instance_extraction,
        plot_quantify_failure, plot_trilha_a_samples, plot_trilha_a_predictions,
        plot_objeto_fronteira_tiles, plot_mosaico_completo,
        avaliar_e_plotar_comparativo_instancias
    )
    from .failures import (
        calcular_campo_receptivo_resnet18, calcular_campo_receptivo_atrous,
        imprimir_tabela_campo_receptivo, extrair_diametros_objetos,
        plot_campo_receptivo_vs_dataset, minerar_5_falhas_estruturais,
        plot_painel_5_falhas, decodificar_watershed_adaptativo, avaliar_e_plotar_correcao,
        decodificar_watershed_donut_anular, avaliar_e_plotar_correcao_donut
    )
    from .stress import (
        filtrar_por_modalidade, treinar_modelo_modalidade,
        avaliar_teste_estresse_modalidade, imprimir_tabela_estresse_modalidade,
        plot_comparacao_estresse_modalidade
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
        gerar_dataset_elipses, carregar_dataset_real,
        identificar_modalidades, split_dataset, gerar_alvos_trilha_a, calcular_pesos_classes_trilha_a
    )
    from metrics import (
        calculate_dice_coefficient, calculate_iou, evaluate_model,
        calculate_instance_iou_matrix, match_instances_greedy, match_instances_hungarian,
        evaluate_instance_metrics_single, evaluate_model_instances, evaluate_instance_level,
        evaluate_instance_metrics_single_trilha_a,
        evaluate_model_instances_trilha_a, evaluate_instance_level_trilha_a,
        compare_baseline_vs_trilha_a, imprimir_tabela_ablação
    )
    from training import (
        train_model, train_model_trilha_a, train_model_trilha_a_twophase, rodar_ablação_seeds,
        treinar_e_avaliar_sintetico_baseline, treinar_e_avaliar_sintetico_trilha_a
    )
    from mosaic import (
        DisjointSetUnion, criar_mosaico_imagens, inferencia_mosaico_tiles,
        fundir_instancias_tiles, avaliar_e_comparar_mosaico
    )
    from visualization import (
        plot_dataset_samples, plot_predictions, plot_naive_instance_extraction,
        plot_quantify_failure, plot_trilha_a_samples, plot_trilha_a_predictions,
        plot_objeto_fronteira_tiles, plot_mosaico_completo,
        avaliar_e_plotar_comparativo_instancias
    )
    from failures import (
        calcular_campo_receptivo_resnet18, calcular_campo_receptivo_atrous,
        imprimir_tabela_campo_receptivo, extrair_diametros_objetos,
        plot_campo_receptivo_vs_dataset, minerar_5_falhas_estruturais,
        plot_painel_5_falhas, decodificar_watershed_adaptativo, avaliar_e_plotar_correcao,
        decodificar_watershed_donut_anular, avaliar_e_plotar_correcao_donut
    )
    from stress import (
        filtrar_por_modalidade, treinar_modelo_modalidade,
        avaliar_teste_estresse_modalidade, imprimir_tabela_estresse_modalidade,
        plot_comparacao_estresse_modalidade
    )

class Dataset:
    def __init__(self, stage1_path="data/stage1_train"):
        self.stage1_path = stage1_path

    def get_data(self, target_size=(128, 128)):
        self.images_real, self.masks_real = carregar_dataset_real(self.stage1_path, target_size=target_size)

    def split_data(self, train_ratio=0.70, val_ratio=0.15, seed=42, stratify=True):
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = split_dataset(
            self.images_real,
            self.masks_real,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            stratify=stratify
        )

        self.y_train = gerar_alvos_trilha_a(self.y_train, border_thickness=1)
        self.y_val = gerar_alvos_trilha_a(self.y_val, border_thickness=1)
        self.y_test = gerar_alvos_trilha_a(self.y_test, border_thickness=1)

class Model:
    def __init__(self, device=get_device(), in_channels=32, out_channels=3, head_type="conv1x1", pretrained=True, freeze_backbone=True):
        self.device = device
        self.head = create_segmentation_head(in_channels=in_channels, out_channels=out_channels, head_type=head_type)
        self.model = UNetResNet(head=self.head, pretrained=pretrained, freeze_backbone=freeze_backbone)  

    def train(self, dataset, gamma=0.0, epochs_phase1=10, epochs_phase2=10, batch_size=16, lr_phase1=0.001, lr_phase2=0.0001):
        self.class_weights = calcular_pesos_classes_trilha_a(dataset.y_train)
        train_model_trilha_a_twophase(
            self.model, 
            dataset.X_train, 
            dataset.y_train, 
            dataset.X_val, 
            dataset.y_val, 
            device=self.device,
            class_weights=self.class_weights, 
            gamma=gamma,
            epochs_phase1=epochs_phase1, 
            epochs_phase2=epochs_phase2,
            batch_size=batch_size, 
            lr_phase1=lr_phase1, 
            lr_phase2=lr_phase2
        )

    def evaluate(self, dataset, threshold_interior=0.35, threshold_fg=0.35, matching_method="hungarian"):
        dice_r, iou_r = evaluate_model(self.model, dataset.X_test, dataset.y_test, device=self.device, batch_size=16)
        print(f"\nMean Dice: {dice_r:.4f} | Mean IoU: {iou_r:.4f}")
        res_evaluate_model = evaluate_instance_level_trilha_a(
            self.model, 
            dataset.X_test, 
            dataset.y_test, 
            device=self.device, 
            threshold_interior=threshold_interior, 
            threshold_fg=threshold_fg, 
            matching_method=matching_method, 
            dataset_name="Reais (DSB2018)"
        )
        return dice_r, iou_r, res_evaluate_model
import numpy as np
import torch
import torch.nn as nn

try:
    from .losses import FocalLossMultiClass
    from .metrics import evaluate_model_instances_trilha_a, evaluate_instance_level, evaluate_instance_level_trilha_a
    from .data import gerar_alvos_trilha_a, calcular_pesos_classes_trilha_a
    from .models import criar_unet_res18
except ImportError:
    from losses import FocalLossMultiClass
    from metrics import evaluate_model_instances_trilha_a, evaluate_instance_level, evaluate_instance_level_trilha_a
    from data import gerar_alvos_trilha_a, calcular_pesos_classes_trilha_a
    from models import criar_unet_res18


def train_model(model, X_train, y_train, X_val, y_val, device, num_epochs=10, batch_size=8, learning_rate=0.001):
    """Treina o modelo reportando a perda no conjunto de treino e validação a cada época."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    num_train = X_train.shape[0]
    num_val = X_val.shape[0]

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = int(np.ceil(num_train / batch_size))

        for i in range(train_batches):
            batch_images = X_train[i * batch_size:(i + 1) * batch_size]
            batch_masks = y_train[i * batch_size:(i + 1) * batch_size]

            batch_images_tensor = torch.from_numpy(batch_images).float().permute(0, 3, 1, 2).contiguous().to(device) / 255.0
            batch_masks_tensor = torch.from_numpy((batch_masks > 0).astype(np.float32)).unsqueeze(1).contiguous().to(device)

            optimizer.zero_grad()
            outputs = model(batch_images_tensor)
            loss = criterion(outputs, batch_masks_tensor)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_images)

        train_loss /= num_train

        model.eval()
        val_loss = 0.0
        val_batches = int(np.ceil(num_val / batch_size))

        with torch.no_grad():
            for i in range(val_batches):
                batch_images = X_val[i * batch_size:(i + 1) * batch_size]
                batch_masks = y_val[i * batch_size:(i + 1) * batch_size]

                batch_images_tensor = torch.from_numpy(batch_images).float().permute(0, 3, 1, 2).contiguous().to(device) / 255.0
                batch_masks_tensor = torch.from_numpy((batch_masks > 0).astype(np.float32)).unsqueeze(1).contiguous().to(device)

                outputs = model(batch_images_tensor)
                loss = criterion(outputs, batch_masks_tensor)
                val_loss += loss.item() * len(batch_images)

        val_loss /= num_val
        print(f'Epoch [{epoch + 1:2d}/{num_epochs:2d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')


def train_model_trilha_a(model, X_train, y_train_3c, X_val, y_val_3c, device,
                         class_weights=None, gamma=0.0, num_epochs=15,
                         batch_size=16, learning_rate=0.0005):
    """Treina o modelo U-Net com 3 classes para a Trilha A usando perda balanceada e otimizador Adam."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = FocalLossMultiClass(weight=class_weights, gamma=gamma)

    num_train = X_train.shape[0]
    num_val = X_val.shape[0]

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = int(np.ceil(num_train / batch_size))

        for i in range(train_batches):
            batch_images = X_train[i * batch_size:(i + 1) * batch_size]
            batch_masks = y_train_3c[i * batch_size:(i + 1) * batch_size]

            batch_images_tensor = torch.from_numpy(batch_images).float().permute(0, 3, 1, 2).contiguous().to(device) / 255.0
            batch_masks_tensor = torch.from_numpy(batch_masks).long().contiguous().to(device)

            optimizer.zero_grad()
            outputs = model(batch_images_tensor)
            loss = criterion(outputs, batch_masks_tensor)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_images)

        train_loss /= num_train

        model.eval()
        val_loss = 0.0
        val_batches = int(np.ceil(num_val / batch_size))

        with torch.no_grad():
            for i in range(val_batches):
                batch_images = X_val[i * batch_size:(i + 1) * batch_size]
                batch_masks = y_val_3c[i * batch_size:(i + 1) * batch_size]

                batch_images_tensor = torch.from_numpy(batch_images).float().permute(0, 3, 1, 2).contiguous().to(device) / 255.0
                batch_masks_tensor = torch.from_numpy(batch_masks).long().contiguous().to(device)

                outputs = model(batch_images_tensor)
                loss = criterion(outputs, batch_masks_tensor)
                val_loss += loss.item() * len(batch_images)

        val_loss /= num_val
        print(f'Epoch [{epoch + 1:2d}/{num_epochs:2d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')


def train_model_trilha_a_twophase(model, X_train, y_train_3c, X_val, y_val_3c, device,
                                   class_weights=None, gamma=0.0,
                                   epochs_phase1=5, epochs_phase2=10,
                                   batch_size=16,
                                   lr_phase1=0.001, lr_phase2=0.0001):
    """Treina a U-Net em 2 fases para evitar degradação catastrófica do backbone pré-treinado.

    Fase 1 (backbone congelado):
        O ResNet18 fica congelado; apenas o decoder e o head aprendem a tarefa de 3 classes.
        Usa LR maior pois só os pesos aleatórios/novos são atualizados.

    Fase 2 (fine-tuning completo):
        Descongela todo o backbone com LR muito menor para ajuste fino sem destruir as
        representações pré-treinadas.
    """
    model.to(device)
    criterion = FocalLossMultiClass(weight=class_weights, gamma=gamma)

    # ── Fase 1: backbone congelado ──────────────────────────────────────────
    print(f"\n[Fase 1] Backbone CONGELADO | {epochs_phase1} épocas | LR={lr_phase1}")
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_phase1
    )

    num_train = X_train.shape[0]
    num_val   = X_val.shape[0]

    for epoch in range(epochs_phase1):
        model.train()
        train_loss = 0.0
        train_batches = int(np.ceil(num_train / batch_size))

        for i in range(train_batches):
            batch_images = X_train[i * batch_size:(i + 1) * batch_size]
            batch_masks  = y_train_3c[i * batch_size:(i + 1) * batch_size]

            imgs   = torch.from_numpy(batch_images).float().permute(0, 3, 1, 2).contiguous().to(device) / 255.0
            labels = torch.from_numpy(batch_masks).long().contiguous().to(device)

            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_images)

        train_loss /= num_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(int(np.ceil(num_val / batch_size))):
                imgs   = torch.from_numpy(X_val[i * batch_size:(i + 1) * batch_size]).float().permute(0, 3, 1, 2).contiguous().to(device) / 255.0
                labels = torch.from_numpy(y_val_3c[i * batch_size:(i + 1) * batch_size]).long().contiguous().to(device)
                val_loss += criterion(model(imgs), labels).item() * len(X_val[i * batch_size:(i + 1) * batch_size])
        val_loss /= num_val
        print(f'  Epoch [{epoch + 1:2d}/{epochs_phase1:2d}] | Train: {train_loss:.4f} | Val: {val_loss:.4f}')

    # ── Fase 2: fine-tuning completo ────────────────────────────────────────
    print(f"\n[Fase 2] Backbone DESCONGELADO (fine-tuning) | {epochs_phase2} épocas | LR={lr_phase2}")
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_phase2)

    for epoch in range(epochs_phase2):
        model.train()
        train_loss = 0.0
        train_batches = int(np.ceil(num_train / batch_size))

        for i in range(train_batches):
            batch_images = X_train[i * batch_size:(i + 1) * batch_size]
            batch_masks  = y_train_3c[i * batch_size:(i + 1) * batch_size]

            imgs   = torch.from_numpy(batch_images).float().permute(0, 3, 1, 2).contiguous().to(device) / 255.0
            labels = torch.from_numpy(batch_masks).long().contiguous().to(device)

            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_images)

        train_loss /= num_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(int(np.ceil(num_val / batch_size))):
                imgs   = torch.from_numpy(X_val[i * batch_size:(i + 1) * batch_size]).float().permute(0, 3, 1, 2).contiguous().to(device) / 255.0
                labels = torch.from_numpy(y_val_3c[i * batch_size:(i + 1) * batch_size]).long().contiguous().to(device)
                val_loss += criterion(model(imgs), labels).item() * len(X_val[i * batch_size:(i + 1) * batch_size])
        val_loss /= num_val
        print(f'  Epoch [{epoch + 1:2d}/{epochs_phase2:2d}] | Train: {train_loss:.4f} | Val: {val_loss:.4f}')


def rodar_ablação_seeds(modelo_fn, X_train, y_train_3c, X_val, y_val_3c, X_test, y_test_gt,
                        device, seeds=(42, 123), class_weights=None, gamma=0.0,
                        num_epochs=12, batch_size=16, learning_rate=0.0005,
                        threshold_interior=0.35, threshold_fg=0.35):
    """Executa o treinamento e avaliação para múltiplas seeds reportando média e desvio padrão."""
    maps = []
    erros = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = modelo_fn()
        train_model_trilha_a(
            model, X_train, y_train_3c, X_val, y_val_3c, device,
            class_weights=class_weights, gamma=gamma, num_epochs=num_epochs,
            batch_size=batch_size, learning_rate=learning_rate
        )

        res = evaluate_model_instances_trilha_a(
            model, X_test, y_test_gt, device,
            threshold_interior=threshold_interior, threshold_fg=threshold_fg
        )
        maps.append(res['mean_mAP'])
        erros.append(res['mean_count_error'])

    return {
        "mAP_mean": float(np.mean(maps)),
        "mAP_std": float(np.std(maps)),
        "err_mean": float(np.mean(erros)),
        "err_std": float(np.std(erros)),
        "seeds": seeds,
        "maps_raw": maps,
        "erros_raw": erros,
    }


def treinar_e_avaliar_sintetico_baseline(
    X_train, y_train, X_val, y_val, X_test, y_test, device,
    model=None, num_epochs=10, batch_size=8, learning_rate=0.001,
    threshold=0.5, matching_method="hungarian"
):
    """
    Parte 1: Treina (ou reutiliza se já fornecido) e avalia o modelo U-Net binário baseline no dataset sintético (In-Domain).
    Retorna (model, results_dict).
    """
    if model is None:
        print(f"\n--> Treinando modelo U-Net Baseline nos dados Sintéticos ({num_epochs} épocas)...")
        model = criar_unet_res18(out_channels=1, pretrained=True, freeze_backbone=True)
        train_model(
            model, X_train, y_train, X_val, y_val, device,
            num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate
        )

    print("\n--> Avaliando modelo Baseline no Teste Sintético (In-Domain)...")
    res = evaluate_instance_level(
        model, X_test, y_test, device=device,
        threshold=threshold, matching_method=matching_method,
        dataset_name="Sintéticos (Elipses - In-Domain)"
    )

    return model, res


def treinar_e_avaliar_sintetico_trilha_a(
    X_train, y_train, X_val, y_val, X_test, y_test, device,
    model=None, border_thickness=1, gamma=1.0, num_epochs=10,
    batch_size=16, learning_rate=0.0005,
    threshold_interior=0.35, threshold_fg=0.35, matching_method="hungarian"
):
    """
    Parte 2: Gera alvos de 3 classes para elipses sintéticas, treina (ou reutiliza) a U-Net Trilha A
    e avalia via Watershed com Algoritmo Húngaro (In-Domain).
    Retorna (model, results_dict).
    """
    if model is None:
        print("\n--> Gerando alvos de 3 classes para as Elipses Sintéticas...")
        y_train_3c = gerar_alvos_trilha_a(y_train, border_thickness=border_thickness)
        y_val_3c = gerar_alvos_trilha_a(y_val, border_thickness=border_thickness)
        weights = calcular_pesos_classes_trilha_a(y_train_3c)

        print(f"--> Treinando U-Net Trilha A nos dados Sintéticos ({num_epochs} épocas)...")
        model = criar_unet_res18(out_channels=3, pretrained=True, freeze_backbone=True)
        train_model_trilha_a(
            model, X_train, y_train_3c, X_val, y_val_3c, device,
            class_weights=weights, gamma=gamma, num_epochs=num_epochs,
            batch_size=batch_size, learning_rate=learning_rate
        )

    print("\n--> Avaliando U-Net Trilha A no Teste Sintético (In-Domain)...")
    res = evaluate_instance_level_trilha_a(
        model, X_test, y_test, device=device,
        threshold_interior=threshold_interior, threshold_fg=threshold_fg,
        matching_method=matching_method,
        dataset_name="Sintéticos (Trilha A - In-Domain)"
    )

    return model, res


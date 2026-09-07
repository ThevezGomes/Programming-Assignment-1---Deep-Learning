import numpy as np
import torch
import torch.nn as nn

try:
    from .losses import FocalLossMultiClass
    from .metrics import evaluate_model_instances_trilha_a
except ImportError:
    from losses import FocalLossMultiClass
    from metrics import evaluate_model_instances_trilha_a


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

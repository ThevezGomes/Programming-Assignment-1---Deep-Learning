import os
import glob
import time
import numpy as np
import cv2
import torch


def generate_ellipse(image, center, axes, angle, color=None, thickness=-1):
    """Desenha uma elipse na imagem com cor ou rótulo especificado."""
    if color is None:
        color = (
            int(np.random.randint(0, 256)),
            int(np.random.randint(0, 256)),
            int(np.random.randint(0, 256)),
        )
    return cv2.ellipse(image, center, axes, angle, 0, 360, color, thickness)


def gerar_dataset_elipses(num_images=100, image_size=(128, 128), num_ellipses_range=(5, 20), seed=42):
    """Gera dataset sintético de elipses com ruído, variação de contraste e máscaras de instâncias (Parte 0)."""
    np.random.seed(seed)
    images = []
    instance_masks = []

    for _ in range(num_images):
        img = np.zeros((image_size[0], image_size[1], 3), dtype=np.float32)
        inst_mask = np.zeros(image_size, dtype=np.int32)
        num_ellipses = np.random.randint(num_ellipses_range[0], num_ellipses_range[1] + 1)

        for i in range(1, num_ellipses + 1):
            center = (np.random.randint(0, image_size[1]), np.random.randint(0, image_size[0]))
            axes = (np.random.randint(5, 20), np.random.randint(5, 20))
            angle = np.random.randint(0, 360)
            color = (
                float(np.random.randint(50, 256)),
                float(np.random.randint(50, 256)),
                float(np.random.randint(50, 256)),
            )

            cv2.ellipse(img, center, axes, angle, 0, 360, color, -1)
            cv2.ellipse(inst_mask, center, axes, angle, 0, 360, int(i), -1)

        alpha = np.random.uniform(0.6, 1.4)
        beta = np.random.uniform(-30, 30)
        img = img * alpha + beta

        noise = np.random.normal(0, 15, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        images.append(img)
        instance_masks.append(inst_mask)

    return np.array(images), np.array(instance_masks)


def carregar_dataset_real(stage1_dir, target_size=(128, 128)):
    """Carrega dataset real DSB2018 gerando máscaras rotuladas por instância (1, 2, 3...)."""
    image_ids = [d for d in os.listdir(stage1_dir) if os.path.isdir(os.path.join(stage1_dir, d))]
    images = []
    instance_masks = []

    print(f"Carregando {len(image_ids)} amostras de '{stage1_dir}'...")
    t0 = time.time()
    lost_labels_count = 0
    total_labels_count = 0

    for img_id in image_ids:
        img_folder = os.path.join(stage1_dir, img_id)
        img_paths = glob.glob(os.path.join(img_folder, 'images', '*.png'))
        if not img_paths:
            continue

        img = cv2.imread(img_paths[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_paths = glob.glob(os.path.join(img_folder, 'masks', '*.png'))
        inst_mask = np.zeros(img.shape[:2], dtype=np.int32)
        for idx, mp in enumerate(mask_paths, start=1):
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            inst_mask[m > 0] = idx

        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(inst_mask, target_size, interpolation=cv2.INTER_NEAREST)

        orig_unique = set(range(1, len(mask_paths) + 1))
        resized_unique = set(np.unique(mask_resized)) - {0}
        lost_ids = orig_unique - resized_unique

        if lost_ids:
            scale_y = target_size[1] / inst_mask.shape[0]
            scale_x = target_size[0] / inst_mask.shape[1]
            for lid in lost_ids:
                ys, xs = np.where(inst_mask == lid)
                if len(ys) > 0:
                    cy = int(np.clip(np.round(ys.mean() * scale_y), 0, target_size[1] - 1))
                    cx = int(np.clip(np.round(xs.mean() * scale_x), 0, target_size[0] - 1))
                    placed = False
                    for dy in [0, -1, 1, -2, 2]:
                        for dx in [0, -1, 1, -2, 2]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < target_size[1] and 0 <= nx < target_size[0]:
                                if mask_resized[ny, nx] == 0:
                                    mask_resized[ny, nx] = lid
                                    placed = True
                                    break
                        if placed:
                            break
                    if not placed:
                        mask_resized[cy, cx] = lid

        final_unique = set(np.unique(mask_resized)) - {0}
        total_labels_count += len(orig_unique)
        lost_labels_count += (len(orig_unique) - len(final_unique))

        relabeled_mask = np.zeros_like(mask_resized, dtype=np.int32)
        for new_id, old_id in enumerate(sorted(final_unique), start=1):
            relabeled_mask[mask_resized == old_id] = new_id

        images.append(img_resized)
        instance_masks.append(relabeled_mask)

    t1 = time.time()
    if lost_labels_count > 0:
        print(f"Aviso: {lost_labels_count}/{total_labels_count} rótulos de instâncias foram perdidos no resize.")
    print(f"Dataset real carregado em {t1 - t0:.2f} segundos!")
    return np.array(images), np.array(instance_masks)


def identificar_modalidades(images):
    """
    Classifica cada imagem em sua modalidade de microscopia no DSB2018:
    - 'fluorescence': Fundo escuro (< 100) com núcleos fluorescentes brilhantes.
    - 'brightfield_color': Campo claro com coloração histológica (H&E, tons púrpuras/rosas).
    - 'brightfield_gray': Campo claro monocromático em transmissão.
    """
    modalidades = []
    for img in images:
        corners = np.concatenate([img[:8, :8], img[-8:, :8], img[:8, -8:], img[-8:, -8:]])
        bg = float(np.mean(corners))
        color_std = float(np.std(img, axis=-1).mean())
        if bg < 100:
            modalidades.append("fluorescence")
        elif color_std > 5.0:
            modalidades.append("brightfield_color")
        else:
            modalidades.append("brightfield_gray")
    return np.array(modalidades)


def split_dataset(images, masks, train_ratio=0.70, val_ratio=0.15, seed=42, stratify=True):
    """
    Divide um conjunto de dados em treino, validação e teste com estratificação por modalidade
    conforme exigido no enunciado do PA1 (Seção 2: Dados).
    """
    num_samples = len(images)
    np.random.seed(seed)

    if stratify and images.ndim == 4:
        mods = identificar_modalidades(images)
        unique_mods = np.unique(mods)

        train_idx_all = []
        val_idx_all = []
        test_idx_all = []

        for m in unique_mods:
            m_indices = np.where(mods == m)[0]
            np.random.shuffle(m_indices)
            n_m = len(m_indices)

            t_end = int(train_ratio * n_m)
            v_end = int((train_ratio + val_ratio) * n_m)

            train_idx_all.extend(m_indices[:t_end])
            val_idx_all.extend(m_indices[t_end:v_end])
            test_idx_all.extend(m_indices[v_end:])

        train_idx = np.array(train_idx_all)
        val_idx = np.array(val_idx_all)
        test_idx = np.array(test_idx_all)

        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)

        from collections import Counter
        print(f"Split estratificado por modalidade (Total: {num_samples}):")
        print(f"  Treino ({len(train_idx)}): {dict(Counter(mods[train_idx]))}")
        print(f"  Validação ({len(val_idx)}): {dict(Counter(mods[val_idx]))}")
        print(f"  Teste ({len(test_idx)}): {dict(Counter(mods[test_idx]))}")
    else:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        train_end = int(train_ratio * num_samples)
        val_end = int((train_ratio + val_ratio) * num_samples)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

    X_train, y_train = images[train_idx], masks[train_idx]
    X_val, y_val = images[val_idx], masks[val_idx]
    X_test, y_test = images[test_idx], masks[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def gerar_alvos_trilha_a(instance_masks, border_thickness=1):
    """Gera alvos de 3 classes (0: fundo, 1: interior, 2: fronteira) preservando sementes de núcleos pequenos."""
    num_samples = len(instance_masks)
    h, w = instance_masks.shape[1:3]
    targets_3c = np.zeros((num_samples, h, w), dtype=np.int64)
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    for idx in range(num_samples):
        mask = instance_masks[idx]
        unique_ids = np.unique(mask[mask > 0])
        interior_mask = np.zeros((h, w), dtype=bool)

        for uid in unique_ids:
            obj = (mask == uid).astype(np.uint8)
            if obj.sum() <= 12:
                coords = np.argwhere(obj > 0)
                cy, cx = coords[len(coords) // 2]
                interior_mask[cy, cx] = True
            else:
                eroded = cv2.erode(obj, kernel_cross, iterations=border_thickness)
                if eroded.sum() == 0:
                    coords = np.argwhere(obj > 0)
                    cy, cx = coords[len(coords) // 2]
                    interior_mask[cy, cx] = True
                else:
                    interior_mask |= (eroded > 0)

        fg_mask = (mask > 0)
        border_mask = fg_mask & (~interior_mask)

        alvo = np.zeros((h, w), dtype=np.int64)
        alvo[border_mask] = 2
        alvo[interior_mask] = 1
        targets_3c[idx] = alvo

    return targets_3c


def calcular_pesos_classes_trilha_a(y_train_3c, power=0.5, epsilon=1e-4):
    """Calcula pesos inversamente proporcionais à frequência das classes normalizados pela média."""
    counts = np.bincount(y_train_3c.flatten(), minlength=3).astype(np.float32)
    freqs = counts / (counts.sum() + epsilon)
    weights = 1.0 / (freqs ** power + epsilon)
    weights = weights / weights.mean()
    return torch.from_numpy(weights).float()

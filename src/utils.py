import os
import glob
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.decomposition import PCA
from torchvision import models
from torchvision.models import ResNet18_Weights


def get_device():
    """Retorna o dispositivo acelerado disponível (MPS, CUDA ou CPU)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# --- Modularização da Cabeça (Head) do Modelo ---

def create_segmentation_head(in_channels=32, out_channels=1, head_type="conv1x1", dropout=0.0):
    """
    Função modular para construção da cabeça (head) de predição da U-Net.
    
    Parâmetros:
      - in_channels: Número de canais de entrada provenientes do último DecoderBlock (padrão 32).
      - out_channels: Número de canais de saída:
          * 1 para segmentação binária (Partes 0 e 1).
          * 1 + D para Trilha B (Canal 0: logits de primeiro plano, Canais 1..D: vetor de embedding).
      - head_type: 'conv1x1' (padrão) ou 'conv3x3' com BatchNorm e ReLU.
      - dropout: Taxa de dropout opcional (padrão 0.0).
    """
    if head_type == "conv1x1":
        if dropout > 0:
            return nn.Sequential(
                nn.Dropout2d(p=dropout),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)
    elif head_type == "conv3x3":
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        return nn.Sequential(*layers)
    else:
        raise ValueError(f"Tipo de head '{head_type}' não reconhecido. Use 'conv1x1' ou 'conv3x3'.")


# --- Arquitetura U-Net com Backbone ResNet18 ---

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1).contiguous()
        return self.conv(x)


class UNetResNet(nn.Module):
    def __init__(self, out_channels=1, pretrained=True, freeze_backbone=True, head=None, head_type="conv1x1"):
        super().__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.backbone = backbone
        # Resoluções reais do encoder: input(H/1, 3ch), stem_skip(H/1, 32ch), encoder_conv1(H/2, 64ch), encoder0(H/4, 64ch), encoder1(H/4, 64ch), encoder2(H/8, 128ch), encoder3(H/16, 256ch), encoder4(H/32, 512ch).
        self.stem_skip = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.encoder_conv1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.encoder0 = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up4 = DecoderBlock(512, 256, 256)
        self.up3 = DecoderBlock(256, 128, 128)
        self.up2 = DecoderBlock(128, 64, 64)
        self.up1 = DecoderBlock(64, 64, 32)
        self.up0 = DecoderBlock(32, 32, 32)
        
        # Cabeça modularizada com suporte a injeção externa ou geração automática
        if head is not None:
            self.head = head
        else:
            self.head = create_segmentation_head(32, out_channels, head_type=head_type)

    def forward(self, x):
        x = x.contiguous()
        input_hw = x.shape[-2:]
        x_stem = self.stem_skip(x)
        x_c1 = self.encoder_conv1(x)
        x0 = self.encoder0(x_c1)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        x = self.center(x4)
        x = self.up4(x, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x_c1)
        x = self.up0(x, x_stem)
        logits = self.head(x)
        return F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False).contiguous()


# --- Geração e Carregamento de Datasets ---

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
        
        total_labels_count += len(orig_unique)
        lost_labels_count += (len(orig_unique) - len(resized_unique))
        
        relabeled_mask = np.zeros_like(mask_resized, dtype=np.int32)
        for new_id, old_id in enumerate(sorted(resized_unique), start=1):
            relabeled_mask[mask_resized == old_id] = new_id
            
        images.append(img_resized)
        instance_masks.append(relabeled_mask)
        
    t1 = time.time()
    if lost_labels_count > 0:
        print(f"Aviso: {lost_labels_count}/{total_labels_count} rótulos de instâncias foram perdidos no resize.")
    print(f"Dataset real carregado em {t1 - t0:.2f} segundos!")
    return np.array(images), np.array(instance_masks)


def split_dataset(images, masks, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Divide um conjunto de dados em treino, validação e teste."""
    num_samples = len(images)
    indices = np.arange(num_samples)
    np.random.seed(seed)
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


# --- Treinamento, Métricas e Extração de Instâncias Ingênua ---

def extract_instances_naive(pred_prob, threshold=0.5):
    """Extrai instâncias binarizando por limiar e aplicando componentes conexos de 8-conectividade (Item 2)."""
    binary_mask = pred_prob > threshold
    structure = np.ones((3, 3), dtype=int)
    labeled_mask, num_instances = label(binary_mask, structure=structure)
    return labeled_mask, num_instances


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


# --- Visualização ---

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


# --- Avaliação por Instâncias, Casamento (Matching) e Quantificação do Fracasso ---

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


def plot_quantify_failure(results, dataset_name="Reais"):
    """Item 5: Plota gráficos de mAP e Erro de Contagem vs. Densidade de Objetos para quantificar o fracasso."""
    num_gt = results["num_gt_list"]
    mAPs = results["mAP_list"]
    count_errors = results["count_error_list"]
    method = results["matching_method"].capitalize()

    plt.figure(figsize=(14, 5))

    # Gráfico 1: mAP vs. Densidade de Objetos
    plt.subplot(1, 2, 1)
    plt.scatter(num_gt, mAPs, alpha=0.6, color='crimson', edgecolors='k', label='Amostras')
    
    if len(num_gt) > 1 and len(np.unique(num_gt)) > 1:
        idx_sort = np.argsort(num_gt)
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

    # Gráfico 2: Erro Absoluto de Contagem vs. Densidade de Objetos
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


# =============================================================================
# --- PARTE 2: TRILHA B — EMBEDDINGS DISCRIMINATIVOS ---
# =============================================================================

class DiscriminativeLoss(nn.Module):
    """
    Função de Perda Discriminativa baseada em De Brabandere et al. (2017):
    'Semantic Instance Segmentation with a Discriminative Loss Function'.

    A rede produz (1 + D) canais:
      - Canal 0: Logits de primeiro plano (Foreground) otimizado com BCEWithLogitsLoss.
      - Canais 1..D: Vetores de embedding D-dimensionais por pixel.

    A perda discriminativa é composta por três termos:
      1. L_var (Variância): Puxa pixels da mesma instância para o centroide da instância (margem delta_v).
      2. L_dist (Distância): Empurra centroides de instâncias diferentes para longe (margem 2 * delta_d).
      3. L_reg (Regularização): Mantém os centroides próximos da origem para evitar dispersão infinita.
    """
    def __init__(self, delta_v=0.5, delta_d=1.5, alpha=1.0, beta=1.0, gamma=0.001, bce_weight=1.0):
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, targets):
        """
        Parâmetros:
          pred: Tensor (B, 1 + D, H, W) com canal 0 (fg) e canais 1..D (embeddings).
          targets: Tensor (B, H, W) contendo máscaras rotuladas por instância (0: Fundo, 1..N: Instâncias).
        """
        B, C, H, W = pred.shape
        fg_logits = pred[:, 0, :, :]
        embeddings = pred[:, 1:, :, :]

        gt_fg = (targets > 0).float()
        loss_bce = self.bce(fg_logits, gt_fg)

        total_var = torch.tensor(0.0, device=pred.device)
        total_dist = torch.tensor(0.0, device=pred.device)
        total_reg = torch.tensor(0.0, device=pred.device)
        valid_samples = 0

        for b in range(B):
            target_b = targets[b]
            emb_b = embeddings[b]  # (D, H, W)

            unique_labels = torch.unique(target_b)
            unique_labels = unique_labels[unique_labels > 0]
            num_instances = len(unique_labels)

            if num_instances == 0:
                continue

            valid_samples += 1
            centroids = []
            loss_var_b = torch.tensor(0.0, device=pred.device)

            for uid in unique_labels:
                mask_c = (target_b == uid)
                emb_c = emb_b[:, mask_c]  # (D, N_c)
                mu_c = emb_c.mean(dim=1, keepdim=True)  # (D, 1)
                centroids.append(mu_c.squeeze(1))

                # Distância euclidiana com epsilon para estabilidade de gradientes (evita NaN no sqrt)
                diff = emb_c - mu_c
                dist_to_centroid = torch.sqrt(torch.sum(diff ** 2, dim=0) + 1e-8)
                var_c = torch.clamp(dist_to_centroid - self.delta_v, min=0.0) ** 2
                loss_var_b = loss_var_b + var_c.mean()

            loss_var_b = loss_var_b / num_instances
            total_var = total_var + loss_var_b

            mu = torch.stack(centroids, dim=0)  # (num_instances, D)

            # Termo de Regularização: penaliza norma L2 dos centroides
            norm_mu = torch.sqrt(torch.sum(mu ** 2, dim=1) + 1e-8)
            loss_reg_b = norm_mu.mean()
            total_reg = total_reg + loss_reg_b

            # Termo de Distância: empurra centroides distintos para além de 2 * delta_d
            if num_instances > 1:
                diff_centroids = mu.unsqueeze(1) - mu.unsqueeze(0)  # (N, N, D)
                dist_centroids = torch.sqrt(torch.sum(diff_centroids ** 2, dim=2) + 1e-8)
                dist_hinge = torch.clamp(2.0 * self.delta_d - dist_centroids, min=0.0) ** 2

                eye = torch.eye(num_instances, dtype=torch.bool, device=pred.device)
                dist_hinge = dist_hinge.masked_fill(eye, 0.0)

                loss_dist_b = dist_hinge.sum() / (num_instances * (num_instances - 1))
                total_dist = total_dist + loss_dist_b

        if valid_samples > 0:
            total_var = total_var / valid_samples
            total_dist = total_dist / valid_samples
            total_reg = total_reg / valid_samples

        total_loss = (self.bce_weight * loss_bce +
                      self.alpha * total_var +
                      self.beta * total_dist +
                      self.gamma * total_reg)

        breakdown = {
            "loss": total_loss.item(),
            "bce": loss_bce.item(),
            "var": total_var.item(),
            "dist": total_dist.item(),
            "reg": total_reg.item(),
        }

        return total_loss, breakdown


def decodificar_embeddings_trilha_b(fg_prob, embeddings, threshold_fg=0.5,
                                    eps=0.5, min_samples=10, min_instance_size=5,
                                    method="dbscan"):
    """
    Decodifica as instâncias a partir dos embeddings e da máscara de primeiro plano.
    Aplica clustering (DBSCAN por padrão) sobre os pixels de primeiro plano (foreground).

    O clustering na inferência NÃO precisa saber o número de objetos:
    O DBSCAN agrupa pixels contíguos no espaço D-dimensional com densidade suficiente
    delimitada pelo raio eps (tipicamente alinhado com a margem de variância delta_v).
    """
    if embeddings.shape[0] < embeddings.shape[-1]:
        embeddings = np.transpose(embeddings, (1, 2, 0))  # (H, W, D)

    H, W, D = embeddings.shape
    fg_mask = (fg_prob > threshold_fg)

    if not np.any(fg_mask):
        return np.zeros((H, W), dtype=np.int32), 0

    coords = np.argwhere(fg_mask)  # (N_fg, 2)
    fg_embs = embeddings[fg_mask]  # (N_fg, D)

    if method == "dbscan":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(fg_embs)
    elif method == "meanshift":
        clusterer = MeanShift(bandwidth=eps, bin_seeding=True)
        cluster_labels = clusterer.fit_predict(fg_embs)
    else:
        raise ValueError(f"Método de clustering desconhecido: {method}")

    labeled_mask = np.zeros((H, W), dtype=np.int32)
    current_id = 1

    unique_clusters = np.unique(cluster_labels)
    for cid in unique_clusters:
        if cid == -1:  # ruído no DBSCAN
            continue
        inst_coords = coords[cluster_labels == cid]
        if len(inst_coords) >= min_instance_size:
            labeled_mask[inst_coords[:, 0], inst_coords[:, 1]] = current_id
            current_id += 1

    return labeled_mask, current_id - 1


def train_model_trilha_b(model, X_train, y_train, X_val, y_val, device,
                         num_epochs=10, batch_size=8, learning_rate=0.001,
                         delta_v=0.5, delta_d=1.5, alpha=1.0, beta=1.0,
                         gamma=0.001, bce_weight=1.0):
    """
    Treina o modelo U-Net para a Trilha B (Embeddings Discriminativos).
    Monitora a perda total e o detalhamento dos componentes (BCE, Variância, Distância).
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = DiscriminativeLoss(
        delta_v=delta_v, delta_d=delta_d, alpha=alpha,
        beta=beta, gamma=gamma, bce_weight=bce_weight
    )

    num_train = X_train.shape[0]
    num_val = X_val.shape[0]

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_bce = 0.0
        train_var = 0.0
        train_dist = 0.0
        train_batches = int(np.ceil(num_train / batch_size))

        for i in range(train_batches):
            batch_images = X_train[i * batch_size:(i + 1) * batch_size]
            batch_masks = y_train[i * batch_size:(i + 1) * batch_size]

            batch_images_tensor = torch.from_numpy(batch_images).float().permute(0, 3, 1, 2).contiguous().to(device) / 255.0
            batch_masks_tensor = torch.from_numpy(batch_masks).long().contiguous().to(device)

            optimizer.zero_grad()
            outputs = model(batch_images_tensor)
            loss, breakdown = criterion(outputs, batch_masks_tensor)
            loss.backward()
            optimizer.step()

            train_loss += breakdown["loss"] * len(batch_images)
            train_bce += breakdown["bce"] * len(batch_images)
            train_var += breakdown["var"] * len(batch_images)
            train_dist += breakdown["dist"] * len(batch_images)

        train_loss /= num_train
        train_bce /= num_train
        train_var /= num_train
        train_dist /= num_train

        model.eval()
        val_loss = 0.0
        val_batches = int(np.ceil(num_val / batch_size))

        with torch.no_grad():
            for i in range(val_batches):
                batch_images = X_val[i * batch_size:(i + 1) * batch_size]
                batch_masks = y_val[i * batch_size:(i + 1) * batch_size]

                batch_images_tensor = torch.from_numpy(batch_images).float().permute(0, 3, 1, 2).contiguous().to(device) / 255.0
                batch_masks_tensor = torch.from_numpy(batch_masks).long().contiguous().to(device)

                outputs = model(batch_images_tensor)
                loss, breakdown = criterion(outputs, batch_masks_tensor)
                val_loss += breakdown["loss"] * len(batch_images)

        val_loss /= num_val
        print(f"Epoch [{epoch + 1:2d}/{num_epochs:2d}] | Train Loss: {train_loss:.4f} (BCE: {train_bce:.3f}, Var: {train_var:.3f}, Dist: {train_dist:.3f}) | Val Loss: {val_loss:.4f}")


def evaluate_instance_metrics_single_trilha_b(fg_prob, embeddings, mask_gt,
                                             threshold_fg=0.5, eps=0.5, min_samples=10,
                                             min_instance_size=5,
                                             iou_thresholds=np.arange(0.50, 1.00, 0.05),
                                             matching_method="hungarian"):
    """Avalia uma única imagem da Trilha B calculando mAP@[.50:.95] e erro de contagem via DBSCAN."""
    pred_labeled, num_pred = decodificar_embeddings_trilha_b(
        fg_prob, embeddings, threshold_fg=threshold_fg, eps=eps,
        min_samples=min_samples, min_instance_size=min_instance_size
    )

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


def evaluate_model_instances_trilha_b(model, X, y, device,
                                     threshold_fg=0.5, eps=0.5, min_samples=10,
                                     min_instance_size=5,
                                     iou_thresholds=np.arange(0.50, 1.00, 0.05),
                                     matching_method="hungarian"):
    """Avalia o modelo da Trilha B em todo o conjunto de teste em nível de instâncias."""
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
            mask_gt = y[i]

            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).contiguous().to(device) / 255.0
            outputs = model(img_tensor)

            fg_prob = torch.sigmoid(outputs[:, 0, :, :]).squeeze(0).cpu().numpy()
            embeddings = outputs[:, 1:, :, :].squeeze(0).cpu().numpy()  # (D, H, W)

            mAP, count_err, num_gt, num_pred, aps = evaluate_instance_metrics_single_trilha_b(
                fg_prob, embeddings, mask_gt,
                threshold_fg=threshold_fg, eps=eps, min_samples=min_samples,
                min_instance_size=min_instance_size,
                iou_thresholds=iou_thresholds, matching_method=matching_method
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


def plot_trilha_b_predictions(model, X, y, device, num_samples=3,
                              threshold_fg=0.5, eps=0.5, min_samples=10):
    """
    Exibe visualizações completas da Trilha B para amostras de teste:
      1. Imagem original.
      2. Ground Truth de instâncias.
      3. Probabilidade de primeiro plano predita.
      4. Projeção 2D (PCA) dos embeddings colorida pelas instâncias.
      5. Instâncias finais decodificadas por clustering (DBSCAN).
    """
    model.eval()
    with torch.no_grad():
        for i in range(min(num_samples, len(X))):
            img = X[i]
            mask_gt = y[i]

            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).contiguous().to(device) / 255.0
            outputs = model(img_tensor)

            fg_prob = torch.sigmoid(outputs[:, 0, :, :]).squeeze(0).cpu().numpy()
            embeddings = outputs[:, 1:, :, :].squeeze(0).cpu().numpy()  # (D, H, W)

            pred_labeled, num_pred = decodificar_embeddings_trilha_b(
                fg_prob, embeddings, threshold_fg=threshold_fg, eps=eps, min_samples=min_samples
            )
            num_gt = len(np.unique(mask_gt[mask_gt > 0]))

            # Projeção PCA dos embeddings nos pixels de primeiro plano
            fg_mask = (fg_prob > threshold_fg)
            pca_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
            if np.any(fg_mask) and embeddings.shape[0] >= 3:
                fg_embs = embeddings[:, fg_mask].T  # (N_fg, D)
                if len(fg_embs) >= 3:
                    pca = PCA(n_components=3)
                    embs_pca = pca.fit_transform(fg_embs)
                    # Normaliza para [0, 1] em RGB
                    embs_pca = (embs_pca - embs_pca.min(axis=0)) / (embs_pca.max(axis=0) - embs_pca.min(axis=0) + 1e-8)
                    coords = np.argwhere(fg_mask)
                    pca_img[coords[:, 0], coords[:, 1]] = embs_pca

            plt.figure(figsize=(18, 3.5))

            plt.subplot(1, 5, 1)
            plt.imshow(img)
            plt.title(f'Imagem {i+1}')
            plt.axis('off')

            plt.subplot(1, 5, 2)
            plt.imshow(mask_gt, cmap='nipy_spectral')
            plt.title(f'GT ({num_gt} objetos)')
            plt.axis('off')

            plt.subplot(1, 5, 3)
            plt.imshow(fg_prob, cmap='magma', vmin=0, vmax=1)
            plt.title('Prob. Primeiro Plano')
            plt.axis('off')

            plt.subplot(1, 5, 4)
            plt.imshow(pca_img)
            plt.title('PCA 3D dos Embeddings')
            plt.axis('off')

            plt.subplot(1, 5, 5)
            plt.imshow(pred_labeled, cmap='nipy_spectral')
            plt.title(f'Instâncias DBSCAN ({num_pred} pred)')
            plt.axis('off')

            plt.tight_layout()
            plt.show()


def compare_baseline_vs_trilha_b(res_baseline, res_trilha_b, dataset_name="Reais (DSB2018)"):
    """Exibe tabela comparativa e gráficos de desempenho entre a Baseline e a Trilha B."""
    mAP_base = res_baseline["mean_mAP"]
    mAP_b = res_trilha_b["mean_mAP"]

    err_base = res_baseline["mean_count_error"]
    err_b = res_trilha_b["mean_count_error"]

    iou_thresh = res_baseline["iou_thresholds"]
    idx_50 = np.where(np.isclose(iou_thresh, 0.50))[0][0]
    idx_75 = np.where(np.isclose(iou_thresh, 0.75))[0][0]

    ap50_base = res_baseline["mean_aps_per_threshold"][idx_50]
    ap50_b = res_trilha_b["mean_aps_per_threshold"][idx_50]

    ap75_base = res_baseline["mean_aps_per_threshold"][idx_75]
    ap75_b = res_trilha_b["mean_aps_per_threshold"][idx_75]

    print("=" * 77)
    print(f"  COMPARAÇÃO: BASELINE (Parte 1) vs TRILHA B (Parte 2: Embeddings Discriminativos)")
    print(f"  Dataset: {dataset_name}")
    print("=" * 77)
    print(f"{'Métrica':<32} | {'Baseline (Ingênua)':<20} | {'Trilha B (Embeddings)':<20}")
    print("-" * 77)
    print(f"{'mAP@[.50:.95]':<32} | {mAP_base:<20.4f} | {mAP_b:<20.4f}")
    print(f"{'Erro Médio de Contagem (abs)':<32} | {err_base:<20.2f} | {err_b:<20.2f}")
    print(f"{'AP @ IoU=0.50':<32} | {ap50_base:<20.4f} | {ap50_b:<20.4f}")
    print(f"{'AP @ IoU=0.75':<32} | {ap75_base:<20.4f} | {ap75_b:<20.4f}")
    print("=" * 77)

    # Gráfico comparativo de AP por limiar de IoU
    plt.figure(figsize=(12, 4.5))

    plt.subplot(1, 2, 1)
    plt.plot(iou_thresh, res_baseline["mean_aps_per_threshold"], "o-", color="crimson", label=f"Baseline (mAP={mAP_base:.4f})")
    plt.plot(iou_thresh, res_trilha_b["mean_aps_per_threshold"], "s-", color="dodgerblue", label=f"Trilha B: Embeddings (mAP={mAP_b:.4f})")
    plt.title(f"Curva AP vs. Limiar de IoU ({dataset_name})")
    plt.xlabel("Limiar de IoU")
    plt.ylabel("AP Médio")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    labels = ['mAP@[.50:.95]', 'AP@0.50', 'AP@0.75']
    base_vals = [mAP_base, ap50_base, ap75_base]
    b_vals = [mAP_b, ap50_b, ap75_b]

    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width/2, base_vals, width, label='Baseline', color='crimson', alpha=0.85)
    plt.bar(x + width/2, b_vals, width, label='Trilha B (Embeddings)', color='dodgerblue', alpha=0.85)
    plt.title(f"Comparação de Métricas ({dataset_name})")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5, axis='y')
    plt.legend()

    plt.tight_layout()
    plt.show()

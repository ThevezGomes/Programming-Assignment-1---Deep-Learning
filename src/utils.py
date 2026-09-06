import os
import glob
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import watershed
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
      - out_channels: Número de canais de saída (1 para binário, 3 para Trilha A: Fundo/Interior/Fronteira).
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

        # Head modularizado: utiliza a função create_segmentation_head por padrão ou head customizado
        if head is not None:
            self.head = head
        else:
            self.head = create_segmentation_head(in_channels=32, out_channels=out_channels, head_type=head_type)

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
# --- PARTE 2: TRILHA A — FRONTEIRAS E WATERSHED ---
# =============================================================================

def gerar_alvos_trilha_a(instance_masks, border_thickness=1):
    """
    Gera alvos de 3 classes para a Trilha A (Fronteiras e Watershed):
      - Classe 0: Fundo (Background)
      - Classe 1: Interior do núcleo (Marcadores / Seeds para Watershed)
      - Classe 2: Fronteira / Borda entre instâncias

    Cada instância é erodida individualmente para gerar o interior garantindo que
    núcleos vizinhos não se toquem. Para núcleos minúsculos, o centróide é preservado.
    A borda de cada núcleo (e a faixa de contato entre eles) torna-se classe 2.
    """
    num_samples = len(instance_masks)
    h, w = instance_masks.shape[1:3]
    targets_3c = np.zeros((num_samples, h, w), dtype=np.int64)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for idx in range(num_samples):
        mask = instance_masks[idx]
        unique_ids = np.unique(mask[mask > 0])
        interior_mask = np.zeros((h, w), dtype=bool)

        for uid in unique_ids:
            obj = (mask == uid).astype(np.uint8)
            eroded = cv2.erode(obj, kernel, iterations=border_thickness)

            # Se a erosão apagou um núcleo muito pequeno, preserva seu centróide
            if eroded.sum() == 0:
                coords = np.argwhere(obj > 0)
                cy, cx = np.round(coords.mean(axis=0)).astype(int)
                if obj[cy, cx] == 0:
                    cy, cx = coords[len(coords) // 2]
                eroded[cy, cx] = 1

            interior_mask |= (eroded > 0)

        fg_mask = (mask > 0)
        border_mask = fg_mask & (~interior_mask)

        alvo = np.zeros((h, w), dtype=np.int64)
        alvo[border_mask] = 2
        alvo[interior_mask] = 1
        targets_3c[idx] = alvo

    return targets_3c


def calcular_pesos_classes_trilha_a(y_train_3c, power=0.5, epsilon=1e-3, max_weight=5.0):
    """
    Calcula pesos balanceados para as classes (0: Fundo, 1: Interior, 2: Fronteira).
    Usa inverso da frequência suavizado (por padrão com power=0.5, i.e., raiz quadrada),
    evitando que a classe fronteira receba um peso desproporcional que deforme o interior.
    """
    counts = np.bincount(y_train_3c.flatten(), minlength=3).astype(np.float32)
    total = counts.sum()
    freqs = counts / (total + epsilon)
    weights = 1.0 / (freqs ** power + epsilon)
    weights = weights / weights.mean()
    if max_weight is not None:
        weights = np.clip(weights, a_min=None, a_max=max_weight)
        weights = weights / weights.mean()
    return torch.from_numpy(weights).float()


def plot_trilha_a_samples(X, y_3c, num_samples=3):
    """Exibe amostras das imagens originais e dos alvos de 3 classes (Fundo=0, Interior=1, Fronteira=2)."""
    plt.figure(figsize=(4 * num_samples, 4))
    for i in range(min(num_samples, len(X))):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(X[i])
        plt.title(f'Imagem {i+1}')
        plt.axis('off')

        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(y_3c[i], cmap='viridis', vmin=0, vmax=2)
        plt.title('Alvo 3C (0:Bg, 1:Core, 2:Bord)')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


class FocalLossMultiClass(nn.Module):
    """
    Focal Loss multi-classe para balanceamento de classes minoritárias (slides 73-79).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Com gamma=0.0, reduz-se exatamente à Cross-Entropy Ponderada.
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        focal_weight = (1.0 - p) ** self.gamma
        loss = -focal_weight * log_p * targets_one_hot

        if self.weight is not None:
            w = self.weight.to(logits.device).view(1, num_classes, 1, 1)
            loss = loss * w

        if self.reduction == 'mean':
            return loss.sum() / targets_one_hot.sum().clamp(min=1.0)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def decodificar_watershed_trilha_a(pred_probs, threshold_interior=0.4, threshold_fg=0.4, min_marker_size=1):
    """
    Decodifica as probabilidades de 3 classes (Fundo, Interior, Fronteira) em instâncias via Watershed.
    
    Parâmetros:
      - pred_probs: array numpy (3, H, W) ou (H, W, 3) contendo probabilidades softmax.
      - threshold_interior: limiar de confiança para a classe interior ser considerada marcador.
      - threshold_fg: limiar para considerar pixel como foreground.
      - min_marker_size: tamanho mínimo em pixels para um marcador ser mantido (elimina ruídos).
    """
    if pred_probs.shape[0] == 3 and pred_probs.ndim == 3:
        p_bg = pred_probs[0]
        p_interior = pred_probs[1]
        p_border = pred_probs[2]
    elif pred_probs.shape[-1] == 3 and pred_probs.ndim == 3:
        p_bg = pred_probs[:, :, 0]
        p_interior = pred_probs[:, :, 1]
        p_border = pred_probs[:, :, 2]
    else:
        raise ValueError("pred_probs deve conter 3 canais de probabilidades.")

    # 1. Marcadores a partir da probabilidade de interior
    interior_binary = (p_interior > threshold_interior)
    structure = np.ones((3, 3), dtype=int)
    markers, num_markers = label(interior_binary, structure=structure)

    # Remove marcadores espúrios muito pequenos se min_marker_size > 0
    if min_marker_size > 0 and num_markers > 0:
        sizes = ndi.sum(np.ones_like(markers), markers, range(1, num_markers + 1))
        small_mask = np.isin(markers, np.where(sizes < min_marker_size)[0] + 1)
        markers[small_mask] = 0
        markers, num_markers = label(markers > 0, structure=structure)

    if num_markers == 0:
        return np.zeros(p_bg.shape, dtype=np.int32), 0

    # 2. Máscara de primeiro plano (Foreground)
    fg_mask = ((p_interior + p_border) > threshold_fg) | (markers > 0)

    # 3. Superfície topográfica baseada na Transformada de Distância Euclidiana
    # A transformada de distância sobre fg_mask gera vales suaves nos centros e elevação contínua até as bordas,
    # eliminando platôs planos e preservando a fidelidade geométrica dos contornos (alto IoU @ 0.75).
    # O termo + p_border atua como crista extra para reforçar a linha divisória entre células em contato.
    dist_map = ndi.distance_transform_edt(fg_mask)
    surface = -dist_map + (p_border * 1.5)

    # 4. Inundação via Watershed do skimage
    labeled_instances = watershed(surface, markers=markers, mask=fg_mask)

    # Renumera sequencialmente de 1 a N
    unique_ids = np.unique(labeled_instances[labeled_instances > 0])
    final_mask = np.zeros_like(labeled_instances, dtype=np.int32)
    for new_id, old_id in enumerate(unique_ids, start=1):
        final_mask[labeled_instances == old_id] = new_id

    return final_mask, len(unique_ids)


def train_model_trilha_a(model, X_train, y_train_3c, X_val, y_val_3c, device,
                         class_weights=None, gamma=0.0, num_epochs=10,
                         batch_size=8, learning_rate=0.001):
    """Treina o modelo U-Net com saída de 3 classes para a Trilha A usando Cross-Entropy Balanceada / Focal Loss."""
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


def evaluate_instance_metrics_single_trilha_a(pred_probs, mask_gt_instances,
                                             threshold_interior=0.4, threshold_fg=0.4,
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
                                     threshold_interior=0.4, threshold_fg=0.4,
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


def evaluate_instance_level_trilha_a(model, X, y, device, threshold_interior=0.5, threshold_fg=0.5,
                                    min_marker_size=3, matching_method="hungarian", dataset_name="Reais (DSB2018)"):
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


def plot_trilha_a_predictions(model, X_test, y_test_gt, device, num_samples=3,
                             threshold_interior=0.5, threshold_fg=0.5, min_marker_size=3):
    """Visualiza as predições da Trilha A: Imagem, GT, Probabilidade Interior, Probabilidade Fronteira e Watershed."""
    model.eval()

    with torch.no_grad():
        for i in range(min(num_samples, len(X_test))):
            img = X_test[i]
            gt_labeled = y_test_gt[i]

            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).contiguous().to(device) / 255.0
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            labeled_mask, num_instances = decodificar_watershed_trilha_a(
                probs, threshold_interior=threshold_interior, threshold_fg=threshold_fg, min_marker_size=min_marker_size
            )
            num_gt_instances = len(np.unique(gt_labeled[gt_labeled > 0]))

            plt.figure(figsize=(16, 3.5))
            plt.subplot(1, 5, 1)
            plt.imshow(img)
            plt.title(f'Imagem {i+1}')
            plt.axis('off')

            plt.subplot(1, 5, 2)
            plt.imshow(gt_labeled, cmap='nipy_spectral')
            plt.title(f'GT ({num_gt_instances} objetos)')
            plt.axis('off')

            plt.subplot(1, 5, 3)
            plt.imshow(probs[1], cmap='hot', vmin=0, vmax=1)
            plt.title('Prob. Interior (Seeds)')
            plt.axis('off')

            plt.subplot(1, 5, 4)
            plt.imshow(probs[2], cmap='viridis', vmin=0, vmax=1)
            plt.title('Prob. Fronteira')
            plt.axis('off')

            plt.subplot(1, 5, 5)
            plt.imshow(labeled_mask, cmap='nipy_spectral')
            plt.title(f'Watershed ({num_instances} det.)')
            plt.axis('off')

            plt.tight_layout()
            plt.show()


def compare_baseline_vs_trilha_a(results_baseline, results_trilha_a, dataset_name="Reais (DSB2018)"):
    """Compara lado a lado a Baseline da Parte 1 e a Trilha A da Parte 2."""
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

    # Gráficos comparativos de mAP e Erro de Contagem vs Densidade
    num_gt = results_baseline["num_gt_list"]
    mAPs_b = results_baseline["mAP_list"]
    mAPs_a = results_trilha_a["mAP_list"]
    err_b = results_baseline["count_error_list"]
    err_a = results_trilha_a["count_error_list"]

    plt.figure(figsize=(15, 5))

    # Gráfico 1: mAP vs Densidade
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

    # Gráfico 2: Erro de Contagem vs Densidade
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

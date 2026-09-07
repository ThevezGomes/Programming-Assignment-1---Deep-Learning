import os
import glob
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.ndimage as ndi
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment

try:
    from skimage.segmentation import watershed
except ImportError:
    watershed = None

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


def plot_trilha_a_samples(X, y_3c, num_samples=3):
    """Exibe amostras das imagens originais e dos alvos de 3 classes gerados."""
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
    """Implementa a Focal Loss multi-classe balanceada conforme os slides 74 a 79 da aula."""
    def __init__(self, weight=None, gamma=0.0, reduction='mean'):
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
            return loss.sum() / (targets.numel() + 1e-8)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def decodificar_watershed_trilha_a(pred_probs, threshold_interior=0.35, threshold_fg=0.35, min_marker_size=1):
    """Decodifica as probabilidades de 3 classes em instâncias separadas usando o algoritmo Watershed."""
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

    # 1. Marcadores a partir da probabilidade de interior sem atenuação de blur
    interior_binary = (p_interior > threshold_interior)
    markers, num_markers = label(interior_binary)

    if min_marker_size > 1 and num_markers > 0:
        sizes = ndi.sum(np.ones_like(markers), markers, range(1, num_markers + 1))
        small_mask = np.isin(markers, np.where(sizes < min_marker_size)[0] + 1)
        markers[small_mask] = 0
        markers, num_markers = label(markers > 0)

    if num_markers == 0:
        return np.zeros(p_bg.shape, dtype=np.int32), 0

    # 2. Máscara de primeiro plano para limitar a expansão do Watershed
    fg_mask = (p_interior + p_border > threshold_fg) | (markers > 0)

    # 3. Superfície topográfica física com vales nos centros e cristas nas fronteiras
    surface = -p_interior + p_border

    # 4. Inundação por Watershed com skimage ou fallback do OpenCV restrito à máscara
    if watershed is not None:
        labeled_instances = watershed(surface, markers=markers, mask=fg_mask)
    else:
        norm_surf = cv2.normalize(surface, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_3ch = cv2.cvtColor(norm_surf, cv2.COLOR_GRAY2BGR)
        cv2_markers = markers.copy().astype(np.int32)
        cv2.watershed(img_3ch, cv2_markers)
        labeled_instances = np.where((cv2_markers > 0) & fg_mask, cv2_markers, 0)

    unique_ids = np.unique(labeled_instances[labeled_instances > 0])
    final_mask = np.zeros_like(labeled_instances, dtype=np.int32)
    for new_id, old_id in enumerate(unique_ids, start=1):
        final_mask[labeled_instances == old_id] = new_id

    return final_mask, len(unique_ids)


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


def evaluate_instance_metrics_single_trilha_a(pred_probs, mask_gt_instances,
                                             threshold_interior=0.35, threshold_fg=0.35,
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
                                     threshold_interior=0.35, threshold_fg=0.35,
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


def evaluate_instance_level_trilha_a(model, X, y, device, threshold_interior=0.35, threshold_fg=0.35,
                                    min_marker_size=1, matching_method="hungarian", dataset_name="Reais (DSB2018)"):
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
                             threshold_interior=0.35, threshold_fg=0.35, min_marker_size=1):
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


# =============================================================================
# --- PARTE 3: ABLAÇÕES (EIXO 1 - RESOLUÇÃO E EIXO 2 - FUNÇÃO DE PERDA) ---
# =============================================================================


class SegNet(nn.Module):
    """Modelo SegNet com recuperação de resolução por Max Unpooling e índices salvos no pooling (slides 14 e 16)."""
    def __init__(self, out_channels=3):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True)
        )
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True)
        )
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True)
        )
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True)
        )
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        self.center = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True)
        )

        self.unpool4 = nn.MaxUnpool2d(2, 2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True)
        )

        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True)
        )

        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True)
        )

        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, out_channels, 1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1, ind1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2, ind2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3, ind3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4, ind4 = self.pool4(e4)

        c = self.center(p4)

        d4 = self.dec4(self.unpool4(c, ind4))
        d3 = self.dec3(self.unpool3(d4, ind3))
        d2 = self.dec2(self.unpool2(d3, ind2))
        d1 = self.dec1(self.unpool1(d2, ind1))
        return d1



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


def imprimir_tabela_ablação(resultados_dict, titulo="Ablações"):
    """Exibe os resultados da ablação em formato tabular com média e desvio padrão."""
    print(f"\n{'='*75}")
    print(f"   TABELA DE ABLAÇÃO: {titulo.upper()} (2 SEEDS: MÉDIA ± DESVIO)")
    print(f"{'='*75}")
    print(f"{'Configuração':<35} | {'mAP@[.50:.95]':<18} | {'Erro Médio Contagem':<18}")
    print("-" * 75)
    for nome, r in resultados_dict.items():
        map_str = f"{r['mAP_mean']:.4f} ± {r['mAP_std']:.4f}"
        err_str = f"{r['err_mean']:.2f} ± {r['err_std']:.2f}"
        print(f"{nome:<35} | {map_str:<18} | {err_str:<18}")
    print("=" * 75)


# =============================================================================
# --- PARTE 4: INFERÊNCIA EM MOSAICO (SLIDE 83 E FUSÃO DE INSTÂNCIAS) ---
# =============================================================================

class DisjointSetUnion:
    """Estrutura Disjoint Set Union (DSU / Union-Find) para agrupamento de instâncias entre tiles."""
    def __init__(self):
        self.parent = {}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
            return i
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def criar_mosaico_imagens(images, masks, grid_shape=(2, 2)):
    """
    Parte 4 (Item 1): Monta uma imagem grande (mosaico de várias imagens do dataset)
    e reindexa as máscaras de instâncias para garantir unicidade global de cada núcleo.
    """
    rows, cols = grid_shape
    assert len(images) >= rows * cols, f"Imagens insuficientes ({len(images)}) para a grade {grid_shape}"
    h, w, c = images[0].shape
    mosaic_img = np.zeros((h * rows, w * cols, c), dtype=images[0].dtype)
    mosaic_mask = np.zeros((h * rows, w * cols), dtype=np.int32)

    current_max_id = 0
    idx = 0
    for r in range(rows):
        for c_idx in range(cols):
            img_curr = images[idx]
            mask_curr = masks[idx]

            y_start = r * h
            x_start = c_idx * w

            mosaic_img[y_start:y_start + h, x_start:x_start + w] = img_curr

            u_ids = np.unique(mask_curr[mask_curr > 0])
            reindexed_mask = np.zeros_like(mask_curr, dtype=np.int32)
            for new_offset, old_id in enumerate(u_ids, start=1):
                reindexed_mask[mask_curr == old_id] = current_max_id + new_offset
            current_max_id += len(u_ids)

            mosaic_mask[y_start:y_start + h, x_start:x_start + w] = reindexed_mask
            idx += 1

    return mosaic_img, mosaic_mask


def inferencia_mosaico_tiles(model, mosaic_img, tile_size=128, stride=96, device="cpu",
                             threshold_interior=0.35, threshold_fg=0.35, min_marker_size=1):
    """
    Parte 4 (Item 2): Executa inferência em tiles com sobreposição (slide 83 da aula).
    - Agrega média de probabilidades semânticas nos patches sobrepostos ('average the results');
    - Extrai instâncias locais por tile via Watershed (Trilha A) ou limiar ingênuo;
    - Monta o mosaico antes da correção considerando a parte interna de cada patch ('consider the inner part'),
      evidenciando a fratura de objetos que cruzam as fronteiras.
    """
    model.eval()
    H, W = mosaic_img.shape[:2]

    # Grade regular de coordenadas de patches com cobertura completa
    y_starts = list(range(0, H - tile_size + 1, stride))
    if y_starts[-1] + tile_size < H:
        y_starts.append(H - tile_size)
    x_starts = list(range(0, W - tile_size + 1, stride))
    if x_starts[-1] + tile_size < W:
        x_starts.append(W - tile_size)

    y_starts = sorted(list(set(y_starts)))
    x_starts = sorted(list(set(x_starts)))

    # Acumuladores de probabilidades semânticas (Slide 83: 'Average the results')
    dummy_in = torch.zeros(1, 3, tile_size, tile_size, device=device)
    with torch.no_grad():
        dummy_out = model(dummy_in)
    out_channels = dummy_out.shape[1]

    semantic_prob_mosaic = np.zeros((out_channels, H, W), dtype=np.float32)
    weight_mosaic = np.zeros((H, W), dtype=np.float32)

    tiles_data = []
    tile_idx = 0

    with torch.no_grad():
        for y in y_starts:
            for x in x_starts:
                patch = mosaic_img[y:y + tile_size, x:x + tile_size]
                patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1).unsqueeze(0).contiguous().to(device) / 255.0
                logits = model(patch_tensor)

                if out_channels == 3:
                    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    labeled_mask, num_instances = decodificar_watershed_trilha_a(
                        probs, threshold_interior=threshold_interior, threshold_fg=threshold_fg, min_marker_size=min_marker_size
                    )
                else:
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                    if probs.ndim == 2:
                        labeled_mask, num_instances = extract_instances_naive(probs, threshold=0.5)
                        probs = np.expand_dims(probs, 0)
                    else:
                        labeled_mask, num_instances = extract_instances_naive(probs[0], threshold=0.5)

                semantic_prob_mosaic[:, y:y + tile_size, x:x + tile_size] += probs
                weight_mosaic[y:y + tile_size, x:x + tile_size] += 1.0

                tiles_data.append({
                    'tile_id': tile_idx,
                    'y': y, 'x': x,
                    'local_mask': labeled_mask,
                    'num_instances': num_instances,
                    'probs': probs,
                    'center': (y + tile_size / 2.0, x + tile_size / 2.0)
                })
                tile_idx += 1

    semantic_prob_mosaic /= np.maximum(weight_mosaic, 1e-6)

    # Montagem do mosaico antes da correção: Slide 83 ('Consider the inner part')
    pred_mosaic_sem_fusao = np.zeros((H, W), dtype=np.int32)
    tile_id_offsets = {}
    global_counter = 0
    for t in tiles_data:
        tile_id_offsets[t['tile_id']] = global_counter
        global_counter += t['num_instances']

    for r in range(H):
        for c in range(W):
            best_t = None
            min_d2 = 1e9
            for t in tiles_data:
                if t['y'] <= r < t['y'] + tile_size and t['x'] <= c < t['x'] + tile_size:
                    cy, cx = t['center']
                    d2 = (r - cy) ** 2 + (c - cx) ** 2
                    if d2 < min_d2:
                        min_d2 = d2
                        best_t = t
            if best_t is not None:
                loc_id = best_t['local_mask'][r - best_t['y'], c - best_t['x']]
                if loc_id > 0:
                    pred_mosaic_sem_fusao[r, c] = tile_id_offsets[best_t['tile_id']] + loc_id

    # Reindexar IDs contíguos de 1 a N_sem_fusao
    u_sem = np.unique(pred_mosaic_sem_fusao[pred_mosaic_sem_fusao > 0])
    pred_sem_compact = np.zeros_like(pred_mosaic_sem_fusao)
    for new_id, old_id in enumerate(u_sem, start=1):
        pred_sem_compact[pred_mosaic_sem_fusao == old_id] = new_id

    return pred_sem_compact, semantic_prob_mosaic, tiles_data


def fundir_instancias_tiles(tiles_data, image_shape, tile_size=128, min_overlap_iou=0.20, min_relative_inter=0.30):
    """
    Parte 4 (Item 4): Algoritmo de correção por fusão de instâncias entre tiles sobrepostos.
    - Avalia a concordância espacial de cada par de instâncias na faixa de sobreposição (overlap strip);
    - Agrupa os fragmentos correspondentes ao mesmo objeto biológico via Disjoint Set Union (Union-Find);
    - Reconstrói a máscara global contínua unificando as metades cortadas sob o mesmo ID global.
    """
    H, W = image_shape[:2]
    dsu = DisjointSetUnion()

    # Compara pares de tiles adjacentes que possuem sobreposição
    num_tiles = len(tiles_data)
    for i in range(num_tiles):
        for j in range(i + 1, num_tiles):
            tA = tiles_data[i]
            tB = tiles_data[j]

            y_min = max(tA['y'], tB['y'])
            y_max = min(tA['y'] + tile_size, tB['y'] + tile_size)
            x_min = max(tA['x'], tB['x'])
            x_max = min(tA['x'] + tile_size, tB['x'] + tile_size)

            if y_max > y_min and x_max > x_min:
                # Região de sobreposição física entre os tiles A e B
                subA = tA['local_mask'][y_min - tA['y']:y_max - tA['y'], x_min - tA['x']:x_max - tA['x']]
                subB = tB['local_mask'][y_min - tB['y']:y_max - tB['y'], x_min - tB['x']:x_max - tB['x']]

                uA = np.unique(subA[subA > 0])
                uB = np.unique(subB[subB > 0])

                for instA in uA:
                    maskA = (subA == instA)
                    areaA = maskA.sum()
                    for instB in uB:
                        maskB = (subB == instB)
                        inter = np.logical_and(maskA, maskB).sum()
                        if inter > 0:
                            areaB = maskB.sum()
                            union = areaA + areaB - inter
                            iou = inter / union
                            rel_inter = inter / min(areaA, areaB)

                            if iou >= min_overlap_iou or rel_inter >= min_relative_inter:
                                dsu.union((tA['tile_id'], instA), (tB['tile_id'], instB))

    # Mapeia raízes do DSU para IDs globais contíguos (1 .. N_fused)
    root_to_id = {}
    curr_id = 1
    for t in tiles_data:
        for inst in range(1, t['num_instances'] + 1):
            r = dsu.find((t['tile_id'], inst))
            if r not in root_to_id:
                root_to_id[r] = curr_id
                curr_id += 1

    # Reconstrói a máscara global com instâncias fundidas
    pred_mosaic_com_fusao = np.zeros((H, W), dtype=np.int32)
    for t in tiles_data:
        for r_loc in range(tile_size):
            for c_loc in range(tile_size):
                inst = t['local_mask'][r_loc, c_loc]
                if inst > 0:
                    gy = t['y'] + r_loc
                    gx = t['x'] + c_loc
                    gid = root_to_id[dsu.find((t['tile_id'], inst))]
                    pred_mosaic_com_fusao[gy, gx] = gid

    return pred_mosaic_com_fusao


def plot_objeto_fronteira_tiles(mosaic_img, mosaic_gt, pred_sem_fusao, pred_com_fusao,
                                tiles_data, tile_size=128, margin=20):
    """
    Parte 4 (Item 3): Mostra detalhadamente o que acontece com um objeto que cai na fronteira entre dois tiles.
    Exibe a imagem geral com os contornos dos tiles, e um zoom na fronteira comparando:
    Ground Truth, Predição no Tile A, Predição no Tile B, Mosaico Sem Fusão (cortado) e Mosaico Com Fusão (unificado).
    """
    H, W = mosaic_gt.shape

    # Busca automática pelo núcleo com maior fratura na fronteira antes da correção
    unique_gt = np.unique(mosaic_gt[mosaic_gt > 0])
    best_uid = None
    max_split = 1
    for uid in unique_gt:
        mask_u = (mosaic_gt == uid)
        preds_in_u = np.unique(pred_sem_fusao[mask_u])
        preds_in_u = preds_in_u[preds_in_u > 0]
        if len(preds_in_u) > max_split:
            max_split = len(preds_in_u)
            best_uid = uid

    if best_uid is None:
        min_dist_border = 1e9
        for uid in unique_gt:
            coords = np.argwhere(mosaic_gt == uid)
            cy, cx = coords.mean(axis=0)
            for t in tiles_data:
                d = min(abs(cy - t['y']), abs(cy - (t['y'] + tile_size)),
                        abs(cx - t['x']), abs(cx - (t['x'] + tile_size)))
                if d < min_dist_border:
                    min_dist_border = d
                    best_uid = uid

    coords = np.argwhere(mosaic_gt == best_uid)
    y_min_roi = max(0, coords[:, 0].min() - margin)
    y_max_roi = min(H, coords[:, 0].max() + margin + 1)
    x_min_roi = max(0, coords[:, 1].min() - margin)
    x_max_roi = min(W, coords[:, 1].max() + margin + 1)

    cy, cx = coords.mean(axis=0)

    touching_tiles = []
    for t in tiles_data:
        if t['y'] <= cy <= t['y'] + tile_size and t['x'] <= cx <= t['x'] + tile_size:
            touching_tiles.append(t)

    tA = touching_tiles[0] if len(touching_tiles) > 0 else tiles_data[0]
    tB = touching_tiles[1] if len(touching_tiles) > 1 else (tiles_data[1] if len(tiles_data) > 1 else tiles_data[0])

    def get_tile_crop(t):
        full_tile_mask = np.zeros((H, W), dtype=np.int32)
        full_tile_mask[t['y']:t['y'] + tile_size, t['x']:t['x'] + tile_size] = t['local_mask']
        return full_tile_mask[y_min_roi:y_max_roi, x_min_roi:x_max_roi]

    crop_img = mosaic_img[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    crop_gt = mosaic_gt[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    crop_tA = get_tile_crop(tA)
    crop_tB = get_tile_crop(tB)
    crop_sem = pred_sem_fusao[y_min_roi:y_max_roi, x_min_roi:x_max_roi]
    crop_com = pred_com_fusao[y_min_roi:y_max_roi, x_min_roi:x_max_roi]

    ids_sem = np.unique(crop_sem[crop_gt == best_uid])
    ids_sem = ids_sem[ids_sem > 0]
    ids_com = np.unique(crop_com[crop_gt == best_uid])
    ids_com = ids_com[ids_com > 0]

    plt.figure(figsize=(18, 7))

    plt.subplot(2, 3, 1)
    plt.imshow(mosaic_img)
    plt.plot([x_min_roi, x_max_roi, x_max_roi, x_min_roi, x_min_roi],
             [y_min_roi, y_min_roi, y_max_roi, y_max_roi, y_min_roi], 'r-', linewidth=2.5, label='ROI Fronteira')
    for t in tiles_data:
        plt.plot([t['x'], t['x'] + tile_size, t['x'] + tile_size, t['x'], t['x']],
                 [t['y'], t['y'], t['y'] + tile_size, t['y'] + tile_size, t['y']],
                 'w--', alpha=0.4, linewidth=1)
    plt.title(f'Mosaico Completo ({H}x{W}) e Grid de Tiles')
    plt.axis('off')
    plt.legend(loc='lower right')

    plt.subplot(2, 3, 2)
    plt.imshow(crop_img)
    plt.imshow(np.ma.masked_where(crop_gt == 0, crop_gt), cmap='spring', alpha=0.6)
    plt.title(f'Ground Truth (Núcleo #{best_uid}: Unificado)')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(crop_img)
    plt.imshow(np.ma.masked_where(crop_tA == 0, crop_tA), cmap='cool', alpha=0.6)
    plt.title(f'Predição Tile {tA["tile_id"]} (Local)')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(crop_img)
    plt.imshow(np.ma.masked_where(crop_tB == 0, crop_tB), cmap='winter', alpha=0.6)
    plt.title(f'Predição Tile {tB["tile_id"]} (Local)')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(crop_img)
    plt.imshow(np.ma.masked_where(crop_sem == 0, crop_sem), cmap='nipy_spectral', alpha=0.65)
    plt.title(f'Sem Fusão: CORTADO! ({len(ids_sem)} IDs: {ids_sem})')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(crop_img)
    plt.imshow(np.ma.masked_where(crop_com == 0, crop_com), cmap='nipy_spectral', alpha=0.65)
    plt.title(f'Com Fusão: UNIFICADO! (1 ID: {ids_com})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"\n--- [Diagnóstico do Objeto na Fronteira (Item 3)] ---")
    print(f"Objeto GT #{best_uid} localizado na coordenada ({int(cy)}, {int(cx)}):")
    print(f" - Antes da Fusão (Slide 83): Cortado ao meio em {len(ids_sem)} IDs independentes ({ids_sem}).")
    print(f" - Após a Fusão (Item 4): Unificado sob o único ID global ({ids_com[0] if len(ids_com) > 0 else 'N/A'}).")


def avaliar_e_comparar_mosaico(mosaic_gt, pred_sem_fusao, pred_com_fusao,
                               iou_thresholds=np.arange(0.50, 1.00, 0.05), matching_method="hungarian"):
    """
    Parte 4 (Item 4): Avalia e compara quantitativamente o mAP@[.50:.95] e o erro absoluto de contagem
    no mosaico completo antes e depois da correção por fusão de instâncias.
    """
    num_gt = len(np.unique(mosaic_gt[mosaic_gt > 0]))
    num_sem = len(np.unique(pred_sem_fusao[pred_sem_fusao > 0]))
    num_com = len(np.unique(pred_com_fusao[pred_com_fusao > 0]))

    iou_mat_sem = calculate_instance_iou_matrix(mosaic_gt, num_gt, pred_sem_fusao, num_sem)
    iou_mat_com = calculate_instance_iou_matrix(mosaic_gt, num_gt, pred_com_fusao, num_com)

    aps_sem = []
    aps_com = []

    match_fn = match_instances_hungarian if matching_method == "hungarian" else match_instances_greedy

    for t in iou_thresholds:
        tp_sem = match_fn(iou_mat_sem, t)
        denom_sem = num_gt + num_sem - tp_sem
        aps_sem.append(tp_sem / denom_sem if denom_sem > 0 else 0.0)

        tp_com = match_fn(iou_mat_com, t)
        denom_com = num_gt + num_com - tp_com
        aps_com.append(tp_com / denom_com if denom_com > 0 else 0.0)

    aps_sem = np.array(aps_sem, dtype=np.float32)
    aps_com = np.array(aps_com, dtype=np.float32)

    map_sem = float(np.mean(aps_sem))
    map_com = float(np.mean(aps_com))

    err_sem = abs(num_sem - num_gt)
    err_com = abs(num_com - num_gt)

    print(f"\n=======================================================================")
    print(f"      COMPARAÇÃO NO MOSAICO: ANTES VS DEPOIS DA FUSÃO DE INSTÂNCIAS     ")
    print(f"=======================================================================")
    print(f"{'Métrica':<35} | {'Antes da Fusão':<16} | {'Depois da Fusão':<16} | {'Variação':<10}")
    print("-" * 75)
    print(f"{'mAP@[.50:.95]':<35} | {map_sem:<16.4f} | {map_com:<16.4f} | {map_com - map_sem:+10.4f}")
    print(f"{'AP @ IoU=0.50':<35} | {aps_sem[0]:<16.4f} | {aps_com[0]:<16.4f} | {aps_com[0] - aps_sem[0]:+10.4f}")
    print(f"{'AP @ IoU=0.75':<35} | {aps_sem[5]:<16.4f} | {aps_com[5]:<16.4f} | {aps_com[5] - aps_sem[5]:+10.4f}")
    print(f"{'Número de Instâncias Preditas':<35} | {num_sem:<16d} | {num_com:<16d} | {num_com - num_sem:+10d}")
    print(f"{'Instâncias no Ground Truth':<35} | {num_gt:<16d} | {num_gt:<16d} | {'--':<10}")
    print(f"{'Erro Absoluto de Contagem':<35} | {err_sem:<16d} | {err_com:<16d} | {err_com - err_sem:+10d}")
    print("=" * 75)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(iou_thresholds, aps_sem, 'r-o', linewidth=2, label=f'Antes da Fusão (mAP = {map_sem:.4f})')
    plt.plot(iou_thresholds, aps_com, 'b-s', linewidth=2, label=f'Depois da Fusão (mAP = {map_com:.4f})')
    plt.title('Precisão Média (AP) por Limiar de IoU')
    plt.xlabel('Limiar de IoU')
    plt.ylabel('AP')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(1, 2, 2)
    categories = ['mAP@[.50:.95] (x100)', 'Erro de Contagem']
    antes_vals = [map_sem * 100, err_sem]
    depois_vals = [map_com * 100, err_com]

    x_bar = np.arange(len(categories))
    bar_width = 0.35
    plt.bar(x_bar - bar_width / 2, antes_vals, bar_width, label='Antes da Fusão', color='crimson', alpha=0.8)
    plt.bar(x_bar + bar_width / 2, depois_vals, bar_width, label='Depois da Fusão', color='royalblue', alpha=0.8)
    plt.xticks(x_bar, categories)
    plt.ylabel('Valor')
    plt.title('Impacto da Correção por Fusão')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    return {
        "mAP_sem": map_sem, "mAP_com": map_com,
        "err_sem": err_sem, "err_com": err_com,
        "aps_sem": aps_sem, "aps_com": aps_com,
        "num_gt": num_gt, "num_sem": num_sem, "num_com": num_com
    }


def plot_mosaico_completo(mosaic_img, mosaic_gt, pred_sem_fusao, pred_com_fusao):
    """Exibe visualmente o mosaico completo: Imagem, Ground Truth, Predição Sem Fusão e Predição Com Fusão."""
    num_gt = len(np.unique(mosaic_gt[mosaic_gt > 0]))
    num_sem = len(np.unique(pred_sem_fusao[pred_sem_fusao > 0]))
    num_com = len(np.unique(pred_com_fusao[pred_com_fusao > 0]))

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(mosaic_img)
    plt.title(f'Mosaico de Entrada ({mosaic_img.shape[0]}x{mosaic_img.shape[1]})')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(mosaic_gt, cmap='nipy_spectral')
    plt.title(f'Ground Truth ({num_gt} núcleos)')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(pred_sem_fusao, cmap='nipy_spectral')
    plt.title(f'Sem Fusão ({num_sem} det. - Fraturas)')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(pred_com_fusao, cmap='nipy_spectral')
    plt.title(f'Com Fusão ({num_com} det. - Unificado)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



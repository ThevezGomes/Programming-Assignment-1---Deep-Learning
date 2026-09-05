import os
import glob
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.models import ResNet18_Weights


def get_device():
    """
    Retorna o dispositivo acelerado disponível (MPS para Mac, CUDA para GPU NVIDIA, ou CPU).
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


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
    def __init__(self, out_channels=1, pretrained=True, freeze_backbone=True):
        super().__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        self.backbone = backbone
        self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
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
        self.up1 = DecoderBlock(64, 64, 64)
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        input_hw = x.shape[-2:]
        x0 = self.encoder0(x)      # 64 channels, H/4
        x1 = self.encoder1(x0)     # 64 channels, H/4
        x2 = self.encoder2(x1)     # 128 channels, H/8
        x3 = self.encoder3(x2)     # 256 channels, H/16
        x4 = self.encoder4(x3)     # 512 channels, H/32

        x = self.center(x4)
        x = self.up4(x, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        logits = self.head(x)
        return F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False).contiguous()


# --- Geração e Carregamento de Datasets ---

def generate_ellipse(image, center, axes, angle, color=None, thickness=-1):
    if color is None:
        color = (
            int(np.random.randint(0, 256)),
            int(np.random.randint(0, 256)),
            int(np.random.randint(0, 256)),
        )
    return cv2.ellipse(image, center, axes, angle, 0, 360, color, thickness)


def gerar_dataset_elipses(num_images=100, image_size=(128, 128), num_ellipses_range=(5, 20), seed=42):
    """
    Gera um dataset sintético de figuras geométricas (elipses) e suas máscaras binárias.
    """
    np.random.seed(seed)
    images = []
    masks = []

    for _ in range(num_images):
        img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        mask = np.zeros(image_size, dtype=np.uint8)
        num_ellipses = np.random.randint(num_ellipses_range[0], num_ellipses_range[1] + 1)

        for _ in range(num_ellipses):
            center = (np.random.randint(0, image_size[1]), np.random.randint(0, image_size[0]))
            axes = (np.random.randint(5, 20), np.random.randint(5, 20))
            angle = np.random.randint(0, 360)
            color = (
                int(np.random.randint(0, 256)),
                int(np.random.randint(0, 256)),
                int(np.random.randint(0, 256)),
            )

            img = generate_ellipse(img, center, axes, angle, color=color, thickness=-1)
            mask = generate_ellipse(mask, center, axes, angle, color=255, thickness=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)


def carregar_dataset_real(stage1_dir, target_size=(128, 128)):
    """
    Carrega as imagens reais do DSB2018 (stage1_train), combinando as máscaras de núcleos individuais em uma única máscara.
    """
    image_ids = [d for d in os.listdir(stage1_dir) if os.path.isdir(os.path.join(stage1_dir, d))]
    images = []
    masks = []
    
    print(f"Carregando {len(image_ids)} amostras de '{stage1_dir}'...")
    t0 = time.time()
    
    for img_id in image_ids:
        img_folder = os.path.join(stage1_dir, img_id)
        img_paths = glob.glob(os.path.join(img_folder, 'images', '*.png'))
        if not img_paths:
            continue
        
        img = cv2.imread(img_paths[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask_paths = glob.glob(os.path.join(img_folder, 'masks', '*.png'))
        combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for mp in mask_paths:
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            combined_mask = np.maximum(combined_mask, m)
            
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(combined_mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        images.append(img_resized)
        masks.append(mask_resized)
        
    t1 = time.time()
    print(f"Dataset real carregado em {t1 - t0:.2f} segundos!")
    return np.array(images), np.array(masks)


def split_dataset(images, masks, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Divide um conjunto de dados em X_train, y_train, X_val, y_val, X_test, y_test.
    """
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


# --- Treinamento e Métricas de Avaliação ---

def train_model(model, X_train, y_train, X_val, y_val, device, num_epochs=10, batch_size=8, learning_rate=0.001):
    """
    Treina o modelo reportando a perda no conjunto de treino e validação a cada época.
    """
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

            batch_images_tensor = torch.from_numpy(batch_images).float().permute(0, 3, 1, 2).to(device) / 255.0
            batch_masks_tensor = torch.from_numpy(batch_masks).float().unsqueeze(1).to(device) / 255.0

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

                batch_images_tensor = torch.from_numpy(batch_images).float().permute(0, 3, 1, 2).to(device) / 255.0
                batch_masks_tensor = torch.from_numpy(batch_masks).float().unsqueeze(1).to(device) / 255.0

                outputs = model(batch_images_tensor)
                loss = criterion(outputs, batch_masks_tensor)
                val_loss += loss.item() * len(batch_images)

        val_loss /= num_val
        print(f'Epoch [{epoch + 1:2d}/{num_epochs:2d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')


def calculate_dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)


def calculate_iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-6)


def evaluate_model(model, X, y, device, threshold=0.5, batch_size=8):
    """
    Avalia o modelo em um conjunto de dados calculando Mean Dice Coefficient e Mean IoU.
    """
    model.eval()
    dice_list = []
    iou_list = []
    num_samples = X.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))

    with torch.no_grad():
        for b in range(num_batches):
            batch_img = X[b * batch_size:(b + 1) * batch_size]
            batch_mask = y[b * batch_size:(b + 1) * batch_size]

            img_tensor = torch.from_numpy(batch_img).float().permute(0, 3, 1, 2).to(device) / 255.0
            outputs = model(img_tensor)
            preds = (torch.sigmoid(outputs) > threshold).cpu().numpy().squeeze(1)

            for i in range(len(batch_img)):
                y_t = (batch_mask[i] > 127).astype(np.float32)
                y_p = preds[i].astype(np.float32)
                dice_list.append(calculate_dice_coefficient(y_t, y_p))
                iou_list.append(calculate_iou(y_t, y_p))

    mean_dice = np.mean(dice_list)
    mean_iou = np.mean(iou_list)
    return mean_dice, mean_iou


# --- Visualização ---

def plot_dataset_samples(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Exibe amostras dos conjuntos de Treino, Validação e Teste.
    """
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
        plt.imshow(mask, cmap='gray')
        plt.title(f'Máscara ({title})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_predictions(model, X_test, y_test, device, num_samples=5):
    """
    Visualiza as predições do modelo nas amostras do conjunto de teste comparando com o Ground Truth.
    """
    model.eval()

    with torch.no_grad():
        for i in range(min(num_samples, len(X_test))):
            img = X_test[i]
            mask = y_test[i]

            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

            output = model(img_tensor)
            output_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            output_mask_binary = (output_mask > 0.5).astype(np.uint8) * 255

            y_t = (mask > 127).astype(np.float32)
            y_p = (output_mask > 0.5).astype(np.float32)
            dice_val = calculate_dice_coefficient(y_t, y_p)
            iou_val = calculate_iou(y_t, y_p)

            plt.figure(figsize=(10, 3.5))
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title(f'Imagem Teste {i+1}')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap='gray')
            plt.title('Máscara Real (Ground Truth)')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(output_mask_binary, cmap='gray')
            plt.title(f'Predição (Dice: {dice_val:.3f}, IoU: {iou_val:.3f})')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

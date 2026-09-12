import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


def create_segmentation_head(in_channels=32, out_channels=1, head_type="conv1x1", dropout=0.0):
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
        # Camada para não passar a imagem "crua" diretamente para o decoder
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


class SegNet(nn.Module):
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


def _resolve_checkpoint_path(filepath):
    """Garante a resolução correta do caminho caso o comando seja executado da raiz ou de src/."""
    if os.path.isabs(filepath):
        return filepath
    if os.path.basename(os.getcwd()) == "src" and not filepath.startswith(".."):
        return os.path.join("..", filepath)
    return filepath


def salvar_checkpoint(model, filepath="checkpoints/checkpoint.pt", **kwargs):
    """
    Salva os pesos e metadados do modelo treinado em formato de checkpoint PyTorch (.pt).
    Cria automaticamente o diretório se necessário.
    
    Parâmetros:
      - model: Instância do modelo PyTorch (ex.: UNetResNet treinado na Trilha A)
      - filepath: Caminho relativo ou absoluto onde o checkpoint será salvo.
      - **kwargs: Metadados adicionais opcionais (ex: epoch, loss, class_weights, thresholds).
    """
    resolved_path = _resolve_checkpoint_path(filepath)
    os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'architecture': 'UNetResNet',
        'backbone': 'resnet18',
        'out_channels': 3,
        'head_type': 'conv1x1',
        'threshold_interior': 0.35,
        'threshold_fg': 0.35,
    }
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, resolved_path)
    size_mb = os.path.getsize(resolved_path) / (1024 * 1024)
    print(f"Checkpoint salvo com sucesso em: {resolved_path} ({size_mb:.2f} MB)")
    return resolved_path


def carregar_checkpoint(filepath="checkpoints/checkpoint.pt", device=None):
    """
    Carrega o modelo final com os pesos salvos no checkpoint pronto para inferência (eval mode).
    
    Parâmetros:
      - filepath: Caminho para o arquivo .pt
      - device: Dispositivo onde alocar o modelo (cuda, mps ou cpu). Se None, detecta automaticamente.
      
    Retorna:
      - model: Instância de UNetResNet carregada com os pesos e em modo eval().
    """
    resolved_path = _resolve_checkpoint_path(filepath)
    if not os.path.exists(resolved_path):
        # Tenta no caminho alternativo relativo (ex.: se executado de diretório diferente)
        if os.path.exists(filepath):
            resolved_path = filepath
        else:
            raise FileNotFoundError(f"Arquivo de checkpoint não encontrado: {resolved_path} nem {filepath}")
    
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
    checkpoint = torch.load(resolved_path, map_location=device, weights_only=True)
    
    out_channels = 3
    head_type = "conv1x1"
    if isinstance(checkpoint, dict):
        out_channels = checkpoint.get('out_channels', 3)
        head_type = checkpoint.get('head_type', 'conv1x1')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        state_dict = checkpoint
        
    head = create_segmentation_head(in_channels=32, out_channels=out_channels, head_type=head_type)
    model = UNetResNet(head=head, pretrained=False, freeze_backbone=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Modelo carregado com sucesso a partir de '{resolved_path}' no dispositivo: {device}")
    return model


def criar_unet_res18(out_channels=3, pretrained=True, freeze_backbone=False, head_type="conv1x1"):
    """Fábrica modular para instanciar a U-Net ResNet18."""
    h = create_segmentation_head(in_channels=32, out_channels=out_channels, head_type=head_type)
    return UNetResNet(head=h, pretrained=pretrained, freeze_backbone=freeze_backbone)


def criar_segnet(out_channels=3):
    """Fábrica modular para instanciar a SegNet."""
    return SegNet(out_channels=out_channels)

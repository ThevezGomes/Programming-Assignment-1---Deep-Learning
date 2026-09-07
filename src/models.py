import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


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

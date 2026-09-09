import torch
import torch.nn.functional as F
from torch import nn
import segmentation_models_pytorch as smp


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Standard U-Net Architecture for Oil Spill Semantic Segmentation.
    Input shape:  [B, in_channels, H, W]
    Output shape: [B, num_classes, H, W]
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 1,
                 features: list | None = None, classification_head: bool = False):
        if features is None:
            features = [64, 128, 256, 512]
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.classification_head_enabled = classification_head
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        if classification_head:
            self.classification_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(features[-1] * 2, 1),
            )

        # Up part of UNet
        reversed_features = list(reversed(features))
        curr_in = features[-1] * 2
        for feature in reversed_features:
            self.ups.append(
                nn.ConvTranspose2d(curr_in, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
            curr_in = feature

        # Final Convolution
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        bottleneck_output = x
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        segmentation_logits = self.final_conv(x)
        if self.classification_head_enabled:
            return segmentation_logits, self.classification_head(bottleneck_output)
        return segmentation_logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference interface: Takes image tensor [B, C, H, W] and returns probability map [B, 1, H, W] in [0, 1].
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probabilities = torch.sigmoid(logits)
        return probabilities


def get_segmentation_model(
    architecture: str = "unet",
    in_channels: int = 1,
    num_classes: int = 1,
    features: list[int] | None = None,
    classification_head: bool = False,
    encoder_name: str = "efficientnet-b0"
) -> nn.Module:
    """
    Factory function for segmentation models to support architectural expansion.
    """
    arch = architecture.lower()
    if arch == "unet":
        return UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            features=features,
            classification_head=classification_head,
        )
    elif arch == "smp_unet":
        aux_params = dict(pooling='avg', dropout=0.5, classes=1) if classification_head else None
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            aux_params=aux_params
        )
        
        # Monkey patch the predict method to match our expected interface
        def predict(x: torch.Tensor) -> torch.Tensor:
            model.eval()
            with torch.no_grad():
                logits = model.forward(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probabilities = torch.sigmoid(logits)
            return probabilities
        model.predict = predict
        return model
    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Currently supported: ['unet', 'smp_unet']")

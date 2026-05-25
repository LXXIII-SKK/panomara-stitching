#!/usr/bin/env python
"""
PhamHungSon_15_CNN_Network.py

Supervised 4-point homography regression network.

Input:
    x: Tensor [B, 2, H, W]
       channel 0 = patch/image A
       channel 1 = patch/image B

Output:
    pred_norm: Tensor [B, 8]
       normalized corner offsets:
       [dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4] / label_scale

Important:
    This model predicts 4 corner displacements for homography estimation.
    It is not a keypoint detector like SIFT/ORB/SuperPoint.

Upgrade notes:
    input_mode="pairdiff_coord" expands [A, B] into
    [A, B, B-A, |B-A|, x_coord, y_coord]. The extra channels make the model
    more sensitive to geometric displacement, which is exactly what plain CNN
    pooling tends to weaken in homography regression.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Do not generate __pycache__ files
sys.dont_write_bytecode = True


DEFAULT_LABEL_SCALE = 42.0
VALID_INPUT_MODES = ("basic", "coord", "pairdiff", "pairdiff_coord")
VALID_OUTPUT_ACTIVATIONS = ("linear", "tanh")


def denormalize_prediction(pred_norm: torch.Tensor, label_scale: float = DEFAULT_LABEL_SCALE) -> torch.Tensor:
    """
    Converts normalized model output back to pixel displacement values.

    Args:
        pred_norm: [B, 8], predicted offsets in normalized units.
        label_scale: Usually the max synthetic perturbation, e.g. 32 px.

    Returns:
        pred_px: [B, 8], predicted offsets in pixels.
    """
    return pred_norm * float(label_scale)


def train_loss_fn(
    pred_norm: torch.Tensor,
    target_px: torch.Tensor,
    label_scale: float = DEFAULT_LABEL_SCALE,
    beta: float = 0.5,
    corner_weight: float = 0.0,
) -> torch.Tensor:
    """
    Stable training loss for homography corner regression.

    The network predicts normalized offsets, while the dataset labels are kept
    in pixels. This function normalizes labels internally.

    Args:
        pred_norm: [B, 8], model output.
        target_px: [B, 8], ground-truth offsets in pixels.
        label_scale: Scale used to normalize labels.
        beta: SmoothL1 beta.

    Returns:
        corner_weight:
            Optional normalized corner-distance term. This directly optimizes
            the same geometric quantity reported as Mean Corner Error (MCE).

    Returns:
        SmoothL1 loss in normalized-coordinate space, optionally plus MCE loss.
    """
    target_norm = target_px.float() / float(label_scale)
    loss = F.smooth_l1_loss(pred_norm, target_norm, beta=beta)

    if corner_weight > 0:
        corner_diff = (pred_norm - target_norm).view(-1, 4, 2)
        corner_loss = torch.linalg.norm(corner_diff, dim=2).mean()
        loss = loss + float(corner_weight) * corner_loss

    return loss


def rmse_loss_px(
    pred_norm: torch.Tensor,
    target_px: torch.Tensor,
    label_scale: float = DEFAULT_LABEL_SCALE,
) -> torch.Tensor:
    """
    Reporting-only RMSE in pixel space.

    Do not use this as the main training loss unless you specifically want RMSE.
    SmoothL1 is usually more stable for training.
    """
    pred_px = denormalize_prediction(pred_norm, label_scale)
    return torch.sqrt(F.mse_loss(pred_px, target_px.float()))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = True):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=not use_bn)
        ]

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _normalize_input_mode(input_mode: str) -> str:
    mode = str(input_mode).lower()
    if mode not in VALID_INPUT_MODES:
        raise ValueError(f"Unknown input_mode: {input_mode}. Supported: {VALID_INPUT_MODES}")
    return mode


def _normalize_output_activation(output_activation: str) -> str:
    activation = str(output_activation).lower()
    if activation not in VALID_OUTPUT_ACTIVATIONS:
        raise ValueError(
            f"Unknown output_activation: {output_activation}. "
            f"Supported: {VALID_OUTPUT_ACTIVATIONS}"
        )
    return activation


def expanded_input_channels(input_mode: str) -> int:
    mode = _normalize_input_mode(input_mode)
    channels = 2

    if "pairdiff" in mode:
        channels += 2

    if "coord" in mode:
        channels += 2

    return channels


def expand_pair_input(x: torch.Tensor, input_mode: str) -> torch.Tensor:
    """
    Adds geometry-friendly channels without changing the dataset on disk.

    Args:
        x: [B, 2, H, W] normalized image pair.
        input_mode:
            basic          -> [A, B]
            coord          -> [A, B, x, y]
            pairdiff       -> [A, B, B-A, |B-A|]
            pairdiff_coord -> [A, B, B-A, |B-A|, x, y]
    """
    mode = _normalize_input_mode(input_mode)

    if mode == "basic":
        return x

    features = [x]

    if "pairdiff" in mode:
        diff = x[:, 1:2] - x[:, 0:1]
        features.extend([diff, diff.abs()])

    if "coord" in mode:
        batch_size, _, height, width = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coord = torch.stack([xx, yy], dim=0).unsqueeze(0)
        features.append(coord.expand(batch_size, -1, -1, -1))

    return torch.cat(features, dim=1)


def make_output_activation(output_activation: str) -> nn.Module:
    activation = _normalize_output_activation(output_activation)
    if activation == "tanh":
        return nn.Tanh()
    return nn.Identity()


class SuperNet(nn.Module):
    """
    Original-style CNN for 128x128 two-channel homography regression.

    The head is intentionally compact. The older 1024/512-style fully
    connected head can memorize a small synthetic split very quickly, so this
    version keeps most capacity in local convolutional features and uses
    Dropout2d + LayerNorm in the regressor.
    """

    def __init__(
        self,
        dropout: float = 0.5,
        input_mode: str = "basic",
        output_activation: str = "tanh",
    ):
        super().__init__()

        self.input_mode = _normalize_input_mode(input_mode)
        output_activation = _normalize_output_activation(output_activation)
        conv_drop = min(0.35, max(0.0, dropout) * 0.35)
        in_channels = expanded_input_channels(self.input_mode)

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(conv_drop),

            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(conv_drop),

            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(conv_drop),

            ConvBlock(128, 128),
            ConvBlock(128, 128),
        )

        # Pool to 3x3 spatial: FC input = 128 * 3 * 3 = 1152.
        self.pool = nn.AdaptiveAvgPool2d((3, 3))

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 8),
            make_output_activation(output_activation),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = expand_pair_input(x, self.input_mode)
        x = self.features(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x


class ResNetHomography(nn.Module):
    """
    ResNet18 backbone modified for 2-channel input and 8-value homography output.
    """

    def __init__(
        self,
        dropout: float = 0.5,
        input_mode: str = "basic",
        output_activation: str = "tanh",
    ):
        super().__init__()

        self.input_mode = _normalize_input_mode(input_mode)
        output_activation = _normalize_output_activation(output_activation)
        in_channels = expanded_input_channels(self.input_mode)

        try:
            self.resnet = models.resnet18(weights=None)
        except TypeError:
            # Older torchvision compatibility
            self.resnet = models.resnet18(pretrained=False)

        self.resnet.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 8),
            make_output_activation(output_activation),
        )

        # Initialize the new 2-channel input conv and final regressor.
        nn.init.kaiming_normal_(self.resnet.conv1.weight, nonlinearity="relu")

        final_linear = self.resnet.fc[-2] if output_activation == "tanh" else self.resnet.fc[-1]
        nn.init.normal_(final_linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = expand_pair_input(x, self.input_mode)
        return self.resnet(x)


class HomographyModel(nn.Module):
    """
    Wrapper for choosing the backbone.

    Args:
        backbone:
            "ConvNet"  -> SuperNet
            "ResNet18" -> ResNetHomography
    """

    def __init__(
        self,
        backbone: str = "ConvNet",
        dropout: float = 0.5,
        input_mode: str = "basic",
        output_activation: str = "tanh",
    ):
        super().__init__()

        backbone_key = backbone.lower()

        if backbone_key in ["resnet", "resnet18"]:
            self.model = ResNetHomography(
                dropout=dropout,
                input_mode=input_mode,
                output_activation=output_activation,
            )
        elif backbone_key in ["convnet", "supernet"]:
            self.model = SuperNet(
                dropout=dropout,
                input_mode=input_mode,
                output_activation=output_activation,
            )
        else:
            raise ValueError(
                f"Unknown backbone: {backbone}. "
                "Supported values: ConvNet, SuperNet, ResNet18."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

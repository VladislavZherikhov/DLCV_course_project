"""Модель: ResNet из torchvision с новой головой на num_classes и опциональным CBAM.

CBAM: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ChannelAttention(nn.Module):
    """Какие каналы важны: avg- и max-pool по пространству -> общий MLP -> сумма логитов (B, C, 1, 1)."""

    def __init__(self, channels, reduction):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x):
        return self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1))


class SpatialAttention(nn.Module):
    """Где на карте важное: mean и max по каналам -> свёртка k×k -> логиты (B, 1, H, W)."""

    def __init__(self, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        return self.conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1))


class CBAM(nn.Module):
    """Channel attention, затем spatial attention: x = x * M_c(x); x = x * M_s(x).

    identity_init=True: последние слои обоих вниманий инициализируются нулями, а маска считается
    как 2 * sigmoid(z). На старте z = 0 -> маска ровно 1, т.е. CBAM не меняет предобученную сеть,
    и внимание постепенно выучивается. identity_init=False — оригинальная схема из статьи (sigmoid(z)).
    """

    def __init__(self, channels, reduction=16, kernel_size=7, identity_init=True):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)
        self.scale = 2.0 if identity_init else 1.0
        if identity_init:
            nn.init.zeros_(self.channel.mlp[2].weight)
            nn.init.zeros_(self.channel.mlp[2].bias)
            nn.init.zeros_(self.spatial.conv.weight)

    def forward(self, x):
        x = x * (self.scale * torch.sigmoid(self.channel(x)))
        return x * (self.scale * torch.sigmoid(self.spatial(x)))


def add_cbam(model, reduction=16, kernel_size=7, identity_init=True):
    """Вставляет CBAM в каждый residual-блок после последнего BN, перед сложением со skip-связью
    (как в статье). Подходит и для BasicBlock (resnet18/34), и для Bottleneck (resnet50+)."""
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in layer:
            bn_name = 'bn3' if hasattr(block, 'bn3') else 'bn2'
            bn = getattr(block, bn_name)
            setattr(block, bn_name, nn.Sequential(OrderedDict([
                ('bn', bn),
                ('cbam', CBAM(bn.num_features, reduction, kernel_size, identity_init)),
            ])))
    return model


def is_attention_param(name):
    """Параметры attention-модулей (для отдельного learning rate)."""
    return '.cbam.' in name


def get_model(model_cfg, num_classes, pretrained=None):
    """ResNet из torchvision (resnet18/34/50/...) с заменённым fc и опциональным CBAM.

    model_cfg — секция model из конфига (DictConfig или dict из чекпоинта).
    pretrained=None — взять значение из model_cfg.
    """
    name = model_cfg['name']
    if not name.startswith('resnet'):
        raise ValueError(f'Unknown model: {name}')
    if pretrained is None:
        pretrained = model_cfg.get('pretrained', True)

    model = getattr(models, name)(weights='DEFAULT' if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    attention = model_cfg.get('attention', 'none')  # в старых чекпоинтах ключа нет -> baseline
    if attention == 'cbam':
        cbam = model_cfg.get('cbam', {})
        add_cbam(model, cbam.get('reduction', 16), cbam.get('kernel_size', 7), cbam.get('identity_init', True))
    elif attention != 'none':
        raise ValueError(f'Unknown attention: {attention}')
    return model

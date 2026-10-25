"""Модель: ResNet из torchvision с новой головой на num_classes."""
import torch.nn as nn
from torchvision import models


def get_model(name, num_classes, pretrained=True):
    """Любой ResNet из torchvision (resnet18/34/50/...) с заменённым fc."""
    if not name.startswith('resnet'):
        raise ValueError(f'Неизвестная модель: {name}')

    model = getattr(models, name)(weights='DEFAULT' if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

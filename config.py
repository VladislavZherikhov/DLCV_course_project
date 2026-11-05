"""Все параметры эксперимента в одном месте. По умолчанию — ConvNeXt-Tiny @ 384.

Любой параметр можно перекрыть из командной строки в формате секция.ключ=значение:
    python main.py general.exp_name=convnext_e20 training.epochs=20 training.lr=5e-5

Прошлые эксперименты (ResNet18 baseline и ResNet18 + CBAM) воспроизводятся так:
    python main.py general.exp_name=resnet18_224 model.name=resnet18 data.img_size=224 \
        training.batch_size=64 training.lr=3e-4 training.weight_decay=1e-4 training.no_weight_decay_1d=false
    (+ model.attention=cbam для CBAM)
"""
import os

from omegaconf import OmegaConf

# автоопределение среды
ON_KAGGLE = os.path.exists('/kaggle/input')

# порядок важен
CLASSES = ['none', 'smoke', 'fire', 'fire_smoke']

# Локальный путь к распакованному D-Fire — поменяй под себя.
LOCAL_DATA_ROOT = r'C:\data\D-Fire'
KAGGLE_DATA_ROOT = '/kaggle/input'

config = {
    'general': {
        'exp_name': 'convnext_tiny_384',
        'seed': 42,
        # debug: берём по debug_samples картинок на каждый сплит и 1 эпоху —
        # чтобы локально на CPU быстро проверить, что пайплайн не падает.
        'debug': False,
        'debug_samples': 128,
    },
    'paths': {
        'data_root': KAGGLE_DATA_ROOT if ON_KAGGLE else LOCAL_DATA_ROOT,
        'out_dir': '/kaggle/working/outputs' if ON_KAGGLE else './outputs',
    },
    'data': {
        'smoke_id': 0,
        'fire_id': 1,
        'val_size': 0.15,
        'img_size': 384,   # ResNet18-эксперименты: 224
    },
    'model': {
        'name': 'convnext_tiny',  # или resnet18/34/50, convnext_small/base
        'pretrained': True,  # веса ImageNet
        # только для ResNet: 'none' — без внимания, 'cbam' — CBAM в каждом residual-блоке (Woo et al., ECCV 2018)
        'attention': 'none',
        'cbam': {
            'reduction': 16,        # сжатие каналов в MLP channel attention
            'kernel_size': 7,       # размер свёртки spatial attention
            'identity_init': True,  # на старте CBAM = тождественное отображение (не ломает pretrained веса)
        },
    },
    'training': {
        'epochs': 15,
        'batch_size': 32,      # ResNet18 @ 224: 64
        'lr': 1e-4,            # ResNet18: 3e-4
        'weight_decay': 0.05,  # ResNet18: 1e-4
        'no_weight_decay_1d': True,  # не применять weight decay к bias и весам нормализаций (ResNet18: false)
        'attention_lr_mult': 10,  # lr новых attention-модулей = lr * attention_lr_mult
        'mixed_precision': True,  
        'class_weights': True,    # веса классов в CrossEntropy против дисбаланса
        'early_stopping': 5,      
        'monitor': 'f1_macro',    # по этой метрике на val выбирается best.pt
        'num_workers': 4 if ON_KAGGLE else 0,  
        'progress_bar': not ON_KAGGLE,         
    },
    'logging': {
        'log_interval': 50,      
        'n_error_examples': 24,  
        'zip_outputs': True,      
    },
}

config = OmegaConf.create(config)

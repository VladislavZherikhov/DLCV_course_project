"""Все параметры эксперимента в одном месте.

Любой параметр можно перекрыть из командной строки в формате секция.ключ=значение:
    python main.py general.exp_name=resnet18_e20 training.epochs=20 training.lr=1e-4
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
        'exp_name': 'resnet18_224',
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
        'img_size': 224,
    },
    'model': {
        'name': 'resnet18',
        'pretrained': True,  # веса ImageNet
    },
    'training': {
        'epochs': 15,
        'batch_size': 64,
        'lr': 3e-4,
        'weight_decay': 1e-4,
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

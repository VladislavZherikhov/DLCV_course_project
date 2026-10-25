"""Все параметры эксперимента в одном месте.

Любой параметр можно перекрыть из командной строки в формате секция.ключ=значение:
    python main.py general.exp_name=resnet18_e20 training.epochs=20 training.lr=1e-4
"""
import os

from omegaconf import OmegaConf

# Автоопределение среды: на Kaggle датасеты лежат в /kaggle/input (только чтение),
# а писать можно только в /kaggle/working.
ON_KAGGLE = os.path.exists('/kaggle/input')

# Порядок важен: индекс в списке = номер класса.
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
        # Внутри data_root рекурсивно ищутся папки <split>/images и <split>/labels,
        # так что точная вложенность датасета не важна.
        'data_root': KAGGLE_DATA_ROOT if ON_KAGGLE else LOCAL_DATA_ROOT,
        'out_dir': '/kaggle/working/outputs' if ON_KAGGLE else './outputs',
    },
    'data': {
        # В разметке D-Fire (YOLO-формат): 0 = smoke, 1 = fire.
        'smoke_id': 0,
        'fire_id': 1,
        'val_size': 0.15,  # доля train, отрезаемая под валидацию (стратифицированно)
        'img_size': 224,
    },
    'model': {
        'name': 'resnet18',
        'pretrained': True,  # веса ImageNet; на Kaggle нужен включённый Internet
    },
    'training': {
        'epochs': 15,
        'batch_size': 64,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'mixed_precision': True,  # работает только на GPU, на CPU отключается сам
        'class_weights': True,    # веса классов в CrossEntropy против дисбаланса
        'early_stopping': 5,      # стоп, если val f1_macro не растёт N эпох (0 = выкл)
        'monitor': 'f1_macro',    # по этой метрике на val выбирается best.pt
        'num_workers': 4 if ON_KAGGLE else 0,  # на Windows лучше 0
        'progress_bar': not ON_KAGGLE,         # в commit-режиме Kaggle tqdm засоряет лог
    },
    'logging': {
        'log_interval': 50,       # каждые N батчей писать в лог loss / lr / скорость (0 = выкл)
        'n_error_examples': 24,   # сколько самых уверенных ошибок нарисовать в <split>_errors.png
        'zip_outputs': True,      # упаковать папку эксперимента в zip для удобного скачивания
    },
}

config = OmegaConf.create(config)

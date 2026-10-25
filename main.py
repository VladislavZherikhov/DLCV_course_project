"""Точка входа: данные -> модель -> обучение -> оценка лучшей модели на val и test.

Запуск:
    python main.py                                   # параметры из config.py
    python main.py general.debug=true                # быстрый прогон на маленькой выборке
"""
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from config import CLASSES, config as base_config
from data import build_dataframes, get_loaders
from model import get_model
from train import fit, predict
from utils import (append_results, format_metrics, format_time, get_logger, save_evaluation,
                   set_seed, system_info)


def main():
    OmegaConf.set_struct(base_config, True)  # опечатка в ключе из CLI -> ошибка, а не тихий игнор
    config = OmegaConf.merge(base_config, OmegaConf.from_cli())

    set_seed(config.general.seed)
    exp_name = config.general.exp_name + ('_debug' if config.general.debug else '')
    out_dir = os.path.join(config.paths.out_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    logger = get_logger(os.path.join(out_dir, 'train.log'))
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))
    run_start = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Experiment: {exp_name}')
    logger.info(f'Output dir: {out_dir}')
    logger.info(f'System: {json.dumps(system_info())}')
    logger.info(f'Config:\n{OmegaConf.to_yaml(config)}')

    # данные
    start = time.time()
    train_df, val_df, test_df = build_dataframes(config)
    splits = {'train': train_df, 'val': val_df, 'test': test_df}

    distribution = pd.DataFrame({
        name: df['label'].value_counts().reindex(range(len(CLASSES)), fill_value=0)
        for name, df in splits.items()
    })
    distribution.index = CLASSES
    distribution.loc['TOTAL'] = distribution.sum()
    share = (distribution.iloc[:-1] / distribution.iloc[:-1].sum() * 100).round(1).add_suffix(' %')
    logger.info(f'Data root: {config.paths.data_root} (parsed in {time.time() - start:.1f}s)')
    logger.info(f'Class distribution:\n{distribution.join(share).fillna("").to_string()}')
    distribution.to_csv(os.path.join(out_dir, 'class_distribution.csv'))
    pd.concat([df.assign(split=name) for name, df in splits.items()]) \
        .to_csv(os.path.join(out_dir, 'labels.csv'), index=False)

    train_loader, val_loader, test_loader = get_loaders(config, train_df, val_df, test_df)

    # модель и лосс
    model = get_model(config.model.name, len(CLASSES), config.model.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model: {config.model.name} (pretrained={config.model.pretrained}), '
                f'{n_params / 1e6:.2f}M parameters')

    weight = None
    if config.training.class_weights:
        counts = np.bincount(train_df['label'], minlength=len(CLASSES)).clip(min=1)
        weight = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32, device=device)
        logger.info(f'Class weights: {{c: round(float(w), 3) for c, w in zip(CLASSES, weight.cpu())}}')
    criterion = nn.CrossEntropyLoss(weight=weight)

    # обучение
    start = time.time()
    _, best_epoch = fit(config, model, train_loader, val_loader, criterion, device, out_dir, logger)
    train_time = int(time.time() - start)

    # оценка лучшей модели 
    checkpoint = torch.load(os.path.join(out_dir, 'best.pt'), map_location=device)
    model.load_state_dict(checkpoint['model'])
    use_amp = config.training.mixed_precision and device.type == 'cuda'
    logger.info(f'\nLoaded best.pt (epoch {checkpoint["epoch"]}) for final evaluation')

    results = {}
    for name, loader, df in (('val', val_loader, val_df), ('test', test_loader, test_df)):
        start = time.time()
        _, probs, targets = predict(model, loader, device, use_amp, desc=name)
        infer_time = time.time() - start
        metrics, per_class, cm = save_evaluation(targets, probs, CLASSES, out_dir, name, df['path'].tolist(),
                                                 config.logging.n_error_examples)
        results[name] = metrics
        logger.info(
            f'\n==================== {name.upper()} | best epoch {best_epoch} | {len(targets)} images | '
            f'{infer_time:.1f}s ({len(targets) / infer_time:.0f} img/s) ====================\n'
            + '\n'.join(f'  {k:<20} {v:.4f}' for k, v in metrics.items())
            + f'\n\nPer-class metrics:\n{per_class.round(4).to_string()}'
            + f'\n\nConfusion matrix:\n{cm.to_string()}'
        )

    summary = {
        'exp_name': exp_name,
        'model': config.model.name,
        'img_size': config.data.img_size,
        'epochs': config.training.epochs,
        'best_epoch': best_epoch,
        'train_time_s': train_time,
        **{f'val_{k}': round(v, 4) for k, v in results['val'].items()},
        **{f'test_{k}': round(v, 4) for k, v in results['test'].items()},
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as file:
        json.dump(summary, file, indent=2)
    append_results(os.path.join(config.paths.out_dir, 'results.csv'), summary)

    logger.info(f'\nDone in {format_time(time.time() - run_start)} (training {format_time(train_time)}). '
                f'TEST: {format_metrics(results["test"], ["accuracy", "f1_macro", "f1_weighted", "roc_auc_macro"])}')
    logger.info(f'Files: {", ".join(sorted(os.listdir(out_dir)))}')

    if config.logging.zip_outputs:
        archive = shutil.make_archive(out_dir, 'zip', root_dir=out_dir)
        logger.info(f'Archive: {archive} ({os.path.getsize(archive) / 1024 ** 2:.0f} MB)')


if __name__ == '__main__':
    main()

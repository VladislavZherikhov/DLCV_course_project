"""Seed, логирование, метрики, графики, сохранение отчётов."""
import json
import logging
import os
import platform
import random
import sys

import matplotlib
matplotlib.use('Agg')  # без GUI: графики только сохраняем в файлы
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_recall_fscore_support,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_logger(log_path):
    """Логгер, который пишет и в консоль, и в файл."""
    logger = logging.getLogger('fire')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding='utf-8')):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def system_info():
    info = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cpu_count': os.cpu_count(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, gpu_count=torch.cuda.device_count(),
                    gpu_memory_gb=round(props.total_memory / 1024 ** 3, 1), cuda=torch.version.cuda)
    return info


def gpu_memory_gb():
    return torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0


def format_time(seconds):
    seconds = int(seconds)
    return f'{seconds // 3600}h {seconds % 3600 // 60:02d}m {seconds % 60:02d}s' if seconds >= 3600 \
        else f'{seconds // 60}m {seconds % 60:02d}s'


# ------------------------------------------------------------------ метрики

def compute_metrics(y_true, y_prob, num_classes):
    """accuracy + precision/recall/f1/roc_auc в вариантах macro и weighted.

    y_true: (N,) номера классов, y_prob: (N, C) вероятности после softmax.
    """
    y_pred = y_prob.argmax(1)
    labels = list(range(num_classes))
    metrics = {'accuracy': accuracy_score(y_true, y_pred)}

    for avg in ('macro', 'weighted'):
        metrics[f'precision_{avg}'] = precision_score(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        metrics[f'recall_{avg}'] = recall_score(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        metrics[f'f1_{avg}'] = f1_score(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        try:
            # one-vs-rest ROC AUC по каждому классу, затем усреднение
            metrics[f'roc_auc_{avg}'] = roc_auc_score(y_true, y_prob, labels=labels,
                                                      multi_class='ovr', average=avg)
        except ValueError:
            # бывает, если в выборке нет какого-то класса (например, в debug-режиме)
            metrics[f'roc_auc_{avg}'] = float('nan')

    return {k: float(v) for k, v in metrics.items()}


def per_class_metrics(y_true, y_prob, classes):
    """Таблица по классам: precision, recall, f1, ROC AUC и AP (one-vs-rest), support."""
    y_pred = y_prob.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0)

    rows = []
    for i, name in enumerate(classes):
        is_class = y_true == i
        has_both = 0 < is_class.sum() < len(is_class)
        rows.append({
            'class': name,
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'roc_auc': roc_auc_score(is_class, y_prob[:, i]) if has_both else float('nan'),
            'avg_precision': average_precision_score(is_class, y_prob[:, i]) if has_both else float('nan'),
            'support': int(support[i]),
        })
    return pd.DataFrame(rows).set_index('class')


def format_metrics(metrics, keys=None):
    keys = keys or metrics.keys()
    return ' | '.join(f'{k} {metrics[k]:.4f}' for k in keys)


# ------------------------------------------------------------------ графики

def plot_confusion_matrix(cm, classes, path, normalize=False):
    values = cm.astype(float) / cm.sum(1, keepdims=True).clip(min=1) if normalize else cm

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(values, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='True',
           title='Confusion matrix (normalized by true class)' if normalize else 'Confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    threshold = values.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = f'{values[i, j]:.2f}' if normalize else f'{values[i, j]:d}'
            ax.text(j, i, text, ha='center', va='center',
                    color='white' if values[i, j] > threshold else 'black')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_pr_curves(y_true, y_prob, classes, path):
    """ROC и Precision-Recall кривые one-vs-rest для каждого класса."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5))

    for i, name in enumerate(classes):
        is_class = y_true == i
        if not 0 < is_class.sum() < len(is_class):
            continue
        fpr, tpr, _ = roc_curve(is_class, y_prob[:, i])
        ax_roc.plot(fpr, tpr, label=f'{name} (AUC {roc_auc_score(is_class, y_prob[:, i]):.3f})')
        precision, recall, _ = precision_recall_curve(is_class, y_prob[:, i])
        ax_pr.plot(recall, precision, label=f'{name} (AP {average_precision_score(is_class, y_prob[:, i]):.3f})')

    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax_roc.set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curves (one-vs-rest)')
    ax_pr.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall curves (one-vs-rest)')
    for ax in (ax_roc, ax_pr):
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right' if ax is ax_roc else 'lower left')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_examples(paths, y_true, y_prob, classes, path, n=24):
    """Сетка самых уверенных ошибок: модель ошиблась с наибольшей вероятностью."""
    y_pred = y_prob.argmax(1)
    errors = np.where(y_pred != y_true)[0]
    if len(errors) == 0:
        return
    errors = errors[np.argsort(-y_prob[errors, y_pred[errors]])][:n]

    cols = 6
    rows = int(np.ceil(len(errors) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, idx in zip(axes.flat, errors):
        image = Image.open(paths[idx]).convert('RGB')
        image.thumbnail((320, 320))
        ax.imshow(image)
        ax.set_title(f'true: {classes[y_true[idx]]}\npred: {classes[y_pred[idx]]} ({y_prob[idx, y_pred[idx]]:.2f})',
                     fontsize=9, color='crimson')

    fig.suptitle('Most confident errors', fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_history(history, classes, path):
    """Все кривые обучения по эпохам на одной картинке."""
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ep = df['epoch']

    ax = axes[0, 0]
    ax.plot(ep, df['train_loss'], 'o-', label='train')
    ax.plot(ep, df['val_loss'], 'o-', label='val')
    ax.set_title('Loss')

    ax = axes[0, 1]
    for split in ('train', 'val'):
        ax.plot(ep, df[f'{split}_accuracy'], 'o-', label=f'{split} accuracy')
        ax.plot(ep, df[f'{split}_f1_macro'], 's--', label=f'{split} f1_macro')
    ax.set_title('Accuracy / F1 macro: train vs val')

    ax = axes[0, 2]
    for metric in ('precision', 'recall', 'f1'):
        ax.plot(ep, df[f'val_{metric}_macro'], 'o-', label=f'{metric} macro')
        ax.plot(ep, df[f'val_{metric}_weighted'], 's--', label=f'{metric} weighted')
    ax.set_title('Validation precision / recall / F1')

    ax = axes[1, 0]
    ax.plot(ep, df['val_roc_auc_macro'], 'o-', label='macro')
    ax.plot(ep, df['val_roc_auc_weighted'], 's--', label='weighted')
    ax.set_title('Validation ROC AUC')

    ax = axes[1, 1]
    for name in classes:
        ax.plot(ep, df[f'val_f1_{name}'], 'o-', label=name)
    ax.set_title('Validation F1 per class')

    ax = axes[1, 2]
    ax.plot(ep, df['lr'], 'o-')
    ax.set_title('Learning rate')

    for ax in axes.flat:
        ax.set_xlabel('epoch')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ сохранение

def save_evaluation(y_true, y_prob, classes, out_dir, prefix, paths=None, n_error_examples=24):
    """Считает все метрики по сплиту и сохраняет в out_dir:

    <prefix>_metrics.json            — общие метрики (macro / weighted)
    <prefix>_per_class.csv           — метрики по каждому классу
    <prefix>_confusion_matrix.csv    — матрица ошибок таблицей
    <prefix>_report.txt              — всё вышеперечисленное в читаемом виде
    <prefix>_confusion_matrix(_norm).png, <prefix>_roc_pr_curves.png
    <prefix>_predictions.csv, <prefix>_errors.png — если переданы paths
    """
    y_pred = y_prob.argmax(1)
    metrics = compute_metrics(y_true, y_prob, len(classes))
    per_class = per_class_metrics(y_true, y_prob, classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_df = pd.DataFrame(cm, index=[f'true_{c}' for c in classes], columns=[f'pred_{c}' for c in classes])

    with open(os.path.join(out_dir, f'{prefix}_metrics.json'), 'w') as file:
        json.dump(metrics, file, indent=2)
    per_class.to_csv(os.path.join(out_dir, f'{prefix}_per_class.csv'))
    cm_df.to_csv(os.path.join(out_dir, f'{prefix}_confusion_matrix.csv'))

    with open(os.path.join(out_dir, f'{prefix}_report.txt'), 'w', encoding='utf-8') as file:
        file.write(f'{prefix.upper()} ({len(y_true)} images)\n\n')
        file.write('\n'.join(f'{k:<20} {v:.4f}' for k, v in metrics.items()))
        file.write(f'\n\nPer-class metrics:\n{per_class.round(4).to_string()}\n')
        file.write(f'\nConfusion matrix:\n{cm_df.to_string()}\n')

    plot_confusion_matrix(cm, classes, os.path.join(out_dir, f'{prefix}_confusion_matrix.png'))
    plot_confusion_matrix(cm, classes, os.path.join(out_dir, f'{prefix}_confusion_matrix_norm.png'),
                          normalize=True)
    plot_roc_pr_curves(y_true, y_prob, classes, os.path.join(out_dir, f'{prefix}_roc_pr_curves.png'))

    if paths is not None:
        pred_df = pd.DataFrame({'path': paths, 'label': [classes[i] for i in y_true],
                                'pred': [classes[i] for i in y_pred], 'correct': y_true == y_pred})
        for i, c in enumerate(classes):
            pred_df[f'prob_{c}'] = y_prob[:, i]
        pred_df.to_csv(os.path.join(out_dir, f'{prefix}_predictions.csv'), index=False)
        if n_error_examples:
            plot_error_examples(paths, y_true, y_prob, classes,
                                os.path.join(out_dir, f'{prefix}_errors.png'), n_error_examples)

    return metrics, per_class, cm_df


def append_results(results_path, row):
    """Одна строка на эксперимент в общую таблицу результатов."""
    df = pd.DataFrame([row])
    if os.path.exists(results_path):
        df = pd.concat([pd.read_csv(results_path), df], ignore_index=True)
    df.to_csv(results_path, index=False)

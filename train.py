"""Цикл обучения: одна эпоха train, предсказание, fit с сохранением best.pt."""
import os
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from config import CLASSES
from utils import (compute_metrics, format_metrics, format_time, gpu_memory_gb,
                   per_class_metrics, plot_history)

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm не обязателен
    def tqdm(iterable, **kwargs):
        return iterable


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp,
                    epoch, logger, log_interval, progress_bar):
    """Одна эпоха обучения. Каждые log_interval батчей пишет в лог loss, lr и скорость.

    Возвращает (средний loss, вероятности, метки) — метрики на train считаются по ним.
    """
    model.train()
    total_loss, n_samples = 0.0, 0
    probs, targets = [], []
    n_steps = len(loader)
    start = window_start = time.time()
    window_images = 0

    for step, (images, labels) in enumerate(tqdm(loader, desc=f'train ep{epoch}', leave=False,
                                                 disable=not progress_bar), 1):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        n_samples += len(labels)
        window_images += len(labels)
        probs.append(outputs.detach().float().softmax(1).cpu().numpy())
        targets.append(labels.cpu().numpy())

        if log_interval and (step % log_interval == 0 or step == n_steps):
            now = time.time()
            logger.info(f'  [ep {epoch}] step {step:>4}/{n_steps} | loss {loss.item():.4f} '
                        f'(avg {total_loss / n_samples:.4f}) | lr {optimizer.param_groups[0]["lr"]:.2e} | '
                        f'{window_images / (now - window_start):.0f} img/s | '
                        f'elapsed {format_time(now - start)} | gpu mem {gpu_memory_gb():.1f} GB')
            window_start, window_images = now, 0

    return total_loss / n_samples, np.concatenate(probs), np.concatenate(targets)


@torch.no_grad()
def predict(model, loader, device, use_amp=False, criterion=None, progress_bar=False, desc='eval'):
    """Возвращает (средний loss или None, вероятности (N, C), метки (N,))."""
    model.eval()
    total_loss, probs, targets = 0.0, [], []

    for images, labels in tqdm(loader, desc=desc, leave=False, disable=not progress_bar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
        outputs = outputs.float()

        if criterion is not None:
            total_loss += criterion(outputs, labels).item() * len(labels)
        probs.append(outputs.softmax(1).cpu().numpy())
        targets.append(labels.cpu().numpy())

    probs, targets = np.concatenate(probs), np.concatenate(targets)
    loss = total_loss / len(targets) if criterion is not None else None
    return loss, probs, targets


def save_checkpoint(path, model, config, epoch, val_metrics):
    """В чекпоинт кладём конфиг и классы, чтобы локально восстановить модель без лишних вопросов."""
    torch.save({
        'model': model.state_dict(),
        'config': OmegaConf.to_container(config, resolve=True),
        'classes': CLASSES,
        'epoch': epoch,
        'val_metrics': val_metrics,
    }, path)


def fit(config, model, train_loader, val_loader, criterion, device, out_dir, logger):
    tc = config.training
    use_amp = tc.mixed_precision and device.type == 'cuda'
    epochs = 1 if config.general.debug else tc.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=tc.lr / 100)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    logger.info(f'Training: {epochs} epochs x {len(train_loader)} steps, batch {tc.batch_size}, '
                f'AMP {use_amp}, optimizer AdamW, scheduler CosineAnnealingLR')

    history = []
    best_score, best_epoch, epochs_since_improvement = -1.0, 0, 0
    fit_start = time.time()

    for epoch in range(1, epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        logger.info(f'\n========== Epoch {epoch}/{epochs} | lr {lr:.2e} ==========')

        train_loss, train_probs, train_targets = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp,
            epoch, logger, config.logging.log_interval, tc.progress_bar)
        train_time = time.time() - start

        val_loss, val_probs, val_targets = predict(model, val_loader, device, use_amp,
                                                   criterion, tc.progress_bar, desc='val')
        scheduler.step()

        train_metrics = compute_metrics(train_targets, train_probs, len(CLASSES))
        val_metrics = compute_metrics(val_targets, val_probs, len(CLASSES))
        val_per_class = per_class_metrics(val_targets, val_probs, CLASSES)

        epoch_time = time.time() - start
        elapsed = time.time() - fit_start
        eta = elapsed / epoch * (epochs - epoch)

        history.append({
            'epoch': epoch, 'lr': lr, 'epoch_time_s': round(epoch_time, 1), 'gpu_mem_gb': round(gpu_memory_gb(), 2),
            'train_loss': train_loss, **{f'train_{k}': v for k, v in train_metrics.items()},
            'val_loss': val_loss, **{f'val_{k}': v for k, v in val_metrics.items()},
            **{f'val_{m}_{c}': val_per_class.loc[c, m] for c in CLASSES for m in ('precision', 'recall', 'f1')},
        })
        pd.DataFrame(history).to_csv(os.path.join(out_dir, 'log.csv'), index=False)

        logger.info(f'Epoch {epoch}/{epochs} done in {format_time(epoch_time)} '
                    f'(train {format_time(train_time)}, val {format_time(epoch_time - train_time)}) | '
                    f'elapsed {format_time(elapsed)} | ETA {format_time(eta)} | gpu mem {gpu_memory_gb():.1f} GB')
        logger.info(f'  train | loss {train_loss:.4f} | '
                    f'{format_metrics(train_metrics, ["accuracy", "f1_macro", "f1_weighted"])}  (with augmentations)')
        logger.info(f'  val   | loss {val_loss:.4f} | {format_metrics(val_metrics)}')
        logger.info('  val per class:\n' + val_per_class.round(4).to_string())

        score = val_metrics[tc.monitor]
        if score > best_score:
            logger.info(f'  >>> new best val {tc.monitor}: {score:.4f} (was {max(best_score, 0):.4f}) -> saved best.pt')
            best_score, best_epoch, epochs_since_improvement = score, epoch, 0
            save_checkpoint(os.path.join(out_dir, 'best.pt'), model, config, epoch, val_metrics)
        else:
            epochs_since_improvement += 1
            logger.info(f'  no improvement for {epochs_since_improvement} epoch(s), '
                        f'best {tc.monitor} {best_score:.4f} at epoch {best_epoch}')
        save_checkpoint(os.path.join(out_dir, 'last.pt'), model, config, epoch, val_metrics)
        plot_history(history, CLASSES, os.path.join(out_dir, 'training_curves.png'))

        if tc.early_stopping and epochs_since_improvement >= tc.early_stopping:
            logger.info(f'Early stopping: no {tc.monitor} improvement for {tc.early_stopping} epochs')
            break

    logger.info(f'\nTraining finished in {format_time(time.time() - fit_start)}. '
                f'Best val {tc.monitor}: {best_score:.4f} (epoch {best_epoch})')
    return history, best_epoch

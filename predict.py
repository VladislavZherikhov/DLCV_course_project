"""Локальная проверка обученной модели (CPU тоже подходит).
"""
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import FireSmokeDataset, IMG_EXTENSIONS, find_split_dirs, get_transforms, make_split_df
from model import get_model
from train import predict
from utils import format_metrics, save_evaluation


def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg, classes = checkpoint['config'], checkpoint['classes']
    model = get_model(cfg['model']['name'], len(classes), pretrained=False)
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), cfg, classes


def collect_images(path):
    if os.path.isfile(path):
        return [path]
    return sorted(os.path.join(path, name) for name in os.listdir(path)
                  if os.path.splitext(name)[1].lower() in IMG_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='путь к best.pt')
    parser.add_argument('--data_root', help='корень D-Fire для оценки на сплите')
    parser.add_argument('--split', default='test')
    parser.add_argument('--images', help='картинка или папка со своими фото')
    parser.add_argument('--out_dir', help='куда сохранить результаты (по умолчанию рядом с ckpt)')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    if not args.data_root and not args.images:
        parser.error('specify --data_root and/or --images')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg, classes = load_model(args.ckpt, device)
    transform = get_transforms(cfg['data']['img_size'], is_train=False)
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ckpt), 'local_eval')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Model: {cfg["model"]["name"]} | classes: {classes} | device: {device}')

    if args.data_root:
        df = make_split_df(*find_split_dirs(args.data_root, args.split),
                           cfg['data']['smoke_id'], cfg['data']['fire_id'])
        loader = DataLoader(FireSmokeDataset(df, transform), batch_size=args.batch_size)
        _, probs, targets = predict(model, loader, device, progress_bar=True)
        metrics, per_class, cm = save_evaluation(targets, probs, classes, out_dir, args.split,
                                                 df['path'].tolist())
        print(f'\n{args.split.upper()}: {format_metrics(metrics)}\n\n'
              f'{per_class.round(4).to_string()}\n\n{cm.to_string()}')
        print(f'\nSaved to {out_dir}')

    if args.images:
        df = pd.DataFrame({'path': collect_images(args.images)})
        loader = DataLoader(FireSmokeDataset(df, transform), batch_size=args.batch_size)
        _, probs, _ = predict(model, loader, device)

        df['pred'] = [classes[i] for i in probs.argmax(1)]
        for i, c in enumerate(classes):
            df[f'prob_{c}'] = probs[:, i].round(4)
        df.to_csv(os.path.join(out_dir, 'my_images_predictions.csv'), index=False)

        pd.set_option('display.width', 200)
        print(df.assign(path=df['path'].map(os.path.basename)).to_string(index=False))


if __name__ == '__main__':
    main()

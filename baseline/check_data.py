"""Проверка датасета перед обучением.

Печатает, где нашлись картинки и лейблы, распределение по 4 классам, и сохраняет
data_check.png: по 4 примера, где есть только class id 0 и только class id 1, с боксами.
Нужно убедиться, что 0 — дым, 1 — огонь (иначе поменять data.smoke_id / data.fire_id).

    python check_data.py
    python check_data.py paths.data_root=/kaggle/input/d-fire
"""
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from PIL import Image

from config import CLASSES, config as base_config
from data import find_split_dirs, make_split_df


def read_boxes(labels_dir, image_name):
    txt = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')
    if not os.path.exists(txt):
        return []
    with open(txt) as file:
        return [list(map(float, line.split())) for line in file if line.strip()]


def main():
    OmegaConf.set_struct(base_config, True)
    config = OmegaConf.merge(base_config, OmegaConf.from_cli())
    print('data_root:', config.paths.data_root)

    for split in ('train', 'test'):
        images_dir, labels_dir = find_split_dirs(config.paths.data_root, split)
        df = make_split_df(images_dir, labels_dir, config.data.smoke_id, config.data.fire_id)
        counts = df['label'].value_counts().reindex(range(len(CLASSES)), fill_value=0)
        print(f'\n{split}:\n  images: {images_dir}\n  labels: {labels_dir}')
        print(f'  {len(df)} images, {len(os.listdir(labels_dir))} label files')
        for i, name in enumerate(CLASSES):
            print(f'  {name:<11} {counts[i]:>6}  ({counts[i] / len(df) * 100:.1f}%)')

    images_dir, labels_dir = find_split_dirs(config.paths.data_root, 'train')
    names = sorted(os.listdir(images_dir))
    random.Random(0).shuffle(names)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for row, class_id in enumerate((0, 1)):
        picked = []
        for name in names:
            boxes = read_boxes(labels_dir, name)
            if boxes and {int(b[0]) for b in boxes} == {class_id}:
                picked.append((name, boxes))
            if len(picked) == 4:
                break
        for ax, (name, boxes) in zip(axes[row], picked):
            image = Image.open(os.path.join(images_dir, name)).convert('RGB')
            width, height = image.size
            ax.imshow(image)
            for _, xc, yc, w, h in boxes:
                ax.add_patch(patches.Rectangle(((xc - w / 2) * width, (yc - h / 2) * height),
                                               w * width, h * height, fill=False, color='red', linewidth=2))
            ax.set_title(f'only class id {class_id}: {name}')
        for ax in axes[row]:
            ax.axis('off')

    fig.tight_layout()
    fig.savefig('data_check.png', dpi=80)
    print('\nSaved data_check.png')


if __name__ == '__main__':
    main()

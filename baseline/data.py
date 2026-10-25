"""Разметка D-Fire -> 4 класса, Dataset и DataLoader'ы."""
import glob
import os

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import CLASSES

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def label_from_txt(txt_path, smoke_id, fire_id):
    """YOLO txt -> номер класса.

    Пустой/отсутствующий файл -> none, только дым -> smoke,
    только огонь -> fire, и то и другое -> fire_smoke.
    """
    ids = set()
    if os.path.exists(txt_path):
        with open(txt_path) as file:
            ids = {int(float(line.split()[0])) for line in file if line.strip()}

    has_smoke, has_fire = smoke_id in ids, fire_id in ids
    if has_smoke and has_fire:
        return CLASSES.index('fire_smoke')
    if has_fire:
        return CLASSES.index('fire')
    if has_smoke:
        return CLASSES.index('smoke')
    return CLASSES.index('none')


def find_split_dirs(data_root, split):
    """(images_dir, labels_dir) для сплита, поиск внутри data_root на любой глубине.

    Поддерживаются обе раскладки:  <split>/images + <split>/labels
                                    images/<split> + labels/<split>
    """
    layouts = (
        (('**', split, 'images'), lambda d: os.path.join(os.path.dirname(d), 'labels')),
        (('**', 'images', split), lambda d: os.path.join(os.path.dirname(os.path.dirname(d)), 'labels', split)),
    )
    for pattern, to_labels_dir in layouts:
        candidates = sorted(glob.glob(os.path.join(data_root, *pattern), recursive=True))
        if candidates:
            images_dir, labels_dir = candidates[0], to_labels_dir(candidates[0])
            # без папки с лейблами все картинки молча стали бы классом none
            if not os.path.isdir(labels_dir):
                raise FileNotFoundError(f'Found {images_dir}, but labels folder {labels_dir} does not exist.')
            return images_dir, labels_dir

    raise FileNotFoundError(
        f'Neither "{split}/images" nor "images/{split}" found inside {data_root}. '
        f'Check paths.data_root in config.py (or --data_root).'
    )


def make_split_df(images_dir, labels_dir, smoke_id, fire_id):
    """Таблица path,label для одного сплита."""
    rows = []
    for name in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMG_EXTENSIONS:
            continue
        txt_path = os.path.join(labels_dir, stem + '.txt')
        rows.append({
            'path': os.path.join(images_dir, name),
            'label': label_from_txt(txt_path, smoke_id, fire_id),
        })
    return pd.DataFrame(rows)


def stratified_subsample(df, n, seed):
    if len(df) <= n:
        return df
    sub, _ = train_test_split(df, train_size=n, stratify=df['label'], random_state=seed)
    return sub


def build_dataframes(config):
    """train / val / test датафреймы. val отрезается от train стратифицированно."""
    dc = config.data
    root = config.paths.data_root

    full_train = make_split_df(*find_split_dirs(root, 'train'), dc.smoke_id, dc.fire_id)
    test_df = make_split_df(*find_split_dirs(root, 'test'), dc.smoke_id, dc.fire_id)

    train_df, val_df = train_test_split(
        full_train, test_size=dc.val_size, stratify=full_train['label'],
        random_state=config.general.seed,
    )

    if config.general.debug:
        n, seed = config.general.debug_samples, config.general.seed
        train_df = stratified_subsample(train_df, n, seed)
        val_df = stratified_subsample(val_df, n, seed)
        test_df = stratified_subsample(test_df, n, seed)

    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def get_transforms(img_size, is_train):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            # hue не трогаем: цвет — важный признак огня и дыма
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FireSmokeDataset(Dataset):
    """Картинка + номер класса. Без label (для своих фото) возвращает -1."""

    def __init__(self, df, transform):
        self.paths = df['path'].tolist()
        self.labels = df['label'].tolist() if 'label' in df else [-1] * len(df)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(image), self.labels[idx]


def get_loader(df, config, is_train):
    tc = config.training
    return DataLoader(
        FireSmokeDataset(df, get_transforms(config.data.img_size, is_train)),
        batch_size=tc.batch_size,
        shuffle=is_train,
        drop_last=is_train,
        num_workers=tc.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=tc.num_workers > 0,
    )


def get_loaders(config, train_df, val_df, test_df):
    return (get_loader(train_df, config, is_train=True),
            get_loader(val_df, config, is_train=False),
            get_loader(test_df, config, is_train=False))

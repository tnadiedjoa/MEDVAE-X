import json
import os
import random

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import albumentations as A

DATA_ROOT = "/home/infres/yrothlin-24/arcade_challenge_datasets"

_SEG_TRAIN_IMAGES = f"{DATA_ROOT}/dataset_phase_1/segmentation_dataset/seg_train/images"
_SEG_TRAIN_ANN    = f"{DATA_ROOT}/dataset_phase_1/segmentation_dataset/seg_train/annotations/seg_train.json"
_SEG_VAL_IMAGES   = f"{DATA_ROOT}/dataset_phase_1/segmentation_dataset/seg_val/images"
_SEG_VAL_ANN      = f"{DATA_ROOT}/dataset_phase_1/segmentation_dataset/seg_val/annotations/seg_val.json"
_SEG_TEST_IMAGES  = f"{DATA_ROOT}/dataset_final_phase/test_case_segmentation/images"
_SEG_TEST_ANN     = f"{DATA_ROOT}/dataset_final_phase/test_case_segmentation/annotations/instances_default.json"


def _train_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                           border_mode=0, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3),
    ])


def _val_transforms(img_size: int) -> A.Compose:
    return A.Compose([A.Resize(img_size, img_size)])


class ArcadeSegDataset(Dataset):
    """Images ARCADE avec masques de segmentation (26 classes)."""

    def __init__(
        self,
        images_dir: str,
        annotations: str,
        image_ids: list = None,
        img_size: int = 512,
        augment: bool = False,
    ):
        self.images_dir = images_dir
        self.transforms = _train_transforms(img_size) if augment else _val_transforms(img_size)

        with open(annotations) as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = image_ids if image_ids is not None else list(self.images.keys())

        self.annotations = {}
        for ann in coco["annotations"]:
            self.annotations.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id   = self.image_ids[idx]
        img_info = self.images[img_id]

        image = np.array(Image.open(os.path.join(self.images_dir, img_info["file_name"])).convert("L"))

        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in self.annotations.get(img_id, []):
            rle = coco_mask.frPyObjects(ann["segmentation"], h, w)
            binary = coco_mask.decode(coco_mask.merge(rle)).astype(bool)
            mask[binary] = ann["category_id"]

        out = self.transforms(image=image, mask=mask)
        image = torch.tensor(out["image"], dtype=torch.float32).unsqueeze(0) / 255.0
        mask  = torch.tensor(out["mask"],  dtype=torch.long)
        return image, mask


class ArcadeImageDataset(Dataset):
    """Images ARCADE seules (sans masques) — pour le pretraining self-supervised."""

    def __init__(self, images_dir: str, img_size: int = 512, augment: bool = False):
        self.images_dir = images_dir
        self.transforms = _train_transforms(img_size) if augment else _val_transforms(img_size)
        self.files = sorted(
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = np.array(Image.open(os.path.join(self.images_dir, self.files[idx])).convert("L"))
        out   = self.transforms(image=image, mask=np.zeros(image.shape[:2], dtype=np.uint8))
        \
        return torch.tensor(out["image"], dtype=torch.float32).unsqueeze(0) / 127.5 - 1.0


def _split_ids(annotations_path: str, train_ratio: float = 0.8, seed: int = 42):
    with open(annotations_path) as f:
        ids = [img["id"] for img in json.load(f)["images"]]
    random.seed(seed)
    random.shuffle(ids)
    cut = int(len(ids) * train_ratio)
    return ids[:cut], ids[cut:]


def get_pretraining_datasets(
    img_size: int = 512,
    val_ratio: float = 0.1,
    seed: int = 42,
    augment: bool = True,
) -> tuple[Dataset, Dataset]:
    """
    Datasets pour le pretraining self-supervised (à passer aux trainers stage 1 & 2).

    Utilise toutes les images ARCADE (seg_train + seg_val + seg_test) sans masks.
    Renvoie (train_dataset, val_dataset).
    """
    all_dirs = [_SEG_TRAIN_IMAGES, _SEG_VAL_IMAGES, _SEG_TEST_IMAGES]
    all_files = []
    for d in all_dirs:
        all_files += [
            (d, f) for f in sorted(os.listdir(d))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    random.seed(seed)
    random.shuffle(all_files)
    cut = int(len(all_files) * (1 - val_ratio))
    train_files, val_files = all_files[:cut], all_files[cut:]

    class _FileListDataset(Dataset):
        def __init__(self, file_list, transforms):
            self.file_list  = file_list
            self.transforms = transforms

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            d, f  = self.file_list[idx]
            image = np.array(Image.open(os.path.join(d, f)).convert("L"))
            out   = self.transforms(image=image, mask=np.zeros(image.shape[:2], dtype=np.uint8))
            return torch.tensor(out["image"], dtype=torch.float32).unsqueeze(0) / 127.5 - 1.0

    train_ds = _FileListDataset(train_files, _train_transforms(img_size) if augment else _val_transforms(img_size))
    val_ds   = _FileListDataset(val_files,   _val_transforms(img_size))
    return train_ds, val_ds


def get_pretraining_loaders(
    img_size: int = 512,
    batch_size: int = 16,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    seed: int = 42,
    augment: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    DataLoaders pour le pretraining self-supervised (stage 1 & 2).

    Utilise toutes les images ARCADE (seg_train + seg_val + seg_test) sans masks.
    Renvoie (train_loader, val_loader).
    """
    all_dirs = [_SEG_TRAIN_IMAGES, _SEG_VAL_IMAGES, _SEG_TEST_IMAGES]
    all_files = []
    for d in all_dirs:
        all_files += [
            (d, f) for f in sorted(os.listdir(d))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    random.seed(seed)
    random.shuffle(all_files)
    cut = int(len(all_files) * (1 - val_ratio))
    train_files, val_files = all_files[:cut], all_files[cut:]

    class _FileListDataset(Dataset):
        def __init__(self, file_list, transforms):
            self.file_list  = file_list
            self.transforms = transforms

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            d, f   = self.file_list[idx]
            image  = np.array(Image.open(os.path.join(d, f)).convert("L"))
            out    = self.transforms(image=image, mask=np.zeros(image.shape[:2], dtype=np.uint8))
            return torch.tensor(out["image"], dtype=torch.float32).unsqueeze(0) / 127.5 - 1.0

    train_ds = _FileListDataset(train_files, _train_transforms(img_size) if augment else _val_transforms(img_size))
    val_ds   = _FileListDataset(val_files,   _val_transforms(img_size))

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **loader_kw),
        DataLoader(val_ds,   shuffle=False, **loader_kw),
    )


def get_segmentation_loaders(
    img_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    DataLoaders pour le fine-tuning de segmentation supervisé.

    Train / val : split du seg_train (1000 images annotées).
    Test        : seg_test officiel (300 images annotées).
    Renvoie (train_loader, val_loader, test_loader).
    """
    train_ids, val_ids = _split_ids(_SEG_TRAIN_ANN, train_ratio=train_ratio, seed=seed)

    train_ds = ArcadeSegDataset(_SEG_TRAIN_IMAGES, _SEG_TRAIN_ANN, train_ids, img_size, augment=True)
    val_ds   = ArcadeSegDataset(_SEG_TRAIN_IMAGES, _SEG_TRAIN_ANN, val_ids,   img_size, augment=False)
    test_ds  = ArcadeSegDataset(_SEG_TEST_IMAGES,  _SEG_TEST_ANN,  None,      img_size, augment=False)

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **loader_kw),
        DataLoader(val_ds,   shuffle=False, **loader_kw),
        DataLoader(test_ds,  shuffle=False, **loader_kw),
    )

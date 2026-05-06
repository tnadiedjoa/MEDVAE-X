from pathlib import Path
from typing import List, Optional, Tuple
import json
import random

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO


def get_files_labels(data_dir: str) -> Tuple[List[str], str]:
    data_dir = Path(data_dir)

    image_dir = data_dir / "images"
    if not image_dir.exists():
        image_dir = data_dir

    ann_files = list(data_dir.glob("*.json")) + list((data_dir / "annotations").glob("*.json"))

    if len(ann_files) == 0:
        raise FileNotFoundError(f"No annotation json found in {data_dir}")

    ann_path = ann_files[0].as_posix()

    files = sorted([
        p.as_posix()
        for p in image_dir.glob("*")
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ])

    return files, ann_path


def split_files(files: List[str], train_ratio: float = 0.8, seed: int = 42):
    files = list(files)
    random.Random(seed).shuffle(files)

    n_train = int(len(files) * train_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:]

    return train_files, val_files


class ArcadeDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        labels: Optional[str] = None,
        transform=None,
        return_path: bool = False,
        num_classes: int = 26,
    ):
        self.files = files
        self.ann_path = labels
        self.transform = transform
        self.return_path = return_path
        self.num_classes = num_classes

        self.coco = COCO(self.ann_path) if self.ann_path is not None else None

        self.filename_to_imgid = {}
        if self.coco is not None:
            for img_id, img_info in self.coco.imgs.items():
                self.filename_to_imgid[img_info["file_name"]] = img_id

    def __len__(self):
        return len(self.files)

    def _load_image(self, path: str):
        img = Image.open(path).convert("L")
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        return img

    def _load_mask(self, path: str):
        file_name = Path(path).name

        if file_name not in self.filename_to_imgid:
            raise KeyError(f"{file_name} not found in COCO annotations")

        img_id = self.filename_to_imgid[file_name]
        img_info = self.coco.imgs[img_id]

        h = img_info["height"]
        w = img_info["width"]

        mask = np.zeros((h, w), dtype=np.int64)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            category_id = int(ann["category_id"])

            if category_id < 1 or category_id > self.num_classes:
                continue

            ann_mask = self.coco.annToMask(ann).astype(bool)
            mask[ann_mask] = category_id

        return torch.from_numpy(mask).long()

    def __getitem__(self, index):
        path = self.files[index]

        image = self._load_image(path)

        if self.coco is not None:
            mask = self._load_mask(path)
        else:
            mask = torch.empty(0)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if self.return_path:
            return image, mask, path

        return image, mask
    


if __name__ == "__main__":
    dataset_dir = "./arcade_challenge_datasets/dataset_phase_1/stenosis_dataset/sten_train"
    test_dataset_dir = "./arcade_challenge_datasets/dataset_phase_1/stenosis_dataset/sten_val"
    files, ann_path = get_files_labels(dataset_dir)
    train_files, val_files = split_files(files)

    train_dataset = ArcadeDataset(train_files, ann_path)
    val_dataset = ArcadeDataset(val_files, ann_path)

    test_files, test_ann_path = get_files_labels(test_dataset_dir)
    test_dataset = ArcadeDataset(test_files, test_ann_path)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")


    # Plot a sample + mask
    import matplotlib.pyplot as plt
    sample_image, sample_mask = train_dataset[0]
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_image.squeeze(), cmap="gray")
    plt.title("Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(sample_mask, cmap="tab20")
    plt.title("Mask")
    plt.axis("off")
    plt.savefig("sample.png")
    plt.show()

    # Superpose mask + sample
    plt.figure(figsize=(10, 5))
    plt.imshow(sample_image.squeeze(), cmap="gray")
    plt.imshow(sample_mask, cmap="tab20", alpha=0.5)
    plt.title("Image + Mask")
    plt.axis("off")
    plt.savefig("sample_superposed.png")
    plt.show()
import os
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools import mask as coco_mask

# data augmentation pour mask et image 
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.7,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.GaussNoise(p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3),
    ])

def get_val_transforms(img_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
    ])

class ArcadeDataset(Dataset):

    def __init__(
        self,
        images_dir: str,
        annotations: str,
        image_ids: list = None,
        img_size: int = 512,
        augment: bool = False,
    ):
        self.images_dir = images_dir
        self.img_size   = img_size
        self.transforms = get_train_transforms(img_size) if augment \
                  else get_val_transforms(img_size)

        # Charge le JSON 
        with open(annotations, "r") as f:
            coco = json.load(f)

        # Indexe les images par leur ID
        self.images = {img["id"]: img for img in coco["images"]}

        # Filtre sur les IDs demandés si fournis, sinon toutes les images
        if image_ids is not None:
            self.image_ids = image_ids
        else:
            self.image_ids = list(self.images.keys())

        # Regroupe les annotations par image_id
        self.annotations = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id   = self.image_ids[idx]
        img_info = self.images[img_id]

        # charge l'image en numpy ce qui est nécessaire pour les transforms d'albumentations
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image    = np.array(Image.open(img_path).convert("L")) 

        # construit le masque de segmentation à partir des annotations
        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in self.annotations.get(img_id, []):
            category_id  = ann["category_id"]
            segmentation = ann["segmentation"]
            rle          = coco_mask.frPyObjects(segmentation, h, w)
            rle          = coco_mask.merge(rle)
            binary_mask  = coco_mask.decode(rle).astype(bool)
            mask[binary_mask] = category_id

        # applique les transforms (image + masque synchronisés)
        augmented = self.transforms(image=image, mask=mask)
        image     = augmented["image"]
        mask      = augmented["mask"]

        # conversion en tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        mask  = torch.tensor(mask,  dtype=torch.long)

        return image, mask


def split_dataset(
    annotations_path: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list, list]:
    
    with open(annotations_path, "r") as f:
        coco = json.load(f)

    all_ids = [img["id"] for img in coco["images"]]

    random.seed(seed)
    random.shuffle(all_ids)

    split = int(len(all_ids) * train_ratio)
    return all_ids[:split], all_ids[split:]

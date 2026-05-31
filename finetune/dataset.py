import os
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools import mask as coco_mask


class ArcadeDataset(Dataset):

    def __init__(
        self,
        images_dir: str,
        annotations: str,
        image_ids: list = None,
        img_size: int = 512,
    ):
        self.images_dir = images_dir
        self.img_size   = img_size

        # Charge le JSON COCO
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

        # --- Charge l'image ---
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image    = Image.open(img_path).convert("L")  # niveaux de gris
        image    = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        image    = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
        image    = image.unsqueeze(0)  # (1, H, W)

        # --- Construit le masque ---
        h, w  = img_info["height"], img_info["width"]
        mask  = np.zeros((h, w), dtype=np.int64)

        anns = self.annotations.get(img_id, [])
        for ann in anns:
            category_id = ann["category_id"]   # 1 à 26
            segmentation = ann["segmentation"]

            # Convertit le polygone en masque binaire avec pycocotools
            rle        = coco_mask.frPyObjects(segmentation, h, w)
            rle        = coco_mask.merge(rle)
            binary_mask = coco_mask.decode(rle).astype(bool)

            # Les pixels de cette annotation prennent la valeur category_id
            mask[binary_mask] = category_id

        # Redimensionne le masque à img_size
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = torch.tensor(np.array(mask), dtype=torch.long)  # (H, W)

        return image, mask


def split_dataset(
    annotations_path: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list, list]:
    """
    Divise les IDs d'images en train et val.

    Args:
        annotations_path : chemin vers le fichier JSON COCO
        train_ratio      : proportion des images pour le train
        seed             : graine aléatoire pour la reproductibilité

    Returns:
        train_ids, val_ids : listes d'IDs d'images
    """
    with open(annotations_path, "r") as f:
        coco = json.load(f)

    all_ids = [img["id"] for img in coco["images"]]

    random.seed(seed)
    random.shuffle(all_ids)

    split = int(len(all_ids) * train_ratio)
    return all_ids[:split], all_ids[split:]

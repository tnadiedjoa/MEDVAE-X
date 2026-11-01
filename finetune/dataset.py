import os
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools import mask as coco_mask

# data augmentation for mask and image
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2 

def seed_worker_augmentations(worker_id: int) -> None:
    """worker_init_fn: each worker draws its own augmentations at every epoch.

    albumentations >= 2 has its own generator, copied as is into the DataLoader
    workers: without this reseeding, the 4 workers drew the same sequence of
    augmentations, replayed identically at every epoch, and the seed changed nothing.
    info.seed equals (base seed drawn by the main process at every epoch) +
    worker_id: it therefore depends on the experiment seed, the worker and the epoch.
    """
    info = torch.utils.data.get_worker_info()
    seed = info.seed % 2**32
    random.seed(seed)
    np.random.seed(seed)
    transforms = getattr(info.dataset, "transforms", None)
    if transforms is not None and hasattr(transforms, "set_random_seed"):
        transforms.set_random_seed(seed)


class PoissonNoise(A.ImageOnlyTransform):
    """Quantum noise: x ← Poisson(x·s) / s, s drawn in scale_range (small s = strong noise).

    Same model and ranges as the robustness evaluation (medvae_eval/robustness).
    """

    def __init__(self, scale_range=(0.05, 1.0), p: float = 0.5):
        super().__init__(p=p)
        self.scale_range = scale_range

    def get_params(self):
        return {"scale": self.py_random.uniform(*self.scale_range)}

    def apply(self, img, scale=1.0, **params):
        noisy = self.random_generator.poisson(img.astype(np.float32) * scale) / scale
        return np.clip(noisy, 0, 255).astype(np.uint8)


class OpenCVGaussianBlur(A.ImageOnlyTransform):
    """OpenCV Gaussian blur, σ derived from the (odd) kernel size, as in the evaluation."""

    def __init__(self, kernel_range=(3, 31), p: float = 0.5):
        super().__init__(p=p)
        self.kernel_range = kernel_range

    def get_params(self):
        return {"ksize": self.py_random.randint(*self.kernel_range) | 1}

    def apply(self, img, ksize=3, **params):
        return cv2.GaussianBlur(img, (ksize, ksize), 0)


def degradation_transform(p: float) -> A.OneOf:
    """One acquisition degradation, with probability p: Poisson noise, JPEG or blur."""
    return A.OneOf([
        PoissonNoise(scale_range=(0.05, 1.0)),
        A.ImageCompression(quality_range=(5, 95)),
        OpenCVGaussianBlur(kernel_range=(3, 31)),
    ], p=p)


def get_train_transforms(img_size: int = 512, noise_std_range=None, degradation_p=None) -> A.Compose:
    # GaussNoise without parameters: standard deviation of 20 to 44 % of the max with
    # albumentations >= 2 (~84/255, image almost destroyed), against ~3-7/255 in 1.x.
    # noise_std_range (fraction of the max) restores a light noise; None = current behaviour.
    gauss_noise = (A.GaussNoise(std_range=tuple(noise_std_range), p=0.3)
                   if noise_std_range is not None else A.GaussNoise(p=0.3))
    # E15: realistic degradations (Poisson, JPEG, blur); None = no degradation
    degradation = [degradation_transform(degradation_p)] if degradation_p else []
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
        gauss_noise,
        *degradation,
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
        noise_std_range=None,
        degradation_p=None,
    ):
        self.images_dir = images_dir
        self.img_size   = img_size
        self.transforms = get_train_transforms(img_size, noise_std_range, degradation_p) if augment \
                  else get_val_transforms(img_size)

        # Load the JSON 
        with open(annotations, "r") as f:
            coco = json.load(f)

        # Index the images by their ID
        self.images = {img["id"]: img for img in coco["images"]}

        # Keep the requested IDs if given, otherwise all images
        if image_ids is not None:
            self.image_ids = image_ids
        else:
            self.image_ids = list(self.images.keys())

        # Group the annotations by image_id
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

        # load the image as numpy, as required by the albumentations transforms
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image    = np.array(Image.open(img_path).convert("L")) 

        # build the segmentation mask from the annotations
        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in self.annotations.get(img_id, []):
            category_id  = ann["category_id"]
            segmentation = ann["segmentation"]
            rle          = coco_mask.frPyObjects(segmentation, h, w)
            rle          = coco_mask.merge(rle)
            binary_mask  = coco_mask.decode(rle).astype(bool)
            mask[binary_mask] = category_id

        # apply the transforms (image and mask in sync)
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

    random.Random(seed).shuffle(all_ids)   # local generator: no effect on the global random state

    split = int(len(all_ids) * train_ratio)
    return all_ids[:split], all_ids[split:]

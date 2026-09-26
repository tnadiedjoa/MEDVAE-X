"""ArcadeDataset sur un mini-jeu COCO synthétique (pas besoin d'ARCADE)."""

import json

import numpy as np
from PIL import Image

from finetune.dataset import ArcadeDataset, split_dataset


def make_coco(tmp_path, n_images=4, size=64):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    images, annotations = [], []
    for i in range(1, n_images + 1):
        Image.fromarray(np.full((size, size), 100, np.uint8)).save(images_dir / f"{i}.png")
        images.append({"id": i, "file_name": f"{i}.png", "height": size, "width": size})
        # Rectangle de la classe 3 : x 10→30, y 20→40
        annotations.append({"id": i, "image_id": i, "category_id": 3,
                            "segmentation": [[10, 20, 30, 20, 30, 40, 10, 40]]})
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(json.dumps({"images": images, "annotations": annotations}))
    return str(images_dir), str(ann_path)


def test_dataset_returns_normalised_image_and_class_mask(tmp_path):
    images_dir, ann = make_coco(tmp_path)
    image, mask = ArcadeDataset(images_dir, ann, img_size=64, augment=False)[0]

    assert image.shape == (1, 64, 64) and image.dtype.is_floating_point
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert abs(image.mean().item() - 100 / 255) < 1e-4
    assert set(mask.unique().tolist()) == {0, 3}
    assert mask[30, 20] == 3 and mask[5, 5] == 0


def test_split_is_deterministic_and_disjoint(tmp_path):
    _, ann = make_coco(tmp_path, n_images=10)
    train, val = split_dataset(ann, train_ratio=0.8, seed=42)
    assert (train, val) == split_dataset(ann, train_ratio=0.8, seed=42)
    assert len(train) == 8 and len(val) == 2 and not set(train) & set(val)


def test_gauss_noise_strength_is_configurable():
    from finetune.dataset import get_train_transforms

    image = np.full((256, 256), 128, np.uint8)

    def noise_std(std_range):
        noise = next(t for t in get_train_transforms(256, std_range).transforms
                     if type(t).__name__ == "GaussNoise")
        noise.p = 1.0
        return noise(image=image)["image"].astype(float).std()

    assert noise_std((0.0124, 0.0277)) < 10        # ~3-7 niveaux sur 255 : bruit léger
    assert noise_std(None) > 30                    # défaut albumentations >= 2 : très fort

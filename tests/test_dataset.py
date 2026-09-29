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


def test_degradation_augmentation_is_optional_and_degrades():
    from finetune.dataset import degradation_transform, get_train_transforms

    names = lambda t: [type(x).__name__ for x in t.transforms]   # noqa: E731
    assert "OneOf" not in names(get_train_transforms(64, (0.0124, 0.0277)))
    assert "OneOf" in names(get_train_transforms(64, (0.0124, 0.0277), degradation_p=0.3))

    rng = np.random.default_rng(0)
    image = (rng.random((128, 128)) * 200 + 20).astype(np.uint8)   # texture : le flou la lisse
    for member in degradation_transform(1.0).transforms:
        member.p = 1.0
        out = member(image=image)["image"]
        name = type(member).__name__
        assert out.shape == image.shape and out.dtype == np.uint8
        assert np.abs(out.astype(float) - image).mean() > 1, name          # l'image change
        assert abs(out.mean() - image.mean()) < 10, name                   # mais reste la même image
    blur = degradation_transform(1.0).transforms[-1]
    blur.p = 1.0
    assert blur(image=image)["image"].std() < image.std()                  # le flou lisse


def test_worker_augmentations_differ_between_workers_and_epochs(tmp_path):
    """Sans réensemencement, albumentations >= 2 rejouait la même suite dans chaque worker."""
    import hashlib

    import torch
    from torch.utils.data import DataLoader

    from finetune.dataset import seed_worker_augmentations

    images_dir, ann = make_coco(tmp_path, n_images=16)
    ds = ArcadeDataset(images_dir, ann, img_size=64, augment=True, noise_std_range=(0.0124, 0.0277))

    def epoch_hashes(loader):
        return [hashlib.md5(x.numpy().tobytes()).hexdigest() for b, _ in loader for x in b]

    torch.manual_seed(0)
    loader = DataLoader(ds, batch_size=2, num_workers=2, worker_init_fn=seed_worker_augmentations)
    e0, e1 = epoch_hashes(loader), epoch_hashes(loader)
    # toutes les images sont identiques : seules les augmentations les distinguent
    assert e0[:2] != e0[2:4]                  # worker 0 (batch 0) ≠ worker 1 (batch 1)
    assert e0 != e1                           # une autre suite à l'epoch suivante
    torch.manual_seed(0)
    assert epoch_hashes(loader) == e0         # reproductible pour un même seed

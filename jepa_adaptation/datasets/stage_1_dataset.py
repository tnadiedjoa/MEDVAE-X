import os
import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
from torchvision import transforms
import medmnist
from medmnist import INFO


MEDMNIST_2D_DATASETS = [
    "pathmnist",
    "dermamnist",
    "pneumoniamnist",
    "bloodmnist",
    "organcmnist",
    "organsmnist",
]


class _ImageOnlyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img


def _build_transform(size, as_rgb):
    if as_rgb:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Resize((size, size), antialias=True),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def _subsample(dataset, fraction, seed):
    if fraction >= 1.0:
        return dataset
    n_keep = max(1, int(round(len(dataset) * fraction)))
    n_drop = len(dataset) - n_keep
    generator = torch.Generator().manual_seed(seed)
    keep, _ = random_split(dataset, [n_keep, n_drop], generator=generator)
    return keep


def build_stage_1_dataset(
    split="train",
    root="~/.medmnist",
    size=224,
    as_rgb=False,
    download=True,
    fraction=1.0,
    seed=42,
):
    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True)
    transform = _build_transform(size, as_rgb)

    datasets = []
    for name in MEDMNIST_2D_DATASETS:
        info = INFO[name]
        DataClass = getattr(medmnist, info["python_class"])
        kwargs = dict(
            split=split,
            transform=transform,
            download=download,
            root=root,
            as_rgb=as_rgb,
        )
        try:
            try:
                ds = DataClass(size=size, **kwargs)
            except TypeError:
                ds = DataClass(**kwargs)
            datasets.append(_subsample(_ImageOnlyDataset(ds), fraction, seed))
        except Exception as e:
            print(f"[stage_1_dataset] skip {name}/{split}: {e}")

    if not datasets:
        raise RuntimeError("No MedMNIST stage-1 datasets could be loaded.")

    return ConcatDataset(datasets)


if __name__ == "__main__":
    ds = build_stage_1_dataset(split="train", size=64, download=True, fraction=0.1)
    print(f"Stage-1 train samples : {len(ds)}")
    print(f"Image shape           : {ds[0].shape}")


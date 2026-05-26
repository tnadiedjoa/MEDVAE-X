import os
import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
from torchvision import transforms
import medmnist
from medmnist import INFO


MEDMNIST_2D_DATASETS = [
    "pathmnist",
    "chestmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "breastmnist",
    "bloodmnist",
    "tissuemnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
]


class _ImageOnlyWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img


def build_medmnist_pretraining_dataset(
    root: str = "~/.medmnist",
    splits: list = None,
    size: int = 224,
    as_rgb: bool = False,
    download: bool = True,
) -> ConcatDataset:

    if splits is None:
        splits = ["train", "val", "test"]

    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True)

    if as_rgb:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Resize((size, size), antialias=True),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    datasets = []
    for name in MEDMNIST_2D_DATASETS:
        info = INFO[name]
        DataClass = getattr(medmnist, info["python_class"])
        for split in splits:
            try:
                ds = DataClass(
                    split=split,
                    transform=transform,
                    download=download,
                    root=root,
                    as_rgb=as_rgb,
                )
                datasets.append(_ImageOnlyWrapper(ds))
            except Exception as e:
                print(f"[pretraining_dataset] skip {name}/{split}: {e}")

    if not datasets:
        raise RuntimeError("No MedMNIST datasets could be loaded.")

    return ConcatDataset(datasets)


def get_pretraining_splits(
    root: str = "~/.medmnist",
    data_fraction: float = 1.0,
    phase1_ratio: float = 0.7,
    splits: list = None,
    size: int = 64,
    as_rgb: bool = False,
    download: bool = True,
    seed: int = 42,
):
    full = build_medmnist_pretraining_dataset(
        root=root,
        splits=splits,
        size=size,
        as_rgb=as_rgb,
        download=download,
    )

    generator = torch.Generator().manual_seed(seed)

    if data_fraction < 1.0:
        n_keep = max(1, round(len(full) * data_fraction))
        n_drop = len(full) - n_keep
        subset, _ = random_split(full, [n_keep, n_drop], generator=generator)
    else:
        subset = full

    n_total = len(subset)
    n_phase1 = round(n_total * phase1_ratio)
    n_phase2 = n_total - n_phase1

    phase1, phase2 = random_split(subset, [n_phase1, n_phase2], generator=generator)

    return phase1, phase2


if __name__ == "__main__":
    phase1, phase2 = get_pretraining_splits(data_fraction=0.1, phase1_ratio=0.7, size=64, download=True)
    print(f"Phase-1 samples : {len(phase1)}")
    print(f"Phase-2 samples : {len(phase2)}")
    print(f"Total kept      : {len(phase1) + len(phase2)}")
    sample = phase1[0]
    print(f"Image shape     : {sample.shape}")

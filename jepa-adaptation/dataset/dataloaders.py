from torch.utils.data import DataLoader
from .dataset import get_files_labels, ArcadeDataset

STEN_TRAIN_PATH = "./arcade_challenge_datasets/dataset_phase_1/stenosis_dataset/sten_train"
STEN_VAL_PATH = "./arcade_challenge_datasets/dataset_phase_1/stenosis_dataset/sten_val"
STEN_TEST_PATH = "./arcade_challenge_datasets/dataset_final_phase/test_cases_stenosis"  

SEG_TRAIN_PATH = "./arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train"
SEG_VAL_PATH = "./arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_val"
SEG_TEST_PATH = "./arcade_challenge_datasets/dataset_final_phase/test_case_segmentation"

def get_loaders(batch_size=4, num_workers=0, pin_memory=True):
    sten_train_files, sten_train_ann = get_files_labels(STEN_TRAIN_PATH)
    sten_val_files, sten_val_ann = get_files_labels(STEN_VAL_PATH)
    sten_test_files, sten_test_ann = get_files_labels(STEN_TEST_PATH)

    seg_train_files, seg_train_ann = get_files_labels(SEG_TRAIN_PATH)
    seg_val_files, seg_val_ann = get_files_labels(SEG_VAL_PATH)
    seg_test_files, seg_test_ann = get_files_labels(SEG_TEST_PATH)

    datasets = {
        "stenosis_train": ArcadeDataset(sten_train_files, labels=sten_train_ann),
        "stenosis_val": ArcadeDataset(sten_val_files, labels=sten_val_ann),
        "stenosis_test": ArcadeDataset(sten_test_files, labels=sten_test_ann),
        "segmentation_train": ArcadeDataset(seg_train_files, labels=seg_train_ann),
        "segmentation_val": ArcadeDataset(seg_val_files, labels=seg_val_ann),
        "segmentation_test": ArcadeDataset(seg_test_files, labels=seg_test_ann),
    }

    loaders = {
        "stenosis_train": DataLoader(
            datasets["stenosis_train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "stenosis_val": DataLoader(
            datasets["stenosis_val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "stenosis_test": DataLoader(
            datasets["stenosis_test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "segmentation_train": DataLoader(
            datasets["segmentation_train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "segmentation_val": DataLoader(
            datasets["segmentation_val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "segmentation_test": DataLoader(
            datasets["segmentation_test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    return loaders


if __name__ == "__main__":
    loaders = get_loaders(batch_size=8)
    for batch in loaders["stenosis_train"]:
        images, masks = batch
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of masks shape: {masks.shape}")
        break
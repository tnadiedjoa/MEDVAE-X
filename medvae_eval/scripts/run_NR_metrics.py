import os
from torchmetrics.image.arniqa import ARNIQA
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import glob

# Sorties dans <repo>/medvae_eval/outputs/, quel que soit le dossier courant
_OUTPUTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

# Racine du dataset ARCADE : $ARCADE_ROOT si défini, sinon <repo>/data/arcade
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCADE_ROOT = os.environ.get("ARCADE_ROOT", os.path.join(_REPO_ROOT, "data", "arcade"))
DATASET_PATH = f"{ARCADE_ROOT}/dataset_phase_1/segmentation_dataset/seg_train"

image_paths = sorted(glob.glob(f"{DATASET_PATH}/**/images/*", recursive=True))
device = "cuda" if torch.cuda.is_available() else "cpu"
metric = ARNIQA(regressor_dataset="koniq10k", normalize=True).to(device)
transform = T.ToTensor()

results = []
for idx, img_path in enumerate(image_paths, start=1):
    img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        score = metric(img).item()
    print(f"{idx}: {score:.4f}")
    results.append({"image": idx, "arniqa_score": score})

pd.DataFrame(results).to_csv(f"{_OUTPUTS}/metrics/arniqa_scores.csv", index=False)

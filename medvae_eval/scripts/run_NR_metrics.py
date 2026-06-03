from torchmetrics.image.arniqa import ARNIQA
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import cv2
import glob
from engineered_score import EngineeredScore

DATASET_PATH = "/home/infres/yrothlin-24/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train"
OUTPUT_DIR = "../outputs/metrics"
METRIC = "engineered"
ENGINEERED_METHOD = "weighted"

image_paths = sorted(glob.glob(f"{DATASET_PATH}/**/images/*", recursive=True))
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(device)

metric_label = f"{METRIC}_{ENGINEERED_METHOD}" if METRIC == "engineered" else METRIC
score_col = f"{metric_label}_score"

if METRIC == "arniqa":
    metric = ARNIQA(regressor_dataset="koniq10k", normalize=True).to(device)
    transform = T.ToTensor()
else:
    scorer = EngineeredScore(method=ENGINEERED_METHOD)
    scorer.fit([cv2.imread(p) for p in image_paths])

results = []
for idx, img_path in enumerate(image_paths, start=1):
    if METRIC == "arniqa":
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            score = metric(img_tensor).item()
    else:
        score = scorer(cv2.imread(img_path))
    image_name = float(idx)
    print(f"{image_name}: {score:.4f}")
    results.append({"image": image_name, score_col: score})

df = pd.DataFrame(results)
OUTPUT_CSV = f"{OUTPUT_DIR}/{metric_label}_scores.csv"
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSauvegardé dans {OUTPUT_CSV}")
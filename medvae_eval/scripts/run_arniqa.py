from torchmetrics.image.arniqa import ARNIQA
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import glob

DATASET_PATH = "/home/infres/yrothlin-24/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train"
image_paths = sorted(glob.glob(f"{DATASET_PATH}/**/images/*", recursive=True))
OUTPUT_CSV = "../outputs/metrics/arniqa_scores.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(device)

metric = ARNIQA(regressor_dataset="koniq10k", normalize=True).to(device)

transform = T.ToTensor()

results = []

for idx, img_path in enumerate(image_paths, start=1):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        score = metric(img_tensor).item()

    image_name = f"{idx}.png"
    print(f"{image_name}: {score:.4f}")
    results.append({"image": image_name, "arniqa_score": score})

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSauvegardé dans {OUTPUT_CSV}")
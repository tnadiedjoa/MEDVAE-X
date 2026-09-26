import os
import json
import cv2
import numpy as np
from pathlib import Path

# Racine du dataset ARCADE : $ARCADE_ROOT si défini, sinon <repo>/data/arcade
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCADE_ROOT = os.environ.get("ARCADE_ROOT", os.path.join(_REPO_ROOT, "data", "arcade"))
DATASET_PATH = f"{ARCADE_ROOT}/dataset_phase_1/segmentation_dataset/seg_train"
ANN_PATH = f"{DATASET_PATH}/annotations/seg_train.json"
OUTPUT_DIR = os.path.join(_REPO_ROOT, "medvae_eval", "outputs", "sample")
POISSON_SCALE = 0.1  # plus bas = plus de bruit shot
SEED = 42

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

coco = json.load(open(ANN_PATH))
images = {img["id"]: img for img in coco["images"]}
anns_by_img = {}
for ann in coco["annotations"]:
    anns_by_img.setdefault(ann["image_id"], []).append(ann)

rng = np.random.default_rng(SEED)
img_id = int(rng.choice(sorted(anns_by_img.keys())))
info = images[img_id]
file_name = info["file_name"]
H, W = info["height"], info["width"]

img_path = next(Path(DATASET_PATH).rglob(file_name))
img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

arr = img_gray.astype(np.float32) * POISSON_SCALE
noisy = np.random.poisson(np.clip(arr, 0, None)).astype(np.float32) / max(POISSON_SCALE, 1e-6)
img_deg = np.clip(noisy, 0, 255).astype(np.uint8)

mask = np.zeros((H, W), np.uint8)
for ann in anns_by_img[img_id]:
    for seg in ann["segmentation"]:
        pts = np.array(seg, np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 255)

stem = Path(file_name).stem
cv2.imwrite(f"{OUTPUT_DIR}/{stem}_clean.png", img_gray)
cv2.imwrite(f"{OUTPUT_DIR}/{stem}_poisson.png", img_deg)
cv2.imwrite(f"{OUTPUT_DIR}/{stem}_mask.png", mask)
print(f"Image {file_name} (id={img_id}) -> {OUTPUT_DIR}/{stem}_*.png", flush=True)
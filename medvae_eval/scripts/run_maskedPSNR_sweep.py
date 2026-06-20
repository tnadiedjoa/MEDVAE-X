import torch
from medvae import MVAE
from torchmetrics.image.arniqa import ARNIQA
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import glob
from tqdm import tqdm
from masked_psnr import MaskedPSNR

DATASET_PATH = "/home/infres/yrothlin-24/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train"
ANN_PATH = f"{DATASET_PATH}/annotations/seg_train.json"
OUTPUT_DIR = "../outputs/degradation"
N_IMAGES = 50
N_LEVELS = 20
SEED = 42
DEGRADATION = "poisson"  # "poisson" ou "jpeg" ou "blur"
DEGRA_METRIC = "arniqa"  # "masked" (masked_psnr clean vs deg) ou "arniqa" (sur l'image dégradée)

POISSON_SCALES = np.linspace(1.0, 0.05, N_LEVELS)
JPEG_QUALITIES = (95 - np.linspace(0, 1, N_LEVELS) * 90).astype(int)
BLUR_KERNELS = np.linspace(1, 31, N_LEVELS).round().astype(int)
BLUR_KERNELS = np.where(BLUR_KERNELS % 2 == 0, BLUR_KERNELS + 1, BLUR_KERNELS)

all_paths = sorted(glob.glob(f"{DATASET_PATH}/**/images/*", recursive=True))
rng = np.random.default_rng(SEED)
image_paths = list(rng.choice(all_paths, size=min(N_IMAGES, len(all_paths)), replace=False))
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(device, flush=True)

print("Chargement MedVAE...", flush=True)
model = MVAE(model_name="medvae_4_1_2d", modality="xray").to(device)
model.requires_grad_(False)
model.eval()
print("MedVAE OK", flush=True)

masked_recon = MaskedPSNR(ANN_PATH, data_range=2.0).to(device)
masked_degra = MaskedPSNR(ANN_PATH, data_range=1.0).to(device)
masked_cleanrec = MaskedPSNR(ANN_PATH, data_range=2.0).to(device)
if DEGRA_METRIC == "arniqa":
    arniqa = ARNIQA(regressor_dataset="koniq10k", normalize=True).to(device)

print(f"{len(image_paths)} images, démarrage du sweep ({DEGRADATION})...", flush=True)
results = []

for level in range(N_LEVELS):
    scale   = float(POISSON_SCALES[level])
    quality = int(JPEG_QUALITIES[level])
    kernel = int(BLUR_KERNELS[level])

    recon_scores = []
    degra_scores = []
    cleanrec_scores = []

    for img_path in tqdm(image_paths, desc=f"level {level:02d}"):
        file_name = Path(img_path).name
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        clean = torch.as_tensor(img_gray, dtype=torch.float32, device=device) / 255.0

        if DEGRADATION == "poisson":
            arr = img_gray.astype(np.float32) * scale
            noisy = np.random.poisson(np.clip(arr, 0, None)).astype(np.float32) / max(scale, 1e-6)
            img_deg = np.clip(noisy, 0, 255).astype(np.uint8)
        elif DEGRADATION == "blur":
            if kernel <= 1:
                img_deg = img_gray.copy()
            else:
                img_deg = cv2.GaussianBlur(img_gray, (kernel, kernel), 0)
        else:
            _, enc = cv2.imencode(".jpg", img_gray, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            img_deg = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

        if DEGRA_METRIC == "arniqa":
            deg_rgb = torch.as_tensor(cv2.cvtColor(img_deg, cv2.COLOR_GRAY2RGB), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
            with torch.no_grad():
                degra_scores.append(arniqa(deg_rgb).item())
        else:
            deg = torch.as_tensor(img_deg, dtype=torch.float32, device=device) / 255.0
            masked_degra.set_image(file_name)
            degra_scores.append(masked_degra(deg, clean).item())

        tmp_path = Path(OUTPUT_DIR) / "_tmp.png"
        cv2.imwrite(str(tmp_path), img_deg)
        img_input = model.apply_transform(str(tmp_path)).to(device)
        with torch.no_grad():
            decoded, _ = model(img_input, decode=True)
        decoded = decoded.unsqueeze(0).unsqueeze(0)

        masked_recon.set_image(file_name)
        recon_scores.append(masked_recon(decoded, img_input).item())

        clean_input = model.apply_transform(img_path).to(device)
        masked_cleanrec.set_image(file_name)
        cleanrec_scores.append(masked_cleanrec(decoded, clean_input).item())

    param = {"poisson": scale, "jpeg": quality, "blur": kernel}[DEGRADATION]
    results.append({
        "level":                level,
        "param":                param,
        "masked_psnr_recon":    np.mean(recon_scores),
        "masked_psnr_degra":    np.mean(degra_scores),
        "masked_psnr_cleanrec": np.mean(cleanrec_scores),
    })
    print(f"level {level:02d} | param={param} | recon={results[-1]['masked_psnr_recon']:.2f} degra={results[-1]['masked_psnr_degra']:.2f} cleanrec={results[-1]['masked_psnr_cleanrec']:.2f}", flush=True)

(Path(OUTPUT_DIR) / "_tmp.png").unlink(missing_ok=True)

prefix = "arniqa" if DEGRA_METRIC == "arniqa" else "masked"
df = pd.DataFrame(results)
df.to_csv(f"{OUTPUT_DIR}/{prefix}_sweep_{DEGRADATION}.csv", index=False)
print(f"\nSauvegardé dans {OUTPUT_DIR}/{prefix}_sweep_{DEGRADATION}.csv")
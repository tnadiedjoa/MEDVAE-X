from torchmetrics.image.arniqa import ARNIQA
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
import torch
from medvae import MVAE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import cv2
import glob
from tqdm import tqdm

DATASET_PATH = "/home/infres/yrothlin-24/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train"
OUTPUT_DIR = "../outputs/degradation"
N_IMAGES = 100
N_LEVELS = 50

image_paths = sorted(glob.glob(f"{DATASET_PATH}/**/images/*", recursive=True))[:N_IMAGES]
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
print(device, flush=True)

print("Chargement ARNIQA...", flush=True)
arniqa_metric = ARNIQA(regressor_dataset="koniq10k", normalize=True).to(device)
transform = T.ToTensor()
print("ARNIQA OK", flush=True)

print("Chargement MedVAE...", flush=True)
model = MVAE(model_name="medvae_4_1_2d", modality="xray").to(device)
model.requires_grad_(False)
model.eval()
print("MedVAE OK", flush=True)

psnr_metric    = PeakSignalNoiseRatio(data_range=1.0).to(device)
ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure().to(device)

t = np.linspace(0, 1, N_LEVELS)
noise_sigmas   = (t * 80).astype(int)
blur_kernels   = (1 + t * 30).astype(int)
blur_kernels   = np.where(blur_kernels % 2 == 0, blur_kernels + 1, blur_kernels)
jpeg_qualities = (95 - t * 90).astype(int)

print(f"{len(image_paths)} images, démarrage du sweep...", flush=True)
results = []

for level in range(N_LEVELS):
    sigma   = int(noise_sigmas[level])
    kernel  = int(blur_kernels[level])
    quality = int(jpeg_qualities[level])

    arniqa_scores  = []
    psnr_scores    = []
    ms_ssim_scores = []

    for img_path in tqdm(image_paths, desc=f"level {level:02d} σ={sigma} k={kernel} q={quality}"):
        img_bgr = cv2.imread(img_path)

        if sigma > 0:
            noise = np.random.normal(0, sigma, img_bgr.shape).astype(np.float32)
            img_bgr = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if kernel > 1:
            img_bgr = cv2.GaussianBlur(img_bgr, (kernel, kernel), 0)

        _, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        img_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            arniqa_scores.append(arniqa_metric(img_tensor).item())

        tmp_path = Path(OUTPUT_DIR) / "_tmp.png"
        cv2.imwrite(str(tmp_path), img_bgr)
        img_input = model.apply_transform(str(tmp_path)).to(device)
        with torch.no_grad():
            decoded, _ = model(img_input, decode=True)
        decoded = decoded.unsqueeze(0).unsqueeze(0)
        psnr_scores.append(psnr_metric(decoded, img_input).item())
        ms_ssim_scores.append(ms_ssim_metric(decoded, img_input).item())

    results.append({
        "level":        level,
        "noise_sigma":  sigma,
        "blur_kernel":  kernel,
        "jpeg_quality": quality,
        "arniqa_mean":  np.mean(arniqa_scores),
        "psnr_mean":    np.mean(psnr_scores),
        "ms_ssim_mean": np.mean(ms_ssim_scores),
    })
    print(f"level {level:02d} | σ={sigma} k={kernel} q={quality} | arniqa={results[-1]['arniqa_mean']:.3f} psnr={results[-1]['psnr_mean']:.2f} ms_ssim={results[-1]['ms_ssim_mean']:.4f}", flush=True)

(Path(OUTPUT_DIR) / "_tmp.png").unlink(missing_ok=True)

df = pd.DataFrame(results)
df.to_csv(f"{OUTPUT_DIR}/degradation_results.csv", index=False)
print(f"\nSauvegardé dans {OUTPUT_DIR}/degradation_results.csv")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(df["arniqa_mean"], df["psnr_mean"], color="blue", marker="o", markersize=3)
plt.xlabel("ARNIQA moyen")
plt.ylabel("PSNR moyen (dB)")
plt.title("PSNR vs ARNIQA")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(df["arniqa_mean"], df["ms_ssim_mean"], color="orange", marker="o", markersize=3)
plt.xlabel("ARNIQA moyen")
plt.ylabel("MS-SSIM moyen")
plt.title("MS-SSIM vs ARNIQA")
plt.grid()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/psnr_mssim_vs_arniqa.png", dpi=150, bbox_inches="tight")
plt.show()
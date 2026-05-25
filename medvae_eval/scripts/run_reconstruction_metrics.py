import torch
from medvae import MVAE
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
from pathlib import Path
import pandas as pd

IMAGE_DIR = "../../arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train/images"
OUTPUT_CSV = "outputs/metrics/reconstruction_metrics.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

model = MVAE(model_name="medvae_4_1_2d", modality="xray").to(device)
model.requires_grad_(False)
model.eval()

psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
ms_ssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)

results = []
image_paths = sorted(Path(IMAGE_DIR).glob("*"))

for img_path in image_paths:
    img = model.apply_transform(str(img_path)).to(device)

    with torch.no_grad():
        decoded_img, latent = model(img, decode=True)

    decoded_img = decoded_img.unsqueeze(0).unsqueeze(0)
    psnr_score = psnr(decoded_img, img).item()
    ms_ssim_score = ms_ssim(decoded_img, img).item()

    print(f"{img_path.name}: PSNR={psnr_score:.2f}, MS-SSIM={ms_ssim_score:.4f}")
    results.append({
        "image": img_path.name,
        "psnr": psnr_score,
        "ms_ssim": ms_ssim_score
    })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSauvegardé dans {OUTPUT_CSV}")
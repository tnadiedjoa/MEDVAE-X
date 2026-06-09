import torch
from medvae import MVAE
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
import pandas as pd
import glob
from masked_psnr import MaskedPSNR

DATASET_PATH = "/home/infres/yrothlin-24/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train"
ANN_PATH = f"{DATASET_PATH}/annotations/seg_train.json"
OUTPUT_CSV = "../outputs/metrics/FR_results.csv"

image_paths = sorted(glob.glob(f"{DATASET_PATH}/**/images/*", recursive=True))

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
model = MVAE(model_name="medvae_4_1_2d", modality="xray").to(device)
model.requires_grad_(False)
model.eval()

psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
ms_ssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)
masked_psnr = MaskedPSNR(ANN_PATH).to(device)

results = []
for idx, img_path in enumerate(image_paths, start=1):
    img = model.apply_transform(str(img_path)).to(device)

    with torch.no_grad():
        decoded_img, latent = model(img, decode=True)
    decoded_img = decoded_img.unsqueeze(0).unsqueeze(0)

    psnr_score = psnr(decoded_img, img).item()
    ms_ssim_score = ms_ssim(decoded_img, img).item()
    file_name = img_path.split("/")[-1]
    masked_psnr_score = masked_psnr.set_image(file_name)(decoded_img, img).item()

    image_name = float(idx)
    print(f"{image_name}: PSNR={psnr_score:.2f}, masked={masked_psnr_score:.2f}, MS-SSIM={ms_ssim_score:.4f}")
    results.append({
        "image": image_name,
        "psnr": psnr_score,
        "masked_psnr": masked_psnr_score,
        "ms_ssim": ms_ssim_score
    })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSauvegardé dans {OUTPUT_CSV}")
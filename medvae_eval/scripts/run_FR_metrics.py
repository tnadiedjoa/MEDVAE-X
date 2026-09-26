import os
import torch
from medvae import MVAE
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
import pandas as pd
import glob
from masked_psnr import MaskedPSNR

# Sorties dans <repo>/medvae_eval/outputs/, quel que soit le dossier courant
_OUTPUTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

# Racine du dataset ARCADE : $ARCADE_ROOT si défini, sinon <repo>/data/arcade
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCADE_ROOT = os.environ.get("ARCADE_ROOT", os.path.join(_REPO_ROOT, "data", "arcade"))
DATASET_PATH = f"{ARCADE_ROOT}/dataset_phase_1/segmentation_dataset/seg_train"
ANN_PATH = f"{DATASET_PATH}/annotations/seg_train.json"

image_paths = sorted(glob.glob(f"{DATASET_PATH}/**/images/*", recursive=True))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MVAE(model_name="medvae_4_1_2d", modality="xray").to(device)
model.requires_grad_(False)
model.eval()

psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)
ms_ssim = MultiScaleStructuralSimilarityIndexMeasure().to(device)
masked_psnr = MaskedPSNR(ANN_PATH).to(device)

results = []
for img_path in image_paths:
    img = model.apply_transform(str(img_path)).to(device)
    with torch.no_grad():
        decoded_img, latent = model(img, decode=True)
    decoded_img = decoded_img.unsqueeze(0).unsqueeze(0)
    file_name = img_path.split("/")[-1]
    results.append({
        "image": file_name,
        "psnr": psnr(decoded_img, img).item(),
        "masked_psnr": masked_psnr.set_image(file_name)(decoded_img, img).item(),
        "ms_ssim": ms_ssim(decoded_img, img).item(),
    })
    print(results[-1])

pd.DataFrame(results).to_csv(f"{_OUTPUTS}/metrics/FR_results.csv", index=False)

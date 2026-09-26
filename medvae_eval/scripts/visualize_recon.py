import os
import torch
from medvae import MVAE
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import glob

# Sorties dans <repo>/medvae_eval/outputs/, quel que soit le dossier courant
_OUTPUTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

# Racine du dataset ARCADE : $ARCADE_ROOT si défini, sinon <repo>/data/arcade
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARCADE_ROOT = os.environ.get("ARCADE_ROOT", os.path.join(_REPO_ROOT, "data", "arcade"))
DATASET_PATH = f"{ARCADE_ROOT}/dataset_phase_1/segmentation_dataset/seg_train"
OUTPUT_DIR = f"{_OUTPUTS}/degradation"
N_LEVELS = 20
SEED = 42
LEVELS_TO_SHOW = [0, 10, 15, 19]

t = np.linspace(0, 1, N_LEVELS)
POISSON_SCALES = np.linspace(1.0, 0.05, N_LEVELS)
JPEG_QUALITIES = (95 - t * 90).astype(int)
BLUR_KERNELS = (1 + t * 30).astype(int)
BLUR_KERNELS = np.where(BLUR_KERNELS % 2 == 0, BLUR_KERNELS + 1, BLUR_KERNELS)

all_paths = sorted(glob.glob(f"{DATASET_PATH}/**/images/*", recursive=True))
rng = np.random.default_rng(SEED)
img_path = str(rng.choice(all_paths))
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
model = MVAE(model_name="medvae_4_1_2d", modality="xray").to(device)
model.requires_grad_(False)
model.eval()

img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

def degrade(img, level, kind):
    if kind == "poisson":
        scale = float(POISSON_SCALES[level])
        arr = img.astype(np.float32) * scale
        noisy = np.random.poisson(np.clip(arr, 0, None)).astype(np.float32) / max(scale, 1e-6)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif kind == "blur":
        k = int(BLUR_KERNELS[level])
        return img.copy() if k <= 1 else cv2.GaussianBlur(img, (k, k), 0)
    else:
        q = int(JPEG_QUALITIES[level])
        _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

def reconstruct(img_deg):
    tmp = Path(OUTPUT_DIR) / "_tmp_vis.png"
    cv2.imwrite(str(tmp), img_deg)
    img_input = model.apply_transform(str(tmp)).to(device)
    with torch.no_grad():
        decoded, _ = model(img_input, decode=True)
    tmp.unlink(missing_ok=True)
    return decoded.squeeze().cpu().numpy()

for kind in ["poisson", "blur", "jpeg"]:
    n = len(LEVELS_TO_SHOW)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    for j, level in enumerate(LEVELS_TO_SHOW):
        deg = degrade(img_gray, level, kind)
        rec = reconstruct(deg)
        axes[0, j].imshow(deg, cmap="gray")
        axes[0, j].set_title(f"Dégradée — niveau {level}")
        axes[1, j].imshow(rec, cmap="gray")
        axes[1, j].set_title(f"Reconstruction — niveau {level}")
        for ax in (axes[0, j], axes[1, j]):
            ax.axis("off")
    axes[0, 0].set_ylabel("Entrée dégradée")
    axes[1, 0].set_ylabel("MedVAE")
    fig.suptitle(f"Reconstruction MedVAE par niveau ({kind})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/recon_visual_{kind}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"recon_visual_{kind}.png OK", flush=True)
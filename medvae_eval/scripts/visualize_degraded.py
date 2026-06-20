import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

DATASET_PATH = "/home/infres/yrothlin-24/arcade_challenge_datasets/dataset_phase_1/stenosis_dataset/sten_train"
OUTPUT_PATH  = "../outputs/degradation/degradation_visual.png"
SEED         = 42

POISSON_LEVELS = [60.0, 20.0, 5.0]
JPEG_LEVELS    = [50, 20, 5]
BLUR_LEVELS    = [5, 13, 25]

image_paths = sorted(glob.glob(f"{DATASET_PATH}/**/images/*", recursive=True))
random.seed(SEED)
img_path = random.choice(image_paths)
img_bgr = cv2.imread(img_path)

def add_poisson_noise(img, scale):
    img_f = img.astype(np.float32) / 255.0
    noisy = np.random.poisson(img_f * scale) / scale
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)

def add_jpeg(img, quality):
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def add_blur(img, kernel):
    k = (kernel | 1)
    return cv2.GaussianBlur(img, (k, k), 0)

col_titles = ["Clean", "Poisson", "JPEG", "Flou gaussien"]
n_rows = 3

fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows))
fig.suptitle("Visualisation des dégradations synthétiques", fontsize=14, fontweight="bold", y=1.0)

for row in range(n_rows):
    cells = [
        (img_bgr, "Clean"),
        (add_poisson_noise(img_bgr.copy(), POISSON_LEVELS[row]), f"Poisson\n(scale={POISSON_LEVELS[row]})"),
        (add_jpeg(img_bgr.copy(), JPEG_LEVELS[row]),             f"JPEG\n(q={JPEG_LEVELS[row]})"),
        (add_blur(img_bgr.copy(), BLUR_LEVELS[row]),             f"Flou\n(k={BLUR_LEVELS[row]})"),
    ]
    for col, (degraded, title) in enumerate(cells):
        degraded_rgb = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
        axes[row, col].imshow(degraded_rgb, cmap="gray")
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(col_titles[col], fontsize=11, fontweight="bold")
        axes[row, col].text(0.5, -0.08, title.split("\n")[-1] if col > 0 else "",
                            transform=axes[row, col].transAxes,
                            ha="center", va="top", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.show()
print(f"Sauvegardé dans {OUTPUT_PATH}")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

# DATASET_PATH = "/home/infres/yrothlin-24/arcade_challenge_datasets/dataset_phase_1/segmentation_dataset/seg_train"
DATASET_PATH = "/home/infres/yrothlin-24/arcade_challenge_datasets/dataset_phase_1/stenosis_dataset/sten_train"
OUTPUT_PATH  = "../outputs/degradation/degradation_visual.png"
N_IMAGES     = 5
NOISE_SIGMA  = 50
BLUR_KERNEL  = 20
JPEG_QUALITY = 80

image_paths = sorted(glob.glob(f"{DATASET_PATH}/**/images/*", recursive=True))
random.seed(42)
sample_paths = random.sample(image_paths, N_IMAGES)

def add_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def add_blur(img, kernel):
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(img, (k, k), 0)

def add_jpeg(img, quality):
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def add_all(img, sigma, kernel, quality):
    img = add_noise(img, sigma)
    img = add_blur(img, kernel)
    img = add_jpeg(img, quality)
    return img

col_titles = ["Original", f"Bruit\n(σ={NOISE_SIGMA})", f"Flou\n(k={BLUR_KERNEL})", f"JPEG\n(q={JPEG_QUALITY})", "Mix"]
transforms = [
    lambda img: img,
    lambda img: add_noise(img, NOISE_SIGMA),
    lambda img: add_blur(img, BLUR_KERNEL),
    lambda img: add_jpeg(img, JPEG_QUALITY),
    lambda img: add_all(img, NOISE_SIGMA, BLUR_KERNEL, JPEG_QUALITY),
]

fig, axes = plt.subplots(N_IMAGES, len(col_titles), figsize=(3 * len(col_titles), 3 * N_IMAGES))
fig.suptitle("Visualisation des dégradations synthétiques", fontsize=14, fontweight="bold", y=1.01)

for row, img_path in enumerate(sample_paths):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    for col, (title, fn) in enumerate(zip(col_titles, transforms)):
        degraded = fn(img_bgr.copy())
        degraded_rgb = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
        axes[row, col].imshow(degraded_rgb)
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(title, fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.show()
print(f"Sauvegardé dans {OUTPUT_PATH}")
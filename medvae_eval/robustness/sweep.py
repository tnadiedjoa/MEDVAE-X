"""Robustesse de MedVAE aux dégradations d'entrée — version corrigée de l'axe A.

Pour chaque image et chaque niveau de dégradation (bruit de Poisson, JPEG, flou) :
    x        image propre            d = degrade(x)        r = MedVAE(d)
et on mesure, sur toute l'image et sur les pixels de vaisseaux :
    psnr_clean_deg  = PSNR(x, d)   qualité de l'entrée dégradée
    psnr_clean_rec  = PSNR(x, r)   qualité de la reconstruction par rapport à l'image PROPRE
    psnr_deg_rec    = PSNR(d, r)   fidélité à l'entrée dégradée (métrique de l'étude d'origine)
La question « MedVAE restaure-t-il l'image ? » se lit dans delta = psnr_clean_rec - psnr_clean_deg
(> 0 : la reconstruction est plus proche de l'image propre que l'entrée).

Différences avec medvae_eval/scripts/run_maskedPSNR_sweep.py :
    - latent déterministe (moyenne du posterior) au lieu d'un tirage aléatoire ;
    - même échelle pour toutes les métriques ([0, 1], image / 255) au lieu d'un
      min-max par image, qui changeait d'une image à l'autre et avec la dégradation ;
    - bruit tiré avec un générateur seedé par image et par niveau ;
    - images de seg_val (200) au lieu de 50 images de seg_train ; une ligne par image
      dans le CSV, pour des intervalles de confiance.

Usage (racine du repo) :
    python medvae_eval/robustness/sweep.py [--n-images 200] [--levels 20]
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from medvae import MVAE
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCADE_ROOT = Path(os.environ.get("ARCADE_ROOT", REPO_ROOT / "data" / "arcade"))
SEG_VAL = ARCADE_ROOT / "dataset_phase_1" / "segmentation_dataset" / "seg_val"


def degradation_params(n_levels: int) -> dict:
    """Mêmes plages que l'étude d'origine (niveau 0 = plus faible)."""
    kernels = np.linspace(1, 31, n_levels).round().astype(int)
    return {
        "poisson": np.linspace(1.0, 0.05, n_levels),                     # échelle (plus bas = plus bruité)
        "jpeg": (95 - np.linspace(0, 1, n_levels) * 90).astype(int),     # qualité JPEG
        "blur": np.where(kernels % 2 == 0, kernels + 1, kernels),        # taille du noyau gaussien
    }


def degrade(img: np.ndarray, kind: str, param, rng: np.random.Generator) -> np.ndarray:
    if kind == "poisson":
        noisy = rng.poisson(img.astype(np.float32) * param) / param
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(param)])
        return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    if kind == "blur":
        return img.copy() if param <= 1 else cv2.GaussianBlur(img, (int(param), int(param)), 0)
    raise ValueError(kind)


def vessel_masks(ann_path: Path) -> dict:
    """Masque binaire des vaisseaux (toutes classes) par nom de fichier."""
    coco = json.load(open(ann_path))
    polys = {}
    for a in coco["annotations"]:
        polys.setdefault(a["image_id"], []).extend(a["segmentation"])
    masks = {}
    for im in coco["images"]:
        m = np.zeros((im["height"], im["width"]), np.uint8)
        for poly in polys.get(im["id"], []):
            cv2.fillPoly(m, [np.array(poly, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
        masks[im["file_name"]] = m.astype(bool)
    return masks


def psnr(a: np.ndarray, b: np.ndarray, mask=None) -> float:
    """PSNR sur des images dans [0, 1] (data_range = 1)."""
    diff = (a - b) if mask is None else (a[mask] - b[mask])
    mse = float(np.mean(diff ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


class Reconstructor:
    """MedVAE déterministe : x ∈ [0,1] → encode (moyenne du posterior) → decode → [0,1]."""

    def __init__(self, device):
        self.device = device
        self.mvae = MVAE(model_name="medvae_4_1_2d", modality="xray").to(device).eval()

    @torch.no_grad()
    def __call__(self, images_u8: list) -> list:
        x = torch.stack([torch.from_numpy(i).float() / 255.0 for i in images_u8]).unsqueeze(1)
        x = x.to(self.device) * 2 - 1
        out = []
        for chunk in x.split(2):   # attention MedVAE : ~1 Go par image en 512²
            z = self.mvae.model.encode(chunk).mode()
            out.append(((self.mvae.model.decode(z) + 1) / 2).clamp(0, 1))
        return [r[0].cpu().numpy() for r in torch.cat(out)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-images", type=int, default=200)
    parser.add_argument("--levels", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(REPO_ROOT / "medvae_eval" / "outputs" / "robustness" / "sweep.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks = vessel_masks(SEG_VAL / "annotations" / "seg_val.json")
    files = sorted(masks)[: args.n_images]
    recon = Reconstructor(device)
    params = degradation_params(args.levels)

    rows = []
    for idx, name in enumerate(tqdm(files, desc="images")):
        clean_u8 = cv2.imread(str(SEG_VAL / "images" / name), cv2.IMREAD_GRAYSCALE)
        clean = clean_u8 / 255.0
        mask = masks[name]
        # Référence : reconstruction de l'image propre
        rec_clean = recon([clean_u8])[0]
        base = {"image": name, "psnr_clean_recclean": psnr(clean, rec_clean),
                "mpsnr_clean_recclean": psnr(clean, rec_clean, mask)}

        for kind, values in params.items():
            degraded = []
            for level, value in enumerate(values):
                rng = np.random.default_rng([args.seed, idx, level])   # bruit reproductible
                degraded.append(degrade(clean_u8, kind, value, rng))
            recs = recon(degraded)
            for level, (value, d_u8, r) in enumerate(zip(values, degraded, recs)):
                d = d_u8 / 255.0
                rows.append({**base, "degradation": kind, "level": level, "param": float(value),
                             "psnr_clean_deg": psnr(clean, d), "psnr_clean_rec": psnr(clean, r),
                             "psnr_deg_rec": psnr(d, r),
                             "mpsnr_clean_deg": psnr(clean, d, mask), "mpsnr_clean_rec": psnr(clean, r, mask),
                             "mpsnr_deg_rec": psnr(d, r, mask)})

    df = pd.DataFrame(rows)
    df["delta"] = df["psnr_clean_rec"] - df["psnr_clean_deg"]
    df["mdelta"] = df["mpsnr_clean_rec"] - df["mpsnr_clean_deg"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} lignes -> {args.out}")


if __name__ == "__main__":
    main()

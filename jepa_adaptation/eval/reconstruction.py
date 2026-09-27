"""Qualité de reconstruction des encodeurs de l'axe C (MedVAE, étape 1, étape 2 JEPA).

À l'étape 2, seul l'encodeur est entraîné (décodeur figé) : si la loss JEPA déplace le
latent hors de ce que le décodeur sait lire, la reconstruction se dégrade. On la mesure
sur les 300 images de test officielles, en 512×512, latent déterministe (moyenne du
posterior), images comparées dans [0, 1] : PSNR, SSIM et PSNR sur les vaisseaux.

Usage (racine du repo) :
    python jepa_adaptation/eval/reconstruction.py --model stage2 --ckpt jepa_adaptation/outputs/stage2/best.pt
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import piq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from finetune.config import load_config    # noqa: E402
from finetune.dataset import ArcadeDataset  # noqa: E402
from probe import load_autoencoder          # noqa: E402


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["medvae", "stage1", "stage2"], required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--max-images", type=int, default=None, help="test rapide")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.model != "medvae" and not args.ckpt:
        parser.error("--ckpt est requis pour stage1 / stage2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = load_autoencoder(args.model, args.ckpt, device)
    d = load_config(str(REPO_ROOT / "finetune" / "configs" / "condition_a.yaml"))["data"]
    loader = DataLoader(ArcadeDataset(d["val_images"], d["val_ann"]), batch_size=2, num_workers=4)

    rows = []
    for x, m in tqdm(loader, desc="reconstruction"):
        x = x.to(device)
        rec = ae.decode(ae.encode(x * 2 - 1).mode())
        r01 = ((rec + 1) / 2).clamp(0, 1)
        ssim = piq.ssim(r01, x, data_range=1.0, reduction="none")
        for i in range(len(x)):
            vessels = m[i].to(device) > 0
            rows.append({"psnr": -10 * math.log10(F.mse_loss(r01[i], x[i]).item()),
                         "mpsnr": -10 * math.log10(F.mse_loss(r01[i, 0][vessels], x[i, 0][vessels]).item()),
                         "ssim": ssim[i].item()})
        if args.max_images and len(rows) >= args.max_images:
            break

    summary = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "mpsnr", "ssim")}
    result = {"model": args.model, "ckpt": args.ckpt, "n_images": len(rows), **summary}
    print(json.dumps(result, indent=2))
    out = Path(args.out or REPO_ROOT / "experiments" / "jepa" / f"reconstruction_{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

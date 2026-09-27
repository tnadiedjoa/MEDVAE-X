"""Axe B refait : le conditionnement FiLM par un score de qualité améliore-t-il MedVAE ?

Protocole (corrige l'étude d'origine, notebooks 04 à 07) :
    - données : seg_train (1000 images) découpé en 900 entraînement / 100 validation
      (découpage fixe, identique pour tous les runs) ; TEST = seg_val (200 images),
      jamais vu à l'entraînement ni pour la sélection du modèle ;
    - résolution 256×256 (contre 64×64, où les vaisseaux font ~1 pixel) ;
    - baseline (MedVAE sans FiLM) et FiLM entraînés exactement de la même façon :
      tous les paramètres, même nombre de pas, même learning rate ;
    - sélection du checkpoint sur l'erreur de reconstruction de validation (et non sur
      la loss totale, dominée par le terme KL) ;
    - évaluation test : PSNR, SSIM, HaarPSI (piq) et PSNR sur les vaisseaux ; pour FiLM,
      avec le vrai c, un c mélangé entre images et un c constant (moyenne d'entraînement).
      Si c est réellement utilisé, le vrai c doit battre les deux contrôles.
    - --steps 0 : évalue le MedVAE pré-entraîné sans aucun entraînement ;
    - loss = MSE moyenne + kl_weight × KL / nombre de pixels (convention MedVAE, cf. main).

Usage (racine du repo) :
    python medvae_eval/film/train_film.py --model film --approach C --seed 42
    python medvae_eval/film/train_film.py --model baseline --seed 42
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import piq
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "cvae"))
sys.path.insert(0, str(HERE.parent / "robustness"))

from finetune.runs import create_run                       # noqa: E402
from model import FiLMMedVAE, PlainMedVAE, load_medvae     # noqa: E402
from pipeline_config import APPROACHES, read_csv_with_paths  # noqa: E402
from sweep import SEG_VAL, vessel_masks                    # noqa: E402

LABELS_CSV = REPO_ROOT / "medvae_eval" / "cvae" / "outputs_cvae" / "03_Inductive_Bias_Generation" / "labels_quality.csv"


class QualityDataset(Dataset):
    """(image dans [-1, 1] de taille img_size², score c, nom de fichier)."""

    def __init__(self, df: pd.DataFrame, score_col: str, img_size: int):
        self.df = df.reset_index(drop=True)
        self.score_col, self.img_size = score_col, img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("L").resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0)[None] * 2 - 1
        return x, torch.tensor(float(row[self.score_col]), dtype=torch.float32), Path(row["path"]).name


def split_data(df: pd.DataFrame, n_val: int = 100):
    """900 / 100 dans seg_train (découpage fixe), test = seg_val."""
    train_all = df[df["split"] == "train"].sort_values("image_id")
    val = train_all.sample(n=n_val, random_state=0)
    return train_all.drop(val.index), val, df[df["split"] == "val"]


@torch.no_grad()
def val_mse(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for x, c, _ in loader:
        x, c = x.to(device), c.to(device)
        rec, _ = model(x, c, sample_posterior=False)
        total += F.mse_loss(rec, x, reduction="sum").item() / x[0].numel()
        n += len(x)
    model.train()
    return total / n


@torch.no_grad()
def evaluate(model, loader, device, masks, c_override=None):
    """Métriques par image, sur [0, 1]. c_override : fonction (c, noms) → c utilisé."""
    model.eval()
    rows = []
    for x, c, names in loader:
        x, c = x.to(device), c.to(device)
        if c_override is not None:
            c = c_override(c, names).to(device)
        rec, _ = model(x, c, sample_posterior=False)
        x01, r01 = ((x + 1) / 2).clamp(0, 1), ((rec + 1) / 2).clamp(0, 1)
        ssim = piq.ssim(r01, x01, data_range=1.0, reduction="none")
        haarpsi = piq.haarpsi(r01, x01, data_range=1.0, reduction="none")
        for i, name in enumerate(names):
            mse = F.mse_loss(r01[i], x01[i]).item()
            m = torch.from_numpy(np.asarray(Image.fromarray(masks[name].astype(np.uint8))
                                            .resize(x.shape[-2:][::-1], Image.Resampling.NEAREST), bool)).to(device)
            vmse = F.mse_loss(r01[i, 0][m], x01[i, 0][m]).item()
            rows.append({"image": name, "psnr": -10 * math.log10(mse), "mpsnr": -10 * math.log10(vmse),
                         "ssim": ssim[i].item(), "haarpsi": haarpsi[i].item()})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["baseline", "film"], required=True)
    parser.add_argument("--approach", choices=list(APPROACHES), default="C",
                        help="score de qualité c (colonne de labels_quality.csv)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--film-lr", type=float, default=None,
                        help="learning rate des couches FiLM (défaut : --lr)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_col = APPROACHES[args.approach]["score_col"]

    df = read_csv_with_paths(LABELS_CSV)
    train_df, val_df, test_df = split_data(df)
    loaders = {name: DataLoader(QualityDataset(d, score_col, args.img_size), batch_size=args.batch_size,
                                shuffle=(name == "train"), num_workers=4, drop_last=(name == "train"))
               for name, d in (("train", train_df), ("val", val_df), ("test", test_df))}

    model = (FiLMMedVAE if args.model == "film" else PlainMedVAE)(load_medvae()).to(device)
    name = args.run_name or f"film_{args.model}_{args.approach if args.model == 'film' else 'noc'}_seed{args.seed}"
    run_dir = Path(create_run({"experiment": {"name": name, "seed": args.seed}, "film": vars(args),
                               "data": {"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)}},
                              name))

    # Entraînement : tous les paramètres, AdamW + cosine (lr → lr/10), loss = MSE + kl_weight * KL
    film_params = [p for n, p in model.named_parameters() if n.startswith("film.")]
    base_params = [p for n, p in model.named_parameters() if not n.startswith("film.")]
    groups = [{"params": base_params, "lr": args.lr}]
    if film_params:
        groups.append({"params": film_params, "lr": args.film_lr or args.lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    T = max(args.steps, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: 0.1 + 0.9 * (1 + math.cos(math.pi * min(s, T) / T)) / 2)
    history, best = [], (val_mse(model, loaders["val"], device), 0)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history.append({"step": 0, "val_mse": best[0]})
    step, it = 0, iter(loaders["train"])
    model.train()
    while step < args.steps:
        try:
            x, c, _ = next(it)
        except StopIteration:
            it = iter(loaders["train"])
            x, c, _ = next(it)
        x, c = x.to(device), c.to(device)
        rec, posterior = model(x, c, sample_posterior=True)
        rec_loss = F.mse_loss(rec, x)
        # KL sommé sur le latent, rapporté au nombre de pixels : comme dans MedVAE /
        # latent-diffusion, où KL et erreur de reconstruction sont tous deux sommés.
        # Sans cette normalisation, en 256×256 le terme KL (x1e-6) pèse ~700 fois la MSE
        # moyenne et l'entraînement dégrade la reconstruction.
        kl = posterior.kl().mean() / x[0].numel()
        loss = rec_loss + args.kl_weight * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        if step % args.val_every == 0 or step == args.steps:
            v = val_mse(model, loaders["val"], device)
            history.append({"step": step, "train_rec_mse": rec_loss.item(), "val_mse": v,
                            "lr": scheduler.get_last_lr()[0]})
            print(f"step {step:5d}  train {rec_loss.item():.5f}  val {v:.5f}" + ("  ★" if v < best[0] else ""),
                  flush=True)
            if v < best[0]:
                best = (v, step)
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "step": best[1], "val_mse": best[0], "args": vars(args)},
               run_dir / "best_model.pth")

    # Évaluation test (seg_val)
    masks = vessel_masks(SEG_VAL / "annotations" / "seg_val.json")
    variants = {"real_c": None}
    if args.model == "film":
        train_mean = float(train_df[score_col].mean())
        # c d'une autre image du test set (permutation fixe sur les 200 images)
        test_names = [Path(p).name for p in test_df["path"]]
        shuffled = np.random.default_rng(0).permutation(test_df[score_col].to_numpy())
        shuffled_c = dict(zip(test_names, shuffled))
        variants["shuffled_c"] = lambda c, names: torch.tensor([shuffled_c[n] for n in names], dtype=torch.float32)
        variants["constant_c"] = lambda c, names: torch.full_like(c.cpu(), train_mean)
    per_image, summary = [], {}
    for variant, override in variants.items():
        df_eval = evaluate(model, loaders["test"], device, masks, override).assign(variant=variant)
        per_image.append(df_eval)
        summary[variant] = {k: float(df_eval[k].mean()) for k in ("psnr", "mpsnr", "ssim", "haarpsi")}
        print(variant, {k: round(v, 4) for k, v in summary[variant].items()})
    pd.concat(per_image).to_csv(run_dir / "test_per_image.csv", index=False)
    with open(run_dir / "results.json", "w") as f:
        json.dump({name: {**summary, "best_step": best[1], "best_val_mse": best[0]}}, f, indent=2)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()

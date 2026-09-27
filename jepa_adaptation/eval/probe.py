"""Linear probe de segmentation sur les représentations figées (axe C, évaluation corrigée).

Mesure ce que chaque encodeur a appris, avec exactement la même procédure pour tous :
    - représentation : « penultimate » = features de l'encodeur avant conv_out (la couche
      que la loss JEPA entraîne) ou « latent » = z (moyenne du posterior, 1 canal) ;
      ramenées à 64×64 pour que tous les modèles aient la même résolution ;
    - probe : standardisation par canal (statistiques d'entraînement) puis convolution 1×1
      linéaire vers 26 classes, logits remis en 512×512 par interpolation bilinéaire ;
    - données : entraînement sur seg_train (1000), sélection sur seg_val (200), test sur les
      300 images officielles ; métrique : Dice sur les artères (fond exclu) ; plusieurs seeds ;
    - tâche : « segments » (26 classes : fond + 25 segments d'artères, qui demandent du
      contexte) ou « vessels » (vaisseau / fond, 2 classes : l'information locale suffit).

Modèles :
    medvae   MedVAE officiel pré-entraîné (medvae_4_1_2d, facteur 4 : features ramenées à 64×64)
    stage1   VAE de l'étape 1 (checkpoint best.pt, clé « autoencoder »)
    stage2   encodeur cible (EMA) de l'étape 2 JEPA (checkpoint best.pt, clé « model »)

Usage (racine du repo) :
    python jepa_adaptation/eval/probe.py --model stage2 --ckpt jepa_adaptation/outputs/stage2/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
JEPA_ROOT = HERE.parent
REPO_ROOT = JEPA_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from finetune.config import load_config           # noqa: E402
from finetune.dataset import ArcadeDataset         # noqa: E402
from finetune.losses import SegLoss                # noqa: E402
from finetune.metrics import SegMetrics            # noqa: E402

FEAT_SIZE = 64
NUM_CLASSES = 26   # remplacé par 2 pour la tâche « vessels »


# ── Encodeurs ────────────────────────────────────────────────────────────────

def _penultimate_hook(encoder, store):
    return encoder.conv_out.register_forward_pre_hook(lambda m, inp: store.append(inp[0]))


def build_encoder(model: str, ckpt: str, device):
    """Renvoie f(x ∈ [-1,1], B×1×512×512) → (features, z), à 64×64."""
    if model == "medvae":
        from medvae import MVAE
        ae = MVAE(model_name="medvae_4_1_2d", modality="xray").to(device).eval().model
    else:
        sys.path.insert(0, str(JEPA_ROOT))
        from utils.launch import bootstrap, load_config as jepa_config
        bootstrap()
        from models.medvae import MVAE as JepaMVAE
        cfg = jepa_config(str(JEPA_ROOT / "configs" / "stage_1.yaml"), str(JEPA_ROOT / "configs" / "model.yaml"))
        m = cfg["model"]
        mvae = JepaMVAE(ddconfig=m["ddconfig"], embed_dim=int(m["embed_dim"]),
                        apply_channel_ds=bool(m.get("apply_channel_ds", True)))
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        if model == "stage1":
            state = raw["autoencoder"]
        else:
            prefix = "target_encoder.encoder.autoencoder."
            state = {k[len(prefix):]: v for k, v in raw["model"].items() if k.startswith(prefix)}
        missing, unexpected = mvae.model.load_state_dict(state, strict=False)
        print(f"{model} : {len(state)} tenseurs chargés, {len(missing)} manquants, {len(unexpected)} inattendus")
        ae = mvae.model.to(device).eval()

    @torch.no_grad()
    def encode(x):
        feats = []
        handle = _penultimate_hook(ae.encoder, feats)
        try:
            z = ae.encode(x).mode()
        finally:
            handle.remove()
        feat = feats[0]
        if feat.shape[-1] != FEAT_SIZE:
            feat = F.adaptive_avg_pool2d(feat, FEAT_SIZE)
            z = F.adaptive_avg_pool2d(z, FEAT_SIZE)
        return feat, z

    return encode


@torch.no_grad()
def extract(encode, dataset, device, representation, batch_size=2):
    feats, masks = [], []
    for x, m in tqdm(DataLoader(dataset, batch_size=batch_size, num_workers=4), desc="features", leave=False):
        feat, z = encode(x.to(device) * 2 - 1)
        feats.append((feat if representation == "penultimate" else z).half().cpu())
        masks.append(m.to(torch.uint8))
    return torch.cat(feats), torch.cat(masks)


# ── Probe linéaire ───────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, channels, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.view(1, -1, 1, 1))
        self.register_buffer("std", std.view(1, -1, 1, 1))
        self.classifier = nn.Conv2d(channels, NUM_CLASSES, kernel_size=1)

    def forward(self, feat, out_size):
        logits = self.classifier((feat.float() - self.mean) / self.std)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


@torch.no_grad()
def score(probe, feats, masks, device):
    probe.eval()
    metrics = SegMetrics(num_classes=NUM_CLASSES, device=device)
    for f, m in DataLoader(TensorDataset(feats, masks), batch_size=8):
        metrics.update(probe(f.to(device), m.shape[-2:]), m.long().to(device))
    return metrics.compute()


def train_probe(train, val, test, device, seed, epochs, lr):
    torch.manual_seed(seed)
    feats = train[0].float()
    mean, std = feats.mean(dim=(0, 2, 3)), feats.std(dim=(0, 2, 3)).clamp_min(1e-6)
    probe = LinearProbe(feats.shape[1], mean, std).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = SegLoss(num_classes=NUM_CLASSES, dice_weight=0.5, ce_weight=0.5, cl_weight=0.0)
    loader = DataLoader(TensorDataset(*train), batch_size=8, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best = (-1.0, None, 0)
    for epoch in range(1, epochs + 1):
        probe.train()
        for f, m in loader:
            m = m.long().to(device)
            loss, _ = criterion(probe(f.to(device), m.shape[-2:]), m)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_dice = score(probe, *val, device)["dice_fg_mean"]
        if val_dice > best[0]:
            best = (val_dice, {k: v.clone() for k, v in probe.state_dict().items()}, epoch)
    probe.load_state_dict(best[1])
    r = score(probe, *test, device)
    return {"seed": seed, "best_epoch": best[2], "val_dice_fg": best[0],
            **{k: r[k] for k in ("dice_mean", "dice_fg_mean", "iou_fg_mean")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["medvae", "stage1", "stage2"], required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--representation", choices=["penultimate", "latent"], default="penultimate")
    parser.add_argument("--task", choices=["segments", "vessels"], default="segments")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.model != "medvae" and not args.ckpt:
        parser.error("--ckpt est requis pour stage1 / stage2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_config(str(REPO_ROOT / "finetune" / "configs" / "condition_a.yaml"))["data"]
    from finetune.config import arcade_root
    seg_val = arcade_root() / "dataset_phase_1" / "segmentation_dataset" / "seg_val"
    datasets = {
        "train": ArcadeDataset(d["train_images"], d["train_ann"]),
        "val": ArcadeDataset(str(seg_val / "images"), str(seg_val / "annotations" / "seg_val.json")),
        "test": ArcadeDataset(d["val_images"], d["val_ann"]),
    }

    encode = build_encoder(args.model, args.ckpt, device)
    data = {k: extract(encode, ds, device, args.representation) for k, ds in datasets.items()}
    if args.task == "vessels":
        global NUM_CLASSES
        NUM_CLASSES = 2
        data = {k: (f, (m > 0).to(torch.uint8)) for k, (f, m) in data.items()}
    print(f"features {args.representation} : {tuple(data['train'][0].shape[1:])}")

    runs = []
    for seed in args.seeds:
        r = train_probe(data["train"], data["val"], data["test"], device, seed, args.epochs, args.lr)
        print(r)
        runs.append(r)
    summary = {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs], ddof=1)) if len(runs) > 1 else 0.0]
               for k in ("dice_mean", "dice_fg_mean", "iou_fg_mean", "val_dice_fg")}
    result = {"model": args.model, "ckpt": args.ckpt, "representation": args.representation, "task": args.task,
              "epochs": args.epochs, "lr": args.lr, "runs": runs, "mean_std": summary}
    print(json.dumps(summary, indent=2))
    out = Path(args.out or REPO_ROOT / "experiments" / "jepa" / f"probe_{args.model}_{args.representation}_{args.task}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

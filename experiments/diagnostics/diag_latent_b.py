"""Diagnostic de la condition B : pourquoi la tête ne détecte-t-elle aucune artère ?

1. Bruit du latent : MVAE.encode renvoie un tirage aléatoire (moyenne + std * bruit).
2. Signal vasculaire : une valeur du latent distingue-t-elle vaisseau et fond ?
3. Capacité : la tête de B peut-elle sur-apprendre 8 images, selon son entrée
   (latent tiré, latent moyen, image réduite en 128×128 comme témoin) ?

Lancer depuis la racine du repo :
    PYTHONPATH=. python experiments/diagnostics/diag_latent_b.py
"""

import torch
import torch.nn.functional as F
from medvae import MVAE
from sklearn.metrics import roc_auc_score

from finetune.config import load_config
from finetune.dataset import ArcadeDataset, split_dataset
from finetune.losses.seg_loss import SegLoss
from finetune.metrics import SegMetrics
from finetune.models import build_seg_head

torch.manual_seed(0)
device = torch.device("cuda")
cfg = load_config("finetune/configs/condition_b.yaml")
d = cfg["data"]

train_ids, _ = split_dataset(d["train_ann"], d["train_ratio"], cfg["experiment"]["seed"])
ds = ArcadeDataset(d["train_images"], d["train_ann"], train_ids[:8], augment=False)
images = torch.stack([ds[i][0] for i in range(8)]).to(device)   # [8,1,512,512] dans [0,1]
masks = torch.stack([ds[i][1] for i in range(8)]).to(device)    # [8,512,512]

mvae = MVAE(model_name="medvae_4_1_2d", modality="xray").to(device).eval()
with torch.no_grad():
    # Par paquets de 2 : l'attention de MedVAE en 512×512 ne tient pas à 8 sur 24 Go
    posteriors = [mvae.model.encode(chunk * 2 - 1) for chunk in images.split(2)]
    mean = torch.cat([p.mean for p in posteriors])              # [8,1,128,128]
    std = torch.cat([p.std for p in posteriors])


def sample_latent():
    """Tirage aléatoire, comme MVAE.encode (moyenne + std * bruit)."""
    return mean + std * torch.randn_like(mean)


sample1, sample2 = sample_latent(), sample_latent()

# 1. Bruit
print("\n=== 1. Bruit du latent ===")
print(f"latent moyen : moyenne {mean.mean():.3f}, écart-type entre pixels {mean.std():.3f}")
print(f"écart-type du bruit (posterior.std) : moyenne {std.mean():.3f}")
print(f"rapport signal/bruit (std des moyennes / std du bruit) : {mean.std() / std.mean():.2f}")
print(f"écart entre deux tirages de la même image : {(sample1 - sample2).abs().mean():.3f}")

# 2. Signal vasculaire : vaisseau = au moins un pixel d'artère dans le bloc 4×4
vessel = F.max_pool2d((masks > 0).float().unsqueeze(1), 4).bool()
print("\n=== 2. Signal vasculaire (AUC vaisseau vs fond, 0.5 = aucun signal) ===")
print(f"part des blocs 4×4 contenant un vaisseau : {vessel.float().mean():.3f}")
y = vessel.flatten().cpu().numpy()
small = F.avg_pool2d(images, 4)
for name, feat in [("latent moyen", mean), ("latent tiré", sample1), ("image réduite", small)]:
    auc = roc_auc_score(y, feat.flatten().cpu().numpy())
    print(f"{name:14s} AUC = {max(auc, 1 - auc):.3f}")

# 3. Sur-apprentissage de 8 images
print("\n=== 3. Sur-apprentissage de 8 images (tête de B, 300 pas, Adam 1e-3) ===")


def overfit(name, get_input, lr=1e-3, steps=300):
    torch.manual_seed(0)
    head = build_seg_head(cfg["model"]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    crit = SegLoss(num_classes=26, dice_weight=0.5, ce_weight=0.5, cl_weight=0.0)
    for _ in range(steps):
        head.train()
        loss, _ = crit(head(get_input()), masks)
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    m = SegMetrics(num_classes=26, device=device)
    with torch.no_grad():
        m.update(head(get_input()), masks)
    r = m.compute()
    arteries = sum(x > 0.01 for x in r["dice_per_class"][1:])
    print(f"{name:14s} Dice {r['dice_mean']:.3f} | artères détectées {arteries}/25")


with torch.no_grad():
    small_norm = small * 2 - 1
overfit("latent tiré", sample_latent)
overfit("latent moyen", lambda: mean)
overfit("image réduite", lambda: small_norm)

# 4. Effet du learning rate et de la durée (latent tiré, comme en B)
print("\n=== 4. Learning rate et durée (latent tiré) ===")
overfit("lr 5e-5 (B)", sample_latent, lr=5e-5)
overfit("lr 1e-3 ×1000", sample_latent, lr=1e-3, steps=1000)

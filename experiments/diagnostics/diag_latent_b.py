"""Diagnosis of condition B: why does the head detect no artery?

1. Latent noise: MVAE.encode returns a random sample (mean + std * noise).
2. Vessel signal: does a latent value separate vessel and background?
3. Capacity: can the head of B overfit 8 images, depending on its input
   (sampled latent, mean latent, image reduced to 128×128 as a control)?

Run from the repository root:
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
images = torch.stack([ds[i][0] for i in range(8)]).to(device)   # [8,1,512,512] in [0,1]
masks = torch.stack([ds[i][1] for i in range(8)]).to(device)    # [8,512,512]

mvae = MVAE(model_name="medvae_4_1_2d", modality="xray").to(device).eval()
with torch.no_grad():
    # In groups of 2: MedVAE attention at 512×512 does not fit 8 images in 24 GB
    posteriors = [mvae.model.encode(chunk * 2 - 1) for chunk in images.split(2)]
    mean = torch.cat([p.mean for p in posteriors])              # [8,1,128,128]
    std = torch.cat([p.std for p in posteriors])


def sample_latent():
    """Random sample, like MVAE.encode (mean + std * noise)."""
    return mean + std * torch.randn_like(mean)


sample1, sample2 = sample_latent(), sample_latent()

# 1. Bruit
print("\n=== 1. Latent noise ===")
print(f"mean latent: mean {mean.mean():.3f}, standard deviation across pixels {mean.std():.3f}")
print(f"noise standard deviation (posterior.std): mean {std.mean():.3f}")
print(f"signal-to-noise ratio (std of the means / std of the noise): {mean.std() / std.mean():.2f}")
print(f"gap between two samples of the same image: {(sample1 - sample2).abs().mean():.3f}")

# 2. Vessel signal: vessel = at least one artery pixel in the 4×4 block
vessel = F.max_pool2d((masks > 0).float().unsqueeze(1), 4).bool()
print("\n=== 2. Vessel signal (AUC vessel vs background, 0.5 = no signal) ===")
print(f"share of 4×4 blocks containing a vessel: {vessel.float().mean():.3f}")
y = vessel.flatten().cpu().numpy()
small = F.avg_pool2d(images, 4)
for name, feat in [("mean latent", mean), ("sampled latent", sample1), ("reduced image", small)]:
    auc = roc_auc_score(y, feat.flatten().cpu().numpy())
    print(f"{name:14s} AUC = {max(auc, 1 - auc):.3f}")

# 3. Overfitting 8 images
print("\n=== 3. Overfitting 8 images (head of B, 300 steps, Adam 1e-3) ===")


def overfit(name, get_input, lr=1e-3, steps=300):
    torch.manual_seed(0)
    # Head of B at the time of E03 (SegHead, 128 channels), not that of the current config
    head = build_seg_head({**cfg["model"], "architecture": "seg_head", "base_channels": 128}).to(device)
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
    print(f"{name:14s} Dice {r['dice_mean']:.3f} | arteries detected {arteries}/25")


with torch.no_grad():
    small_norm = small * 2 - 1
overfit("sampled latent", sample_latent)
overfit("mean latent", lambda: mean)
overfit("reduced image", lambda: small_norm)

# 4. Effect of the learning rate and duration (sampled latent, as in B)
print("\n=== 4. Learning rate and duration (sampled latent) ===")
overfit("lr 5e-5 (B)", sample_latent, lr=5e-5)
overfit("lr 1e-3 ×1000", sample_latent, lr=1e-3, steps=1000)

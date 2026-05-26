from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JEPA_ROOT    = Path(__file__).resolve().parents[1]
MEDVAE_ROOT  = PROJECT_ROOT.parent / "MedVAE"


def _ensure_paths() -> None:
    for p in (MEDVAE_ROOT, JEPA_ROOT):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _register_medvae_stubs() -> None:
    stubs = {
        "medvae":               MEDVAE_ROOT / "medvae",
        "medvae.losses":        MEDVAE_ROOT / "medvae" / "losses",
        "medvae.utils":         MEDVAE_ROOT / "medvae" / "utils",
        "medvae.utils.vae":     MEDVAE_ROOT / "medvae" / "utils" / "vae",
    }
    for name, pkg_path in stubs.items():
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__    = [str(pkg_path)]
            m.__package__ = name
            sys.modules[name] = m


_ensure_paths()
_register_medvae_stubs()

from medvae.utils.vae.loss_components import (   # noqa: E402
    LPIPS,
    NLayerDiscriminator,
    hinge_loss,
    weights_init,
)
from models import AutoencoderKL2D               # noqa: E402


class _Phase1Loss(nn.Module):
    def __init__(
        self,
        disc_start:        int   = 50001,
        kl_weight:         float = 1e-6,
        disc_weight:       float = 0.5,
        perceptual_weight: float = 1.0,
        num_channels:      int   = 3,
    ):
        super().__init__()
        self.disc_start        = disc_start
        self.kl_weight         = kl_weight
        self.disc_weight       = disc_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss   = LPIPS().eval()
        self.logvar            = nn.Parameter(torch.zeros(()))
        self.discriminator     = NLayerDiscriminator(input_nc=num_channels).apply(weights_init)

    def forward(
        self,
        inputs:          torch.Tensor,
        reconstructions: torch.Tensor,
        posteriors,
        optimizer_idx:   int,
        global_step:     int,
        last_layer,
        split:           str = "train",
        **_,
    ):
        bsz      = inputs.shape[0]
        d_active = global_step >= self.disc_start

        if optimizer_idx == 0:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            p_loss   = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
            nll_loss = (rec_loss / torch.exp(self.logvar) + self.logvar).sum() / bsz

            kl_loss = posteriors.kl().sum() / bsz

            g_loss   = torch.tensor(0.0, device=inputs.device)
            d_weight = torch.tensor(0.0, device=inputs.device)
            if d_active and last_layer is not None:
                logits_fake = self.discriminator(reconstructions.contiguous())
                g_loss = -torch.mean(logits_fake)
                try:
                    nll_g  = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
                    g_g    = torch.autograd.grad(g_loss,   last_layer, retain_graph=True)[0]
                    d_weight = (torch.norm(nll_g) / (torch.norm(g_g) + 1e-4)).clamp(0, 1e4).detach()
                    d_weight = d_weight * self.disc_weight
                except RuntimeError:
                    pass

            loss = nll_loss + self.kl_weight * kl_loss + d_weight * float(d_active) * g_loss
            logs = {
                f"{split}/total_loss": loss.detach(),
                f"{split}/nll_loss":   nll_loss.detach(),
                f"{split}/kl_loss":    kl_loss.detach(),
                f"{split}/g_loss":     g_loss.detach(),
                f"{split}/d_weight":   d_weight.detach(),
            }
            return loss, logs

        elif optimizer_idx == 1:
            if d_active:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
                d_loss = hinge_loss(logits_real, logits_fake)
            else:
                d_loss = torch.tensor(0.0, device=inputs.device)
            logs = {f"{split}/disc_loss": d_loss.detach()}
            return d_loss, logs


def _resolve(path: Optional[str | Path]) -> Optional[Path]:
    if path in (None, ""):
        return None
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _build_optimizer(params, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = cfg.get("name", "adamw").lower()
    lr   = float(cfg.get("lr", 1e-4))
    wd   = float(cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {name!r}")


def _set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_autoencoder(
    model_cfg: Dict[str, Any],
    phase1_ckpt: Optional[str | Path] = None,
) -> AutoencoderKL2D:
    ae = AutoencoderKL2D(
        ddconfig         = model_cfg["ddconfig"],
        embed_dim        = int(model_cfg.get("embed_dim", 4)),
        ckpt_path        = None,
        ignore_keys      = model_cfg.get("ignore_keys", []),
        apply_channel_ds = bool(model_cfg.get("apply_channel_ds", True)),
    )
    if phase1_ckpt is not None:
        raw = torch.load(_resolve(phase1_ckpt), map_location="cpu")
        sd  = raw.get("autoencoder", raw.get("state_dict", raw))
        missing, unexpected = ae.load_state_dict(sd, strict=False)
        print(f"[ckpt] loaded ({len(missing)} missing, {len(unexpected)} unexpected)")
    return ae


class Phase1Trainer:
    def __init__(
        self,
        config: Dict[str, Any],
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ):
        self.config   = config
        train_cfg     = config["training"]
        dev           = train_cfg.get("device", "auto")
        _dev          = ("cuda" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev
        self.device   = torch.device(_dev)
        _set_seed(int(train_cfg.get("seed", 42)))

        self.ae = _build_autoencoder(config["model"]).to(self.device)

        loss_cfg      = train_cfg.get("loss", {})
        self.criterion = _Phase1Loss(
            disc_start        = int(loss_cfg.get("disc_start", 50001)),
            kl_weight         = float(loss_cfg.get("kl_weight", 1e-6)),
            disc_weight       = float(loss_cfg.get("disc_weight", 0.5)),
            perceptual_weight = float(loss_cfg.get("perceptual_weight", 1.0)),
            num_channels      = int(loss_cfg.get("num_channels", 3)),
        ).to(self.device)
        self.disc_start = int(loss_cfg.get("disc_start", 50001))

        self.opt_ae   = _build_optimizer(list(self.ae.parameters()),
                                          train_cfg.get("optimizer", {}))
        self.opt_disc = _build_optimizer(list(self.criterion.discriminator.parameters()),
                                          train_cfg.get("disc_optimizer",
                                                        train_cfg.get("optimizer", {})))

        loader_cfg = config["data"].get("loader", {})
        bs, nw, pm = (int(loader_cfg.get("batch_size", 16)),
                      int(loader_cfg.get("num_workers", 0)),
                      bool(loader_cfg.get("pin_memory", True)))
        self.train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                       num_workers=nw, pin_memory=pm, drop_last=True)
        self.val_loader   = (DataLoader(val_dataset, batch_size=bs, shuffle=False,
                                        num_workers=nw, pin_memory=pm)
                             if val_dataset is not None else None)

        self.output_dir = _resolve(train_cfg.get("output_dir", "outputs/pretraining/phase1"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_val    = float("inf")
        self.global_step = 0
        self.history: Dict[str, list] = {"epoch": []}

    def _to3(self, x: torch.Tensor) -> torch.Tensor:
        return x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x

    def save_checkpoint(self, name: str, epoch: int, metrics: dict) -> None:
        torch.save({
            "epoch":       epoch,
            "autoencoder": self.ae.state_dict(),
            "opt_ae":      self.opt_ae.state_dict(),
            "opt_disc":    self.opt_disc.state_dict(),
            "criterion":   self.criterion.state_dict(),
            "metrics":     metrics,
            "config":      self.config,
        }, self.output_dir / name)

    def fit(self) -> None:
        train_cfg  = self.config["training"]
        epochs     = int(train_cfg.get("epochs", 30))
        log_every  = int(train_cfg.get("log_every", 200))
        save_every = int(train_cfg.get("save_every", 5))

        for epoch in range(1, epochs + 1):
            self.ae.train()
            self.criterion.train()
            running: Dict[str, float] = {}
            n = 0
            epoch_totals: Dict[str, float] = {}
            epoch_n = 0

            for i, x in enumerate(self.train_loader):
                x    = x.to(self.device, non_blocking=True).float()
                x3   = self._to3(x)
                compute_disc = self.global_step >= self.disc_start

                if (compute_disc and i % 2 == 0) or not compute_disc:
                    rec, posterior, latent = self.ae(x, sample_posterior=True, decode=True)
                    loss, logs = self.criterion(
                        inputs=x3, reconstructions=self._to3(rec),
                        posteriors=posterior, optimizer_idx=0,
                        global_step=self.global_step,
                        last_layer=self.ae.get_last_layer(), split="train",
                    )
                    self.opt_ae.zero_grad(set_to_none=True)
                    loss.backward()
                    self.opt_ae.step()
                    self.global_step += 1

                elif compute_disc and i % 2 == 1:
                    with torch.no_grad():
                        rec, posterior, _ = self.ae(x, sample_posterior=True, decode=True)
                    loss, logs = self.criterion(
                        inputs=x3, reconstructions=self._to3(rec),
                        posteriors=posterior, optimizer_idx=1,
                        global_step=self.global_step,
                        last_layer=None, split="train",
                    )
                    self.opt_disc.zero_grad(set_to_none=True)
                    loss.backward()
                    self.opt_disc.step()

                for k, v in logs.items():
                    fv = float(v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                    running[k]      = running.get(k, 0.0) + fv
                    epoch_totals[k] = epoch_totals.get(k, 0.0) + fv
                n += 1
                epoch_n += 1

                if (i + 1) % log_every == 0:
                    avg = {k: v / n for k, v in running.items()}
                    running, n = {}, 0
                    key = " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in sorted(avg.items())
                                   if any(s in k for s in ("total", "nll", "kl", "disc")))
                    print(f"epoch={epoch} batch={i+1} gs={self.global_step} {key}")

            epoch_train = {k: v / max(epoch_n, 1) for k, v in epoch_totals.items()}

            val_m    = self._validate()
            val_loss = val_m.get("val/nll_loss", float("inf"))
            if val_m:
                print(f"epoch={epoch} val_nll={val_loss:.5f}")
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.save_checkpoint("best.pt", epoch, val_m)
            if epoch % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch:04d}.pt", epoch, val_m)

            self.history["epoch"].append(epoch)
            for k, v in epoch_train.items():
                self.history.setdefault(k, []).append(v)
            for k, v in val_m.items():
                self.history.setdefault(k, []).append(v)

            self._save_recon_example(epoch)

        self.save_checkpoint("last.pt", epochs, {"best_val": self.best_val})
        print(f"[Phase1Trainer] Done. Best val_nll: {self.best_val:.5f}")
        self._plot_training_curves()

    @torch.no_grad()
    def _save_recon_example(self, epoch: int, n_samples: int = 8) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        loader = self.val_loader if self.val_loader is not None else self.train_loader
        self.ae.eval()
        x = next(iter(loader))
        x = x[:n_samples].to(self.device, non_blocking=True).float()
        rec, _, _ = self.ae(x, sample_posterior=False, decode=True)

        # Denormalise from [-1,1] to [0,1]
        x_np   = (x.clamp(-1, 1) * 0.5 + 0.5).cpu()
        rec_np = (rec.clamp(-1, 1) * 0.5 + 0.5).cpu()

        n = x_np.shape[0]
        fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
        for j in range(n):
            img_in  = x_np[j].permute(1, 2, 0).numpy()
            img_rec = rec_np[j].permute(1, 2, 0).numpy()
            if img_in.shape[2] == 1:
                img_in  = img_in[:, :, 0]
                img_rec = img_rec[:, :, 0]
            axes[0, j].imshow(img_in,  cmap="gray" if img_in.ndim == 2 else None, vmin=0, vmax=1)
            axes[1, j].imshow(img_rec, cmap="gray" if img_rec.ndim == 2 else None, vmin=0, vmax=1)
            axes[0, j].axis("off")
            axes[1, j].axis("off")
        axes[0, 0].set_ylabel("input",   fontsize=9)
        axes[1, 0].set_ylabel("recon",   fontsize=9)
        fig.suptitle(f"Epoch {epoch}", fontsize=10)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"recon_epoch_{epoch:04d}.png", dpi=100)
        plt.close(fig)

    def _plot_training_curves(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = self.history.get("epoch", [])
        if not epochs:
            return

        metrics_to_plot = [
            ("train/total_loss", "Train total loss"),
            ("train/nll_loss",   "Train NLL loss"),
            ("train/kl_loss",    "Train KL loss"),
            ("train/disc_loss",  "Train disc loss"),
            ("val/total_loss",   "Val total loss"),
            ("val/nll_loss",     "Val NLL loss"),
        ]
        available = [(k, lbl) for k, lbl in metrics_to_plot if k in self.history and self.history[k]]
        if not available:
            return

        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
        if len(available) == 1:
            axes = [axes]
        for ax, (key, label) in zip(axes, available):
            ax.plot(epochs, self.history[key], marker="o", markersize=3)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_dir / "training_curves.png", dpi=120)
        plt.close(fig)
        print(f"[Phase1Trainer] Training curves saved to {self.output_dir / 'training_curves.png'}")

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.ae.eval()
        self.criterion.eval()
        totals: Dict[str, float] = {}
        count = 0
        for x in self.val_loader:
            x = x.to(self.device, non_blocking=True).float()
            rec, posterior, _ = self.ae(x, sample_posterior=False, decode=True)
            _, logs = self.criterion(
                inputs=self._to3(x), reconstructions=self._to3(rec),
                posteriors=posterior, optimizer_idx=0,
                global_step=self.global_step,
                last_layer=self.ae.get_last_layer(), split="val",
            )
            for k, v in logs.items():
                totals[k] = totals.get(k, 0.0) + float(
                    v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            count += 1
        return {k: v / max(count, 1) for k, v in totals.items()}


class Phase2CLIPTrainer:
    def __init__(
        self,
        config: Dict[str, Any],
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        phase1_ckpt: Optional[str | Path] = None,
    ):
        try:
            from medvae.losses.vae_losses import BiomedClipLoss
        except ImportError as e:
            raise ImportError(
                "Phase 2 CLIP requires open_clip. Install it with:\n"
                "  pip install open-clip-torch\n"
                f"(original error: {e})"
            ) from e

        self.config   = config
        train_cfg     = config["training"]
        dev           = train_cfg.get("device", "auto")
        _dev          = ("cuda" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev
        self.device   = torch.device(_dev)
        _set_seed(int(train_cfg.get("seed", 42)))

        self.ae = _build_autoencoder(config["model"], phase1_ckpt).to(self.device)
        self.ae.encoder.requires_grad_(False)
        self.ae.decoder.requires_grad_(False)
        self.ae.quant_conv.requires_grad_(False)
        self.ae.post_quant_conv.requires_grad_(False)

        self.criterion = BiomedClipLoss(
            compute_rec_loss=False, compute_lat_loss=True
        ).to(self.device)

        trainable = (
            list(self.ae.channel_ds.parameters())
            + list(self.ae.channel_proj.parameters())
        )
        self.opt  = _build_optimizer(trainable, train_cfg.get("optimizer", {}))

        loader_cfg    = config["data"].get("loader", {})
        bs, nw, pm    = (int(loader_cfg.get("batch_size", 16)),
                         int(loader_cfg.get("num_workers", 0)),
                         bool(loader_cfg.get("pin_memory", True)))
        self.train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                       num_workers=nw, pin_memory=pm, drop_last=False)
        self.val_loader   = (DataLoader(val_dataset, batch_size=bs, shuffle=False,
                                        num_workers=nw, pin_memory=pm)
                             if val_dataset is not None else None)

        self.output_dir  = _resolve(train_cfg.get("output_dir", "outputs/pretraining/phase2_clip"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_val    = float("inf")
        self.global_step = 0

    def save_checkpoint(self, name: str, epoch: int, metrics: dict) -> None:
        torch.save({
            "epoch":       epoch,
            "autoencoder": self.ae.state_dict(),
            "opt":         self.opt.state_dict(),
            "metrics":     metrics,
            "config":      self.config,
        }, self.output_dir / name)

    def fit(self) -> None:
        train_cfg  = self.config["training"]
        epochs     = int(train_cfg.get("epochs", 15))
        log_every  = int(train_cfg.get("log_every", 100))
        save_every = int(train_cfg.get("save_every", 5))

        for epoch in range(1, epochs + 1):
            self.ae.train()
            running = 0.0

            for i, x in enumerate(self.train_loader):
                x = x.to(self.device, non_blocking=True).float()
                _, _, latent = self.ae(x, sample_posterior=True, decode=False)
                loss = self.criterion(x, latent=latent).sum() / x.shape[0]
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                running          += float(loss.detach().cpu())
                self.global_step += 1

                if (i + 1) % log_every == 0:
                    print(f"epoch={epoch} batch={i+1} gs={self.global_step} "
                          f"bc_loss={running / log_every:.5f}")
                    running = 0.0

            val_loss = self._validate()
            print(f"epoch={epoch} val_bc_loss={val_loss:.5f}")
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.save_checkpoint("best.pt", epoch, {"val_bc_loss": val_loss})
            if epoch % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch:04d}.pt", epoch, {"val_bc_loss": val_loss})

        self.save_checkpoint("last.pt", epochs, {"best_val": self.best_val})
        print(f"[Phase2CLIPTrainer] Done. Best val_bc_loss: {self.best_val:.5f}")

    @torch.no_grad()
    def _validate(self) -> float:
        if self.val_loader is None:
            return float("inf")
        self.ae.eval()
        total, count = 0.0, 0
        for x in self.val_loader:
            x = x.to(self.device, non_blocking=True).float()
            _, _, latent = self.ae(x, sample_posterior=False, decode=False)
            loss  = self.criterion(x, latent=latent).sum() / x.shape[0]
            total += float(loss.cpu())
            count += 1
        return total / max(count, 1)

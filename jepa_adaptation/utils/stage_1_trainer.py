from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from ..models.stage_1_loss import Stage1Loss
except ImportError:
    from models.stage_1_loss import Stage1Loss

try:
    from .launch import print_loader_sizes, print_model_summary
except ImportError:
    from launch import print_loader_sizes, print_model_summary


def _build_single_optimizer(params, cfg):
    name = cfg.get("name", "adamw").lower()
    lr = float(cfg.get("lr", 1e-4))
    wd = float(cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {name!r}")


class TrainerStage1:
    def __init__(self, model, config, train_dataset, val_dataset=None):
        self.config = config
        self.model = model
        self.ae = model.model

        train_cfg = config.get("training", {})
        dev = train_cfg.get("device", "auto")
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)

        seed = int(train_cfg.get("seed", 42))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.amp = bool(train_cfg.get("amp", False))
        self.ae.to(self.device)
        self.criterion = self._build_criterion().to(self.device)
        self.disc_start = self.criterion.adversarial_loss.discriminator_iter_start

        self.opt_ae, self.opt_disc = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        loader_cfg = config.get("data", {}).get("loader", {})
        bs = int(loader_cfg.get("batch_size", 16))
        nw = int(loader_cfg.get("num_workers", 0))
        pm = bool(loader_cfg.get("pin_memory", True))
        self.train_loader = DataLoader(
            train_dataset, batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=pm, drop_last=True,
        )
        self.val_loader = (
            DataLoader(val_dataset, batch_size=bs, shuffle=False,
                       num_workers=nw, pin_memory=pm)
            if val_dataset is not None else None
        )

        self.output_dir = Path(train_cfg.get("output_dir", "outputs/stage1"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_val = float("inf")
        self.global_step = 0

        print_model_summary({
            "encoder": self.ae.encoder,
            "decoder": self.ae.decoder,
            "quant_conv": self.ae.quant_conv,
            "post_quant_conv": self.ae.post_quant_conv,
            "channel_ds": getattr(self.ae, "channel_ds", None),
            "channel_proj": getattr(self.ae, "channel_proj", None),
            "discriminator": self.criterion.adversarial_loss.discriminator,
            "autoencoder": self.ae,
        })
        print_loader_sizes(self.train_loader, self.val_loader)

    def _build_criterion(self):
        loss_cfg = self.config.get("training", {}).get("loss", {})
        return Stage1Loss(
            disc_start=int(loss_cfg.get("disc_start", 50001)),
            kl_weight=float(loss_cfg.get("kl_weight", 1e-6)),
            perceptual_weight=float(loss_cfg.get("perceptual_weight", 1.0)),
            disc_weight=float(loss_cfg.get("disc_weight", 0.5)),
            num_channels=int(loss_cfg.get("num_channels", 3)),
            learn_logvar=bool(loss_cfg.get("learn_logvar", False)),
        )

    def _build_optimizer(self):
        train_cfg = self.config.get("training", {})
        ae_params = [p for p in self.ae.parameters() if p.requires_grad]
        if self.criterion.logvar.requires_grad:
            ae_params.append(self.criterion.logvar)

        opt_ae = _build_single_optimizer(ae_params, train_cfg.get("optimizer", {}))
        opt_disc = _build_single_optimizer(
            list(self.criterion.adversarial_loss.discriminator.parameters()),
            train_cfg.get("disc_optimizer", train_cfg.get("optimizer", {})),
        )
        return opt_ae, opt_disc

    def _build_scheduler(self):
        sched_cfg = self.config.get("training", {}).get("scheduler")
        if not sched_cfg:
            return None
        t_max = int(sched_cfg.get("t_max", self.config.get("training", {}).get("epochs", 30)))
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_ae, T_max=t_max)

    def _autocast(self):
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.amp and self.device.type == "cuda",
        )

    def _to3(self, x):
        return x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x

    def save_checkpoint(self, name, epoch, metrics):
        torch.save(
            {
                "epoch": epoch,
                "autoencoder": self.ae.state_dict(),
                "criterion": self.criterion.state_dict(),
                "opt_ae": self.opt_ae.state_dict(),
                "opt_disc": self.opt_disc.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            self.output_dir / name,
        )

    def fit(self):
        train_cfg = self.config.get("training", {})
        epochs = int(train_cfg.get("epochs", 30))
        log_every = int(train_cfg.get("log_every", 100))
        save_every = int(train_cfg.get("save_every", 5))

        for epoch in range(1, epochs + 1):
            self.ae.train()
            self.criterion.train()
            running, n = {}, 0

            for i, x in enumerate(self.train_loader):
                x = x.to(self.device, non_blocking=True).float()
                x3 = self._to3(x)
                disc_active = self.global_step >= self.disc_start

                if not disc_active or i % 2 == 0:
                    with self._autocast():
                        rec, posterior, _ = self.ae(x, sample_posterior=True, decode=True)
                        loss, logs = self.criterion(
                            x3, self._to3(rec), posterior,
                            optimizer_idx=0, global_step=self.global_step,
                            last_layer=self.ae.get_last_layer(), split="train",
                        )
                    self.opt_ae.zero_grad(set_to_none=True)
                    loss.backward()
                    self.opt_ae.step()
                    self.global_step += 1
                else:
                    with torch.no_grad(), self._autocast():
                        rec, posterior, _ = self.ae(x, sample_posterior=True, decode=True)
                    with self._autocast():
                        loss, logs = self.criterion(
                            x3, self._to3(rec), posterior,
                            optimizer_idx=1, global_step=self.global_step,
                            last_layer=None, split="train",
                        )
                    self.opt_disc.zero_grad(set_to_none=True)
                    loss.backward()
                    self.opt_disc.step()

                for k, v in logs.items():
                    running[k] = running.get(k, 0.0) + float(v)
                n += 1

                if (i + 1) % log_every == 0:
                    avg = {k: v / n for k, v in running.items()}
                    running, n = {}, 0
                    summary = " ".join(
                        f"{k.split('/')[-1]}={v:.4f}"
                        for k, v in sorted(avg.items())
                        if any(s in k for s in ("total", "nll", "kl", "disc"))
                    )
                    print(f"epoch={epoch} batch={i+1} gs={self.global_step} {summary}")

            if self.scheduler is not None:
                self.scheduler.step()

            val_m = self.validate()
            val_loss = val_m.get("val/nll_loss", float("inf"))
            if val_m:
                print(f"epoch={epoch} val_nll={val_loss:.5f}")
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.save_checkpoint("best.pt", epoch, val_m)
            if epoch % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch:04d}.pt", epoch, val_m)

            self._save_reconstructions(epoch, n=2)

        self.save_checkpoint("last.pt", epochs, {"best_val": self.best_val})
        print(f"[TrainerStage1] Done. Best val_nll: {self.best_val:.5f}")

    @torch.no_grad()
    def _save_reconstructions(self, epoch, n=2):
        if self.val_loader is None:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.ae.eval()
        x = next(iter(self.val_loader))[:n].to(self.device, non_blocking=True).float()
        with self._autocast():
            rec, _, _ = self.ae(x, sample_posterior=False, decode=True)
        x = (x.clamp(-1, 1) * 0.5 + 0.5).float().cpu()
        rec = (rec.clamp(-1, 1) * 0.5 + 0.5).float().cpu()

        for j in range(min(n, x.shape[0])):
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            for ax, img, title in zip(axes, (x[j], rec[j]), ("input", "recon")):
                arr = img.permute(1, 2, 0).numpy()
                if arr.shape[2] == 1:
                    arr = arr[:, :, 0]
                ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None, vmin=0, vmax=1)
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle(f"epoch {epoch}")
            fig.tight_layout()
            fig.savefig(self.output_dir / f"recon_epoch{epoch:04d}_{j}.png", dpi=100)
            plt.close(fig)

    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return {}
        self.ae.eval()
        self.criterion.eval()
        totals, count = {}, 0
        for x in self.val_loader:
            x = x.to(self.device, non_blocking=True).float()
            with self._autocast():
                rec, posterior, _ = self.ae(x, sample_posterior=False, decode=True)
                _, logs = self.criterion(
                    self._to3(x), self._to3(rec), posterior,
                    optimizer_idx=0, global_step=self.global_step,
                    last_layer=self.ae.get_last_layer(), split="val",
                )
            for k, v in logs.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            count += 1
        return {k: v / max(count, 1) for k, v in totals.items()}

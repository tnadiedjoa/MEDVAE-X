from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from ..models.stage_2_loss import Stage2Loss
except ImportError:
    from models.stage_2_loss import Stage2Loss

try:
    from .launch import print_loader_sizes, print_model_summary
except ImportError:
    from launch import print_loader_sizes, print_model_summary


class TrainerStage2:
    def __init__(self, model, config, train_dataset, val_dataset=None):
        self.config = config
        self.model = model

        train_cfg = config.get("training", {})
        dev = train_cfg.get("device", "auto")
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)

        seed = int(train_cfg.get("seed", 42))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.amp = bool(train_cfg.get("amp", False))
        self.model.to(self.device)
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.max_grad_norm = train_cfg.get("max_grad_norm")

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

        self.output_dir = Path(train_cfg.get("output_dir", "outputs/stage2"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_val = float("inf")
        self.global_step = 0

        print_model_summary({
            "context_encoder": self.model.context_encoder,
            "predictor": self.model.predictor,
            "target_encoder": self.model.target_encoder,
            "mvae_jepa": self.model,
        })
        print_loader_sizes(self.train_loader, self.val_loader)

    def _build_criterion(self):
        loss_cfg = self.config.get("training", {}).get("loss", {})
        return Stage2Loss(
            loss_type=loss_cfg.get("loss_type", "smooth_l1"),
            normalize=bool(loss_cfg.get("normalize", True)),
            sigreg_weight=float(loss_cfg.get("sigreg_weight", 1.0)),
            num_projections=int(loss_cfg.get("num_projections", 64)),
            max_tokens=int(loss_cfg.get("max_tokens", 512)),
        )

    def _build_optimizer(self):
        opt_cfg = self.config.get("training", {}).get("optimizer", {})
        name = opt_cfg.get("name", "adamw").lower()
        lr = float(opt_cfg.get("lr", 1e-4))
        wd = float(opt_cfg.get("weight_decay", 1e-4))
        params = [p for p in self.model.trainable_parameters() if p.requires_grad]
        if name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        raise ValueError(f"Unsupported optimizer: {name!r}")

    def _build_scheduler(self):
        sched_cfg = self.config.get("training", {}).get("scheduler")
        if not sched_cfg:
            return None
        t_max = int(sched_cfg.get("t_max", self.config.get("training", {}).get("epochs", 30)))
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max)

    def _autocast(self):
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.amp and self.device.type == "cuda",
        )

    def save_checkpoint(self, name, epoch, metrics):
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            self.output_dir / name,
        )

    def train_step(self, x):
        self.model.train()
        x = x.to(self.device, non_blocking=True).float()
        with self._autocast():
            outputs = self.model(x)
            loss, logs = self.criterion(outputs, split="train")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.trainable_parameters(), float(self.max_grad_norm)
            )
        self.optimizer.step()
        self.model.update_target_encoder()

        metrics = {k: float(v) for k, v in logs.items()}
        metrics["loss"] = float(loss.detach())
        return metrics

    def fit(self):
        train_cfg = self.config.get("training", {})
        epochs = int(train_cfg.get("epochs", 30))
        log_every = int(train_cfg.get("log_every", 50))
        save_every = int(train_cfg.get("save_every", 5))

        for epoch in range(1, epochs + 1):
            running, n = 0.0, 0
            for i, x in enumerate(self.train_loader):
                metrics = self.train_step(x)
                running += metrics["loss"]
                n += 1
                self.global_step += 1

                if (i + 1) % log_every == 0:
                    print(
                        f"epoch={epoch} batch={i+1} gs={self.global_step} "
                        f"loss={running / n:.5f}"
                    )
                    running, n = 0.0, 0

            if self.scheduler is not None:
                self.scheduler.step()

            val_m = self.validate()
            val_loss = val_m.get("loss", float("inf"))
            if val_m:
                print(f"epoch={epoch} val_loss={val_loss:.5f}")
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.save_checkpoint("best.pt", epoch, val_m)
            if epoch % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch:04d}.pt", epoch, val_m)

        self.save_checkpoint("last.pt", epochs, {"best_val": self.best_val})
        print(f"[TrainerStage2] Done. Best val_loss: {self.best_val:.5f}")

    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return {}
        self.model.eval()
        totals, count = {}, 0
        for x in self.val_loader:
            x = x.to(self.device, non_blocking=True).float()
            with self._autocast():
                outputs = self.model(x)
                loss, logs = self.criterion(outputs, split="val")
            metrics = {k: float(v) for k, v in logs.items()}
            metrics["loss"] = float(loss.detach())
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            count += 1
        return {k: v / max(count, 1) for k, v in totals.items()}

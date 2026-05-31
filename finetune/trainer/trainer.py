import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from finetune.losses.seg_loss import SegLoss
from finetune.metrics.seg_metrics import SegMetrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
    ):
        self.model  = model.to(device)
        self.config = config
        self.device = device

        train_cfg = config["training"]
        loss_cfg  = config["loss"]
        data_cfg  = config["data"]
        log_cfg   = config["logging"]

        # Optimizer 
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        # Scheduler 
        self.scheduler = self._build_scheduler()

        # Loss 
        self.criterion = SegLoss(
            num_classes=data_cfg["num_classes"],
            dice_weight=loss_cfg["dice_weight"],
            ce_weight=loss_cfg["ce_weight"],
            cl_weight=loss_cfg.get("cl_weight", 0.2),
            skel_iters=loss_cfg.get("skel_iters", 5),
            ignore_index=loss_cfg.get("ignore_index", None),
        )

        # Métriques 
        self.train_metrics = SegMetrics(num_classes=data_cfg["num_classes"], device=device)
        self.val_metrics   = SegMetrics(num_classes=data_cfg["num_classes"], device=device)

        # Early stopping
        self.patience      = train_cfg["early_stopping_patience"]
        self.best_dice     = 0.0
        self.epochs_no_imp = 0

        # Gradient clipping 
        self.grad_clip = train_cfg.get("grad_clip", 1.0)

        # Dossiers de sauvegarde 
        self.save_dir = log_cfg["save_dir"]
        os.makedirs(self.save_dir, exist_ok=True)

        # Historique des métriques 
        self.history = {"train": [], "val": []}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:

        num_epochs = self.config["training"]["epochs"]

        for epoch in range(1, num_epochs + 1):

            print(f"\nEpoch {epoch}/{num_epochs}")

            train_results = self._train_one_epoch(train_loader)
            val_results   = self._val_one_epoch(val_loader)

            self._step_scheduler(val_results["dice_mean"])

            # Sauvegarde des métriques dans l'historique
            self.history["train"].append({
                "epoch":     epoch,
                "loss":      train_results["loss"],
                "dice_loss": train_results["dice_loss"],
                "ce_loss":   train_results["ce_loss"],
                "dice_mean": train_results["dice_mean"],
                "iou_mean":  train_results["iou_mean"],
                "lr":        self.optimizer.param_groups[0]["lr"],
                "cl_loss": train_results["cl_loss"],   
            })
            self.history["val"].append({
                "epoch":     epoch,
                "loss":      val_results["loss"],
                "dice_loss": val_results["dice_loss"],
                "ce_loss":   val_results["ce_loss"],
                "dice_mean": val_results["dice_mean"],
                "iou_mean":  val_results["iou_mean"],
                "cl_loss": val_results["cl_loss"],
            })

            # Affichage console
            print(f"  Train — loss: {train_results['loss']:.4f}  "
                  f"dice: {train_results['dice_mean']:.4f}")
            print(f"  Val   — loss: {val_results['loss']:.4f}  "
                  f"dice: {val_results['dice_mean']:.4f}")

            # Early stopping + sauvegarde checkpoint
            if val_results["dice_mean"] > self.best_dice:
                self.best_dice     = val_results["dice_mean"]
                self.epochs_no_imp = 0
                self._save_checkpoint(epoch, val_results["dice_mean"])
                print(f"  ✓ Meilleur Dice val : {self.best_dice:.4f} — checkpoint sauvegardé")
            else:
                self.epochs_no_imp += 1
                print(f"  Pas d'amélioration ({self.epochs_no_imp}/{self.patience})")
                if self.epochs_no_imp >= self.patience:
                    print(f"\nEarly stopping déclenché à l'epoch {epoch}.")
                    break

        # Sauvegarde de l'historique complet dans un fichier JSON
        history_path = os.path.join(
            self.save_dir,
            f"{self.config['experiment']['name']}_history.json"
        )
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nHistorique sauvegardé : {history_path}")

    def _train_one_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        self.train_metrics.reset()

        total_loss = 0.0
        total_dice = 0.0
        total_ce   = 0.0
        total_cl   = 0.0

        for images, masks in loader:
            images = images.to(self.device)
            masks  = masks.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)

            loss, loss_dict = self.criterion(logits, masks)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            self.train_metrics.update(logits.detach(), masks)
            total_loss += loss_dict["loss"]
            total_dice += loss_dict["dice_loss"]
            total_ce   += loss_dict["ce_loss"]
            total_cl += loss_dict["cl_loss"]   # ajoute cette ligne

        n = len(loader)
        metric_results = self.train_metrics.compute()

        return {
            "loss":      total_loss / n,
            "dice_loss": total_dice / n,
            "ce_loss":   total_ce   / n,
            "dice_mean": metric_results["dice_mean"],
            "iou_mean":  metric_results["iou_mean"],
            "cl_loss": total_cl / n,           
        }


    def _val_one_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        self.val_metrics.reset()

        total_loss = 0.0
        total_dice = 0.0
        total_ce   = 0.0
        total_cl   = 0.0

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks  = masks.to(self.device)

                logits = self.model(images)
                _, loss_dict = self.criterion(logits, masks)
                self.val_metrics.update(logits, masks)

                total_loss += loss_dict["loss"]
                total_dice += loss_dict["dice_loss"]
                total_ce   += loss_dict["ce_loss"]
                total_cl += loss_dict["cl_loss"]   # ajoute cette ligne

        n = len(loader)
        metric_results = self.val_metrics.compute()

        return {
            "loss":      total_loss / n,
            "dice_loss": total_dice / n,
            "ce_loss":   total_ce   / n,
            "dice_mean": metric_results["dice_mean"],
            "iou_mean":  metric_results["iou_mean"],
            "cl_loss": total_cl / n,
        }


    def evaluate(self, test_loader: DataLoader) -> dict:
        ckpt_path = os.path.join(self.save_dir, "best_model.pth")
        self._load_checkpoint(ckpt_path)

        self.model.eval()
        test_metrics = SegMetrics(
            num_classes=self.config["data"]["num_classes"],
            device=self.device,
        )

        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(self.device)
                masks  = masks.to(self.device)
                logits = self.model(images)
                test_metrics.update(logits, masks)

        results = test_metrics.compute()

        print(f"\n=== Résultats test set ===")
        print(f"Dice mean : {results['dice_mean']:.4f}")
        print(f"IoU  mean : {results['iou_mean']:.4f}")

        # Sauvegarde dans le fichier results.json central
        results_path = os.path.join(self.save_dir, "results.json")

        # Charge le fichier existant s'il existe déjà (pour ne pas écraser B ou C)
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                all_results = json.load(f)
        else:
            all_results = {}

        # Ajoute les résultats de la condition courante
        condition = self.config["experiment"]["condition"]  # "A", "B" ou "C"
        all_results[f"condition_{condition.lower()}"] = {
            "dice_mean":      results["dice_mean"],
            "iou_mean":       results["iou_mean"],
            "dice_per_class": results["dice_per_class"],
        }

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Résultats sauvegardés dans : {results_path}")

        return results


    def _build_scheduler(self):
        sched_cfg = self.config["scheduler"]
        name      = sched_cfg["name"]

        if name == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=sched_cfg["cosine"]["T_max"],
                eta_min=sched_cfg["cosine"]["eta_min"],
            )
        elif name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=sched_cfg["plateau"]["factor"],
                patience=sched_cfg["plateau"]["patience"],
                min_lr=sched_cfg["plateau"]["min_lr"],
            )
        else:
            raise ValueError(f"Scheduler inconnu : {name}")

    def _step_scheduler(self, val_dice: float) -> None:
        name = self.config["scheduler"]["name"]
        if name == "cosine":
            self.scheduler.step()
        elif name == "plateau":
            self.scheduler.step(val_dice)

    def _save_checkpoint(self, epoch: int, dice: float) -> None:
        path = os.path.join(self.save_dir, "best_model.pth")
        torch.save({
            "epoch":     epoch,
            "dice":      dice,
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config":    self.config,
        }, path)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        print(f"Checkpoint chargé : epoch {ckpt['epoch']}, Dice {ckpt['dice']:.4f}")

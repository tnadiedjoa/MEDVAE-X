import copy
import sys
import types
from pathlib import Path

JEPA_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEDVAE_ROOT = PROJECT_ROOT.parent / "MedVAE"


def bootstrap():
    for p in (MEDVAE_ROOT, JEPA_ROOT):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    # Sans clone local de MedVAE, on utilise le paquet pip `medvae` : déclarer des
    # modules vers un dossier inexistant masquerait le paquet installé.
    if not (MEDVAE_ROOT / "medvae").exists():
        return

    stubs = {
        "medvae": MEDVAE_ROOT / "medvae",
        "medvae.losses": MEDVAE_ROOT / "medvae" / "losses",
        "medvae.utils": MEDVAE_ROOT / "medvae" / "utils",
        "medvae.utils.vae": MEDVAE_ROOT / "medvae" / "utils" / "vae",
    }
    for name, pkg_path in stubs.items():
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(pkg_path)]
            module.__package__ = name
            sys.modules[name] = module


def _deep_update(base, update):
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(parts):
    print("=== model summary ===")
    for name, module in parts.items():
        if module is None:
            continue
        total, trainable = count_parameters(module)
        print(f"{name:18s} {total / 1e6:8.2f}M params ({trainable / 1e6:6.2f}M trainable)")
    print("=====================")


def print_loader_sizes(train_loader, val_loader=None):
    print(
        f"train loader : {len(train_loader.dataset)} samples / "
        f"{len(train_loader)} batches"
    )
    if val_loader is not None:
        print(
            f"val loader   : {len(val_loader.dataset)} samples / "
            f"{len(val_loader)} batches"
        )


def load_config(config_path, model_config=None):
    import yaml

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if model_config:
        with open(model_config, "r", encoding="utf-8") as handle:
            model_cfg = yaml.safe_load(handle) or {}
        cfg = _deep_update(cfg, model_cfg)
    return cfg

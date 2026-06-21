"""
Configuration partagée du pipeline biais inductif.
Modifier APPROACH pour changer la méthode de calcul du score de qualité c.

Notebooks à relancer après changement d'approche :
  - Approche A (score pondéré, poids = loadings PCA) ou B (PCA PC1) : relancer NB3 → NB4 → NB5.
  - Approche C (score appris MLP + ResNet-18) : relancer NB2 → NB3 → NB4 → NB5
    (NB2 doit être relancé/disponible car l'approche C nécessite les features ResNet-18).
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  HYPERPARAMÈTRE PRINCIPAL — Approche de calcul du score c  ║
# ╚══════════════════════════════════════════════════════════════╝

APPROACH = "C"

# ── Définition des approches ──────────────────────────────────

APPROACHES = {
    "A": {
        "score_col": "score_weighted",
        "name": "Score pondéré (poids PCA)",
        "use_resnet": False,
        "description": (
            "Combinaison linéaire de 6 métriques IQA normalisées Min-Max, "
            "pondérées par les loadings absolus du PC1 (|loading| / Σ|loading|, "
            "fittés sur TRAIN). Poids objectifs et reproductibles, plus de "
            "valeurs manuelles arbitraires."
        ),
    },
    "B": {
        "score_col": "score_pca",
        "name": "PCA (PC1)",
        "use_resnet": False,
        "description": (
            "Premier composant principal normalisé [0,1], signe corrigé "
            "par corrélation avec Tenengrad. Objectif, data-driven."
        ),
    },
    "C": {
        "score_col": "score_dl",
        "name": "Score appris (MLP + ResNet-18)",
        "use_resnet": True,
        "description": (
            "MLP (512→128→32→1) entraîné par MSE sur les features ResNet-18 "
            "figées d'images TRAIN dégradées synthétiquement (bruit/flou/JPEG), "
            "avec une cible de qualité = 1 - sévérité connue. Score appliqué "
            "ensuite aux features ResNet-18 réelles (NB2). Supervision = proxy "
            "synthétique (ARCADE n'a pas d'annotation qualité réelle)."
        ),
    },
}


def get_approach_config(approach=None):
    """Retourne la config de l'approche sélectionnée."""
    approach = approach or APPROACH
    if approach not in APPROACHES:
        raise ValueError(
            f"Approche '{approach}' inconnue. Choix : {list(APPROACHES.keys())}"
        )
    return APPROACHES[approach]


def get_score_column(approach=None):
    """Retourne le nom de la colonne de score pour l'approche."""
    return get_approach_config(approach)["score_col"]


def use_resnet(approach=None):
    """Retourne True si l'approche requiert les features ResNet-18."""
    return get_approach_config(approach)["use_resnet"]


def print_approach_summary(approach=None):
    """Affiche un résumé de l'approche sélectionnée."""
    approach = approach or APPROACH
    cfg = get_approach_config(approach)
    print(f"╔{'═' * 58}╗")
    print(f"║  Approche : {approach} — {cfg['name']:<43s} ║")
    print(f"╚{'═' * 58}╝")
    print(f"  Colonne de score : {cfg['score_col']}")
    print(f"  {cfg['description']}")
    print()

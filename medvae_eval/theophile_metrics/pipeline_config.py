"""
Configuration partagée du pipeline biais inductif.
Modifier APPROACH pour changer la méthode de calcul du score de qualité c.
Puis relancer NB3 → NB4 → NB5.
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║  HYPERPARAMÈTRE PRINCIPAL — Approche de calcul du score c  ║
# ╚══════════════════════════════════════════════════════════════╝

APPROACH = "A"

# ── Définition des approches ──────────────────────────────────

APPROACHES = {
    "A": {
        "score_col": "score_weighted",
        "name": "Score pondéré",
        "use_resnet": False,
        "description": (
            "Combinaison linéaire de 6 métriques IQA avec poids manuels "
            "(Tenengrad=0.30, Laplacian=0.25, RMS=0.15, Entropy=0.15, "
            "Homogénéité=0.10, Balance=0.05)"
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
        "score_col": "score_hybrid",
        "name": "Hybride (A+B+ResNet)",
        "use_resnet": True,
        "description": (
            "Moyenne de score_weighted + score_pca + score_resnet (PC1 ResNet-18). "
            "Nécessite les features ResNet-18 du NB2."
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

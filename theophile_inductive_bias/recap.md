# Récapitulatif – Biais Inductif & Qualité d'Image (ARCADE)

---

## 1. Adaptation du notebook au nouveau projet

**Action :** Le chemin `DATASET_PATH` pointait vers un ancien dossier (`projet_med_vae`). Il a été remplacé par un chemin relatif résolu dynamiquement depuis l'emplacement du notebook :

```python
DATASET_PATH = str((Path(os.path.abspath('')) / '..' / 'arcade_challenge_datasets').resolve())
```

**Justification :** Le dataset est maintenant à la racine du projet `projet_IM06/arcade_challenge_datasets/`. Le reste du code (scan récursif des sous-dossiers `images/`) est déjà compatible avec cette structure — aucune autre modification n'a été nécessaire.

---

## 2. Métriques de qualité calculées

Le notebook calcule **11 métriques No-Reference (NR-IQA)** sur chaque image du dataset ARCADE (~3000 images coronariennes en rayons X) :

| Catégorie | Métrique | Direction |
|-----------|----------|-----------|
| Netteté (classique) | Laplacian Variance | ↑ = plus net |
| Netteté (classique) | Tenengrad | ↑ = plus net |
| Netteté (classique) | Brenner | ↑ = plus net |
| Contraste | RMS Contrast | ↑ = plus de contraste |
| Contraste | Michelson | ↑ = plus de contraste |
| Exposition | Mean Brightness | milieu = bien exposé |
| Bruit (classique) | Immerkaer σ | ↓ = moins de bruit |
| Bruit (classique) | Wavelet σ | ↓ = moins de bruit |
| NR-IQA classique | BRISQUE | ↓ = meilleure qualité |
| NR-IQA classique | NIQE | ↓ = meilleure qualité |
| NR-IQA deep | CLIP-IQA | ↑ = meilleure qualité |

---

## 3. Score composite (`quality_score`)

**Action :** Un score composite unique `quality_score ∈ [0, 1]` est calculé comme la **moyenne non pondérée** de 7 métriques normalisées (min-max).

**Métriques incluses dans le composite :**

| Métrique | Normalisation |
|----------|---------------|
| `laplacian_var` | directe (↑ = mieux) |
| `tenengrad` | directe (↑ = mieux) |
| `rms_contrast` | directe (↑ = mieux) |
| `clip_iqa` | directe (↑ = mieux) |
| `immerkaer_noise` | inversée : `1 - minmax(x)` |
| `brisque` | inversée : `1 - minmax(x)` |
| `niqe` | inversée : `1 - minmax(x)` |

**Métriques exclues du composite et justifications :**

- **`brenner`** — Redondant avec `laplacian_var` et `tenengrad` (les trois mesurent la netteté, forte corrélation entre elles). L'inclure donnerait un poids disproportionné à la netteté (3x au lieu de 2x).

- **`michelson`** — Sensible aux outliers : un seul pixel très clair ou très sombre fausse le résultat. Moins robuste que `rms_contrast` qui utilise l'écart-type sur toute l'image.

- **`mean_brightness`** — Ce n'est pas une métrique de qualité à proprement parler. Une image sombre n'est pas forcément de mauvaise qualité ; c'est une caractéristique d'exposition. Il n'y a pas de direction claire "bon/mauvais", donc pas de place dans un score composite linéaire.

- **`wavelet_noise`** — Redondant avec `immerkaer_noise` (deux estimateurs du même phénomène : le niveau de bruit). Inclure les deux surpondérerait le bruit dans le composite.

**Pourquoi calculer ces métriques exclues malgré tout ?**
- Pour l'**analyse exploratoire** (la heatmap de corrélation permet de vérifier les hypothèses de redondance).
- Pour pouvoir **ajuster** le composite après coup si nécessaire.
- Pour des **analyses spécifiques** (ex. filtrer les images trop sombres via `mean_brightness`).

---

## 4. Visualisation : comparaison Low vs High quality

**Action :** Affichage des 5 images les plus floues et des 5 images les plus nettes, triées par `laplacian_var`.

**Justification du choix de `laplacian_var` pour cette visualisation :** C'est la métrique de netteté la plus intuitive visuellement — on voit immédiatement la différence entre une image floue et une image nette sur des vignettes. D'autres métriques (bruit, contraste, BRISQUE…) seraient moins évidentes à juger à l'œil.

---

## 5. Sorties produites par le notebook

| Fichier | Contenu |
|---------|---------|
| `quality_metrics.csv` | CSV avec toutes les métriques pour chaque image (1 ligne = 1 image) |
| `quality_distributions.png` | Histogrammes de distribution de chaque métrique |
| `quality_correlation.png` | Heatmap de corrélation entre métriques |
| `quality_comparison.png` | 10 vignettes : 5 low quality vs 5 high quality (Laplacian Var) |
| `quality_composite.png` | Histogramme du score composite `quality_score` |

Tous les fichiers sont écrits dans le répertoire `theophile_biais_inductif/`.

---

# Récapitulatif – Pipeline Biais Inductif & Qualité d'Image (ARCADE)

> Ce dossier `theophile_metrics/` contient **7 notebooks** formant un pipeline complet :
> extraction de métriques de qualité No-Reference sur les images coronariennes ARCADE,
> génération d'un biais inductif (3 approches), modification de l'architecture MedVAE,
> évaluation de robustesse, ablation, et comparaison avec un PSNR masqué par
> l'anatomie vasculaire.

---

## Configuration globale — `pipeline_config.py`

Un **hyperparamètre unique** `APPROACH` contrôle la méthode de calcul du score de qualité `c` à travers tout le pipeline. Modifier ce paramètre et relancer NB3 → NB4 → NB5 pour comparer les approches.

| Approche | Colonne | Description |
|----------|---------|-------------|
| **A** | `score_weighted` | Combinaison linéaire de 6 métriques IQA normalisées Min-Max, pondérées par les **loadings absolus du PC1** (`\|loading\| / Σ\|loading\|`, fittés sur TRAIN) — poids objectifs, pas de choix manuel arbitraire |
| **B** | `score_pca` | PCA (PC1 normalisé), signe corrigé par corrélation Tenengrad |
| **C** | `score_dl` | **Score appris** : MLP (512→128→32→1) entraîné sur features ResNet-18 figées d'images TRAIN dégradées synthétiquement (bruit/flou/JPEG), cible = `1 - sévérité connue` |

Les sorties de NB4, NB5 et NB6 sont organisées par approche (`approach_A/`, `approach_B/`, `approach_C/`) pour permettre la comparaison. NB7 compare les 3 approches simultanément.

---

## Vue d'ensemble du pipeline

```
pipeline_config.py → APPROACH = "A" | "B" | "C"
    ↓
NB1 → Exploration & MSCN
NB2 → Cartes de qualité par patch (8 métriques × G×G) [+ features ResNet-18 si approche C]
NB3 → Score unique c ∈ [0,1] → labels_quality.csv (score_weighted + score_pca + score_dl)
NB4 → QualityAwareAutoencoderKL (FiLM) → approach_{X}/checkpoints/quality_aware_vae.pt
NB5 → Évaluation de robustesse (baseline vs cVAE) → approach_{X}/{figures,tables}
NB6 → Ablation par c constant (à relancer pour A, B et C) → approach_{X}/{figures,tables,checkpoints}
NB7 → Comparaison PSNR classique vs Masked PSNR (3 approches + ablation)
quality_metrics.ipynb → Pipeline autonome simplifié (11 métriques NR-IQA + score composite)
```

**Ordre de ré-exécution après changement de méthodologie (A et/ou C) :** pour repartir d'un
état parfaitement cohérent, exécuter dans l'ordre `NB3(A)→NB4(A)→NB5(A)`,
`NB3(C)→NB4(C)→NB5(C)`, puis `NB3(B)→NB4(B)→NB5(B)→NB6(B)`, et terminer par une dernière
exécution de **NB3 avec `APPROACH="C"`** pour régénérer `labels_quality.csv` avec les
3 colonnes de score simultanément (NB3 n'exporte `score_dl` que lorsque
`APPROACH="C"`), avant de lancer NB7.

---

## Notebook 1 — Exploration et Prétraitement (`01_Data_Exploration_and_Preprocessing.ipynb`)

**Objectif :** Comprendre la variabilité de qualité du dataset ARCADE et préparer la normalisation MSCN.

### Étapes

1. **Chargement d'un échantillon** — 20 images aléatoires depuis `seg_train/images/`, converties en grayscale float32 [0, 1].
2. **Panorama visuel** — Mosaïque de l'échantillon pour observer la variabilité de qualité à l'œil.
3. **Calcul de 3 métriques basiques** — Laplacian variance (netteté), RMS contrast, Entropie de Shannon. Tri par netteté croissante pour identifier les images les plus floues.
4. **Comparaison good vs bad** — Figure côte à côte de l'image la plus nette et la plus floue (figure clé du rapport).
5. **Distribution des métriques** — Histogrammes des 3 métriques sur l'échantillon.
6. **Normalisation MSCN** (Mean Subtracted Contrast Normalized) — `Î(i,j) = (I(i,j) - μ(i,j)) / (σ(i,j) + C)` — supprime les biais d'illumination non-uniforme et les biais globaux de contraste entre patients.
7. **Validation de MSCN** — Comparaison visuelle avant/après, histogrammes de distribution, et vérification que les métriques différencient encore les images post-MSCN.

### Sorties (dans `outputs_theophile/01_Data_Exploration/figures/`)

| Fichier | Contenu |
|---------|---------|
| `01_panorama_sample.png` | Mosaïque de l'échantillon |
| `01_good_vs_bad.png` | Comparaison image nette vs floue |
| `01_metrics_distribution.png` | Histogrammes des 3 métriques |
| `01_mscn_comparison.png` | Effet de la normalisation MSCN |
| `01_histograms_mscn.png` | Distributions pixel avant/après MSCN |
| `01_metrics_raw_vs_mscn.png` | Métriques brutes vs post-MSCN |

---

## Notebook 2 — Extraction des Métriques IQA par Patch (`02_IQA_Metrics_Extraction.ipynb`)

**Objectif :** Calculer des métriques de qualité No-Reference sur **l'ensemble du dataset ARCADE**, en évaluant chaque image **localement** via une grille de patchs 4×4.

### Métriques implémentées (8 par patch)

| Métrique | Type | Justification |
|---|---|---|
| Laplacian variance | Netteté | Standard, robuste sur les radios |
| RMS Contrast | Contraste local | Sensible aux artefacts de mouvement |
| Entropie de Shannon | Désordre / artefacts | Validé sur DSA (PatchDSA, 2024) |
| Tenengrad | Gradient global (Sobel L2) | Robuste aux bords fins des artères |
| Tenengrad H / V / D1 / D2 | Gradient directionnel (4 noyaux) | Détecte le flou de mouvement cardiaque (direction prédominante) |

### Étapes

1. **Chargement du dataset complet** — Toutes les images de `seg_train` + `seg_val`.
2. **Découpe en patchs** — Grille 4×4, chaque image → 16 patchs de taille `(H/4, W/4)`.
3. **Normalisation MSCN** — Appliquée avant le calcul des métriques.
4. **Carte de qualité 2D** — Tableau `(4, 4, 8)` par image, contenant les 8 métriques pour chaque patch.
5. **Analyse directionnelle** — Visualisation du Tenengrad H/V/D1/D2 pour détecter les flous de mouvement (le battement cardiaque dégrade surtout la direction verticale).
6. **Extraction de features ResNet-18** — Vecteur 512D par image via un ResNet-18 pré-entraîné figé (couche `AdaptiveAvgPool2d`), utilisé comme entrée du score appris (Méthode C, NB3) — uniquement extrait si `APPROACH="C"`.
7. **Traitement du dataset complet** — Parallélisé (4 threads). Les patchs « noirs » (bords de la radio, intensité moyenne < 2%) sont exclus. Agrégation des patchs valides en mean/std/max/min par métrique.
8. **Analyse statistique** — Distributions des 8 métriques sur tout le dataset, matrice de corrélation pour identifier les redondances.
9. **Export** — CSV + cartes de qualité `.npy`.

### Sorties (dans `outputs_theophile/02_IQA_Metrics_Extraction/`)

| Fichier | Contenu |
|---------|---------|
| `metrics/patch_metrics_full.csv` | 8 métriques × (mean/std/max/min) pour chaque image |
| `quality_maps/*.npy` | Carte de qualité `(4, 4, 8)` par image |
| `resnet_features/resnet18_features.npy` | Matrice (N, 512) de features ResNet-18 |
| `resnet_features/resnet18_ids.csv` | Correspondance image_id ↔ indice dans la matrice |
| `figures/02_qmap_demo.png` | Heatmaps de qualité par métrique (démo) |
| `figures/02_directional_tenengrad.png` | Analyse directionnelle H/V/D1/D2 |
| `figures/02_full_dataset_distributions.png` | Histogrammes des métriques (dataset complet) |
| `figures/02_correlation_matrix.png` | Matrice de corrélation entre métriques |
| `figures/02_extreme_images.png` | Images extrêmes (worst vs best Tenengrad) |

---

## Notebook 3 — Génération du Biais Inductif (`03_Inductive_Bias_Generation.ipynb`)

**Objectif :** Fusionner les 8 métriques IQA du Notebook 2 en un **score unique c ∈ [0, 1]** interprétable par le cVAE (1 = bonne qualité, 0 = image dégradée).

### Feature engineering

Deux indicateurs spatiaux dérivés des données du Notebook 2 :

| Feature | Définition | Interprétation |
|---|---|---|
| `spatial_homogeneity` | `1 - σ_tenengrad / (μ_tenengrad + ε)` | 1 = qualité uniforme, 0 = zones floues locales |
| `directional_balance` | `min(H, V, D1, D2) / (max(H, V, D1, D2) + ε)` | 1 = isotrope, 0 = flou directionnel (mouvement) |

### Séparation train/val (anti-leakage)

Une colonne `split` est ajoutée au chargement des métriques, déduite du chemin de l'image (`seg_train` → `"train"`, `seg_val` → `"val"`). **Toutes les opérations statistiques (clipping, normalisation, PCA, entraînement du MLP, seuils) sont fitted uniquement sur les images TRAIN**, puis appliquées (transform) sur l'ensemble du dataset.

### Trois méthodes de scoring comparées

| Méthode | Principe | Avantage |
|---|---|---|
| **B — PCA (PC1)** *(calculée en premier)* | Premier composant principal normalisé à [0, 1] (**PCA fitted sur TRAIN**), signe corrigé par corrélation avec Tenengrad (sur TRAIN) | Objectif, data-driven, pas de choix arbitraire de poids |
| **A — Score pondéré** | Combinaison linéaire de 6 features normalisées Min-Max (**fitted sur TRAIN**), avec poids = **loadings absolus du PC1 de la Méthode B**, normalisés à somme 1 | Interprétable physiquement, poids objectifs (plus de valeurs manuelles) |
| **C — Score appris** | MLP (512→128→32→1, sortie sigmoïde) entraîné par MSE sur les features ResNet-18 figées d'images **TRAIN dégradées synthétiquement** (bruit gaussien, flou de mouvement, JPEG), cible = `1 - sévérité connue`. Appliqué ensuite aux features ResNet-18 réelles (NB2), puis renormalisé Min-Max (bornes TRAIN) | Exploite une représentation apprise (non-linéaire), pas seulement une combinaison linéaire de métriques classiques |

> **Honnêteté scientifique (Méthode C) :** ARCADE ne fournit aucune annotation de qualité
> réelle. La cible d'entraînement du MLP est un **proxy synthétique** (sévérité de
> dégradation connue), pas une vérité terrain humaine. Le split train/val interne au MLP
> est fait par image (jamais par échantillon), toujours à l'intérieur des images TRAIN.

### Sélection du score final

Le score final (`quality_score`) est déterminé par l'hyperparamètre `APPROACH` dans `pipeline_config.py` :
- `APPROACH = "A"` → `score_weighted`
- `APPROACH = "B"` → `score_pca`
- `APPROACH = "C"` → `score_dl`

`score_weighted` et `score_pca` sont **toujours** calculés et exportés, quelle que soit
l'approche active. `score_dl` n'est calculé (et exporté) que lorsque `APPROACH="C"` (les
features ResNet-18 du NB2 doivent être disponibles).

### Labels catégoriels

Le score continu est discrétisé en 3 classes par tertiles **calculés sur TRAIN uniquement** :

| Label | Tertile | Signification |
|---|---|---|
| `bad` | [0, 33%[ | Artefacts sévères |
| `medium` | [33%, 67%[ | Qualité acceptable |
| `good` | [67%, 100%] | Cliniquement exploitable |

### Validation

- Galerie triée par score (vérification visuelle de la cohérence)
- Corrélation du score final avec chaque métrique brute
- Galerie de 3 exemples par classe (bad / medium / good)
- Méthode C : courbe d'entraînement du MLP, scatter prédiction vs sévérité synthétique connue, comparaison `score_dl` vs `score_weighted`/`score_pca`

### Sorties (dans `outputs_theophile/03_Inductive_Bias_Generation/`)

| Fichier | Contenu |
|---------|---------|
| `labels_quality.csv` | Fichier **partagé**, écrasé à chaque run de NB3 : image_id, path, **split**, quality_score, quality_label, score_weighted, score_pca [, score_dl]. `quality_score`/`quality_label` reflètent la **dernière** approche exécutée (les colonnes `score_*` sont elles préservées entre runs). |
| `common/03_engineered_features.png` | Distribution des features spatial_homogeneity et directional_balance (indépendant de l'approche) |
| `common/03_score_comparison.png` | Corrélation et différence entre Méthode A et Méthode B (indépendant de l'approche : compare toujours score_weighted vs score_pca) |
| `approach_{A,B,C}/labels_quality.csv` | **Copie figée** par approche : mêmes colonnes que ci-dessus, mais `quality_score`/`quality_label` correspondent bien à CETTE approche, même après que d'autres aient tourné ensuite. |
| `approach_{A,B,C}/03_label_distribution.png` | Distribution du score final (de cette approche) avec seuils + camembert des labels |
| `approach_{A,B,C}/03_gallery_sorted_by_score.png` | Galerie triée par quality_score (de cette approche) |
| `approach_{A,B,C}/03_examples_per_class.png` | 3 exemples par classe (bad / medium / good), labels de cette approche |
| `approach_A_weighted_pca/03_score_weighted.png` | Distribution du score pondéré + poids PCA par feature |
| `approach_B_pca1/03_pca_biplot.png` | Loadings PC1 + biplot PC1 vs PC2 |
| `approach_C_learned/quality_mlp_resnet.pt` | MLP entraîné (poids + scaler + métadonnées) |
| `approach_C_learned/03_dl_training_curve.png` | Courbe d'entraînement + validation prédite vs sévérité connue |
| `approach_C_learned/03_dl_score_comparison.png` | `score_dl` vs `score_weighted`/`score_pca` |

> ⚠️ `03_label_distribution.png`, `03_gallery_sorted_by_score.png` et `03_examples_per_class.png` dépendent de `quality_score`/`quality_label`, donc de l'approche active — ils ont été déplacés de `common/` vers `approach_{A,B,C}/` (corrigé le 2026-06-17) pour ne plus être écrasés entre deux runs d'approches différentes.

---

## Notebook 4 — Architecture cVAE Quality-Aware (`04_MedVAE_Architecture_Mod.ipynb`)

**Objectif :** Brancher le score de qualité c ∈ [0,1] sur l'architecture MedVAE de Stanford pour créer un **Variational Autoencoder Conditionnel (cVAE) quality-aware** via FiLM conditioning.

### Principe du conditionnement FiLM (Perez et al., AAAI 2018)

Un MLP partagé transforme le scalaire c en un embedding de dimension 128. Puis 19 têtes linéaires (une par `ResnetBlock` de l'encodeur + décodeur) produisent des paramètres affines (γᵢ, βᵢ). Après chaque `ResnetBlock` :

```
h ← (1 + γᵢ) · h + βᵢ
```

| Aspect | Concaténation de canal | FiLM (retenu) |
|---|---|---|
| Point d'injection | Entrée seule | Chaque couche (19 blocs) |
| Modification du modèle | Patch `conv_in` (+1 canal) | Aucune (forward hooks) |
| Type de modulation | Canal spatial constant | Scale + shift appris par couche |
| Initialisation | Zéro | Zéro (identité au départ → stabilité) |
| Paramètres ajoutés | ~0.01% | ~2.7% |

### Étapes

1. **Dataset** — `QualityAwareArcadeDataset` retourne `(image, quality_score)` au lieu de `(image, mask)`. Images redimensionnées à 64×64, recadrées dans [-1, 1]. **Anti-leakage : le DataLoader d'entraînement utilise uniquement les images TRAIN** (colonne `split` du CSV), avec un DataLoader de validation séparé.
2. **Architecture** — `QualityAwareAutoencoderKL` wrappant `AutoencoderKL2D` avec un `FiLMConditioner`. Zero-init sur toutes les heads → pas d'effet au démarrage.
3. **Vérification** — Forward pass, flux de gradient (∂loss/∂c ≠ 0), comptage de paramètres.
4. **Schéma architectural** — Figure annotée du pipeline x → Encoder → z → Decoder → x̂ avec injection FiLM.
5. **Visualisation pré-entraînement** — Reconstruction à différents c (identiques car zero-init).
6. **Phase 1 — Warm-up FiLM** (2 500 steps, ~10 epochs) — Seuls les paramètres FiLM sont entraînés (base VAE gelé). Loss ELBO = MSE + KL_weight × KL. Validation toutes les 250 steps avec early stopping (patience=10).
7. **Visualisation post-warm-up** — Reconstruction d'une image bad et good à c=0, 0.5, 1. Le score commence à moduler la sortie.
8. **Phase 2 — Fine-tuning global** (5 000 steps, ~20 epochs) — Tous les poids sont dégelés (base VAE + FiLM). LR réduit à 1e-6 avec scheduler cosine → 1e-7. Validation et early stopping identiques.
9. **Sauvegarde du checkpoint** — `quality_aware_vae.pt` contenant le meilleur modèle (sélectionné sur val loss), ddconfig, embed_dim, historiques des deux phases.

### Sorties (dans `outputs_theophile/04_MedVAE_Architecture_Mod/approach_{APPROACH}/`)

| Fichier | Contenu |
|---------|---------|
| `checkpoints/quality_aware_vae.pt` | Checkpoint du cVAE + baseline ablation (modèle + config + historique) |
| `figures/04_architecture_cvae.png` | Schéma de l'architecture FiLM |
| `figures/04_conditioning_before_training.png` | Reconstructions à c variés (zero-init) |
| `figures/04_warmup_curves.png` | Courbes de loss Phase 1 (train + val) |
| `figures/04_conditioning_after_training.png` | Effet du conditionnement post-warm-up |
| `figures/04_finetune_curves.png` | Courbes de loss Phase 2 (train + val) |
| `figures/04_baseline_curves.png` | Courbes de loss de l'ablation baseline (sans FiLM) |

---

## Notebook 5 — Évaluation de la Robustesse (`05_Evaluation_of_Robustness.ipynb`)

**Objectif :** Prouver que le biais inductif améliore la reconstruction — le **cVAE Quality-Aware** doit reconstruire les images dégradées mieux que le MedVAE baseline.

### Protocole expérimental

| Expérience | Description |
|---|---|
| **A — Images réelles** | Comparer baseline vs cVAE sur les images `bad/medium/good` d'ARCADE (20 images/classe) |
| **B — Dégradations synthétiques** | Dégrader artificiellement les meilleures images et comparer la robustesse |

### Métriques de reconstruction

| Métrique | Description | Direction |
|---|---|---|
| MSE | Erreur quadratique moyenne pixel-à-pixel | ↓ meilleur |
| SSIM | Structural Similarity Index | ↑ meilleur |
| PSNR | Peak Signal-to-Noise Ratio | ↑ meilleur |
| HaarPSI | Haar Perceptual Similarity Index (Reisenhofer et al., 2018) | ↑ meilleur |

> **Note :** HaarPSI est privilégié car plus fiable que SSIM/PSNR sur les images médicales (Breger et al., 2024). Le Notebook 7 complète cette analyse avec un **PSNR masqué par le masque vasculaire** (voir plus bas).

### Grille de dégradations synthétiques (7 types)

Propre, Bruit gaussien (σ=0.05, 0.10, 0.20), Flou de mouvement horizontal (k=5), Flou de mouvement vertical (k=5, k=11).

### Étapes

1. **Chargement des deux modèles** — cVAE (depuis checkpoint NB4) + Baseline (mêmes poids de base, sans FiLM).
2. **Construction des sets de test** — Sélection stratifiée de 20 images par classe (bad/medium/good).
3. **Score de qualité inline** — Fonction `inline_quality_score()` recalculant un score c directement depuis un tenseur (pour les images synthétiques sans label).
4. **Évaluation sur images réelles** — Reconstruction avec le vrai `quality_score` pour le cVAE.
5. **Évaluation sur dégradations synthétiques** — Métriques calculées vs l'image **propre originale** (ground truth).
6. **Comparaison statistique** — Test de Wilcoxon (non-paramétrique) pour chaque métrique sur les images `bad`.
7. **Visualisations** — Box plots par classe, bar charts par type de dégradation, galeries qualitatives (Original | Baseline | cVAE | Diff).

### Lecture des résultats

- Le cVAE est entraîné avec un warm-up FiLM (2 500 steps) + fine-tuning global (5 000 steps) = **7 500 steps** au total, avec validation et early stopping.
- Ce notebook est exécuté indépendamment pour chacune des 3 approches (`approach_A/`, `approach_B/`, `approach_C/`).

### Sorties (dans `outputs_theophile/05_Evaluation_of_Robustness/approach_{APPROACH}/`)

| Fichier | Contenu |
|---------|---------|
| `tables/05_evaluation_results.csv` | Métriques par image (baseline + cVAE) sur images réelles |
| `tables/05_synthetic_results.csv` | Métriques par image × dégradation sur images synthétiques |
| `figures/05_degradation_grid.png` | Grille des 7 types de dégradations |
| `figures/05_boxplots.png` | Box plots MSE/SSIM/HaarPSI par classe de qualité |
| `figures/05_synthetic_metrics.png` | Bar charts baseline vs cVAE par dégradation |
| `figures/05_gallery_bad.png` | Galerie qualitative sur images bad |
| `figures/05_gallery_synthetic.png` | Galerie sur images synthétiquement dégradées |

---

## Notebook 6 — Ablation par c Constant (`06_Ablation_c_Constant.ipynb`)

**Objectif :** Vérifier que le signal `c` contient une information utile pour le cVAE, en
comparant le `cvae_{APPROACH}` (entraîné avec le vrai score de l'approche active) à un
second cVAE entraîné avec `c=0.5` constant pour toutes les images — même architecture,
mêmes données, même budget compute. Si le cVAE(c réel) est significativement meilleur, le
signal `c` est exploité ; sinon, le conditionnement FiLM n'apporte rien au-delà d'une
simple capacité supplémentaire du modèle. **Généralisé aux 3 approches** : le notebook lit
`APPROACH` depuis `pipeline_config.py` comme NB4/NB5, donc à relancer une fois par approche
(A, B, C) pour avoir l'ablation complète des 3 — ce n'est plus limité à B.

### Sorties (dans `outputs_theophile/06_Ablation_c_Constant/approach_{APPROACH}/`)

| Fichier | Contenu |
|---------|---------|
| `checkpoints/quality_aware_vae_c_constant.pt` | Checkpoint du cVAE entraîné avec c=0.5 constant |
| `tables/06_ablation_results.csv` | Métriques par image, c réel vs c constant |
| `tables/06_synthetic_results.csv` | Métriques sur dégradations synthétiques |
| `figures/06_training_curves.png` | Courbes d'entraînement du cVAE(c=0.5) |
| `figures/06_boxplots.png` | Distribution des métriques, c réel vs c constant |
| `figures/06_delta_per_image.png` | Δmétrique par image en fonction du score qualité |
| `figures/06_gallery.png` | Galerie qualitative bad/good |
| `figures/06_synthetic_barplot.png` | Comparaison sur dégradations synthétiques |

---

## Notebook 7 — Comparaison Masked PSNR (`07_Masked_PSNR_Comparison.ipynb`)

**Objectif :** Reprendre les évaluations des Notebooks 5/6 (baseline vs cVAE, 3 approches +
ablation) en complétant le **PSNR classique** par un **Masked PSNR**
(`medvae_eval/scripts/masked_psnr.py`) qui ne calcule le MSE que sur les pixels du masque
vasculaire annoté (polygones COCO ARCADE), au lieu de l'image entière.

### Point méthodologique important

Chaque modèle est conditionné avec **le score natif de son approche** (`score_weighted`
pour les modèles `_A`, `score_pca` pour `_B`, `score_dl` pour `_C`), lues directement
depuis `labels_quality.csv` (qui contient les 3 colonnes simultanément). Les classes
bad/medium/good sont **recalculées indépendamment pour chaque approche** (tertiles sur
TRAIN), au lieu de réutiliser la colonne `quality_label` générique du CSV — qui ne reflète
que la dernière approche avec laquelle NB3 a été exécuté. Ceci évite toute contamination
croisée entre approches lors de l'évaluation.

### Sorties (dans `outputs_theophile/07_Masked_PSNR_Comparison/`)

| Fichier | Contenu |
|---------|---------|
| `tables/07_full_results.csv` | Résultats complets (3 approches + ablation) |
| `tables/07_psnr_comparison_table.csv` | PSNR classique vs masked, par classe et modèle |
| `tables/07_verdict_flip_table.csv` | Le gagnant change-t-il selon la métrique ? |
| `tables/07_synthetic_results.csv` | Résultats sur dégradations synthétiques |
| `figures/07_mask_explanation.png` | Visualisation du masque vasculaire (signal vs bruit) |
| `figures/07_psnr_vs_masked_bars.png` | Comparaison barres classique/masked par approche |
| `figures/07_psnr_correlation_scatter.png` | Corrélation par image entre les deux métriques |
| `figures/07_boxplots_*.png` | Distributions par comparaison |
| `figures/07_gallery_bad_approach_{A,B,C}.png` | Galeries qualitatives par approche, overlay masque |
| `figures/07_gallery_good_ablation.png` | Galerie ablation c réel vs c constant |
| `figures/07_gallery_synthetic_masked.png` | Galerie sur dégradations synthétiques |
| `figures/07_synthetic_psnr_vs_masked.png` | PSNR classique vs masked selon sévérité de dégradation |

---

## quality_metrics.ipynb — Pipeline autonome simplifié

**Objectif :** Pipeline autonome et léger d'évaluation de qualité NR-IQA, indépendant de la chaîne NB1→NB7. Produit directement un score composite par image.

### Métriques calculées (11 métriques No-Reference)

| Catégorie | Métrique | Direction |
|-----------|----------|-----------|
| Netteté (classique) | Laplacian Variance | ↑ = plus net |
| Netteté (classique) | Tenengrad (Sobel) | ↑ = plus net |
| Netteté (classique) | Brenner | ↑ = plus net |
| Contraste | RMS Contrast | ↑ = plus de contraste |
| Contraste | Michelson | ↑ = plus de contraste |
| Exposition | Mean Brightness | milieu = bien exposé |
| Bruit (classique) | Immerkaer σ | ↓ = moins de bruit |
| Bruit (classique) | Wavelet σ (Haar) | ↓ = moins de bruit |
| NR-IQA classique | BRISQUE (piq) | ↓ = meilleure qualité |
| NR-IQA classique | NIQE (piq) | ↓ = meilleure qualité |
| NR-IQA deep | CLIP-IQA (piq) | ↑ = meilleure qualité |

### Score composite (`quality_score`)

Moyenne non pondérée de 7 métriques normalisées (min-max) :

| Métrique | Normalisation |
|----------|---------------|
| `laplacian_var` | directe (↑ = mieux) |
| `tenengrad` | directe (↑ = mieux) |
| `rms_contrast` | directe (↑ = mieux) |
| `clip_iqa` | directe (↑ = mieux) |
| `immerkaer_noise` | inversée : `1 - minmax(x)` |
| `brisque` | inversée : `1 - minmax(x)` |
| `niqe` | inversée : `1 - minmax(x)` |

**Métriques exclues du composite :**

- **`brenner`** — Redondant avec `laplacian_var` et `tenengrad` (surpondérerait la netteté).
- **`michelson`** — Sensible aux outliers (un seul pixel extrême fausse le résultat).
- **`mean_brightness`** — Pas une métrique de qualité (pas de direction claire bon/mauvais).
- **`wavelet_noise`** — Redondant avec `immerkaer_noise` (surpondérerait le bruit).

### Visualisations

- Histogrammes de distribution de chaque métrique
- Heatmap de corrélation entre métriques
- Comparaison visuelle des 5 images de plus basse vs plus haute qualité (Laplacian Var)
- Histogramme du score composite

### Sorties

| Fichier | Contenu |
|---------|---------|
| `outputs_theophile/quality_metrics_standalone/quality_metrics_scores.csv` | `image, quality_score` pour chaque image |
| `outputs_theophile/quality_metrics_standalone/quality_distributions.png` | Histogrammes des 11 métriques |
| `outputs_theophile/quality_metrics_standalone/quality_correlation.png` | Matrice de corrélation |
| `outputs_theophile/quality_metrics_standalone/quality_comparison.png` | 5 low quality vs 5 high quality |
| `outputs_theophile/quality_metrics_standalone/quality_composite.png` | Distribution du score composite |

---

## Fichier utilitaire — `medvae_standalone.py`

**Objectif :** Ré-implémentation locale et autonome de l'architecture MedVAE de Stanford (MIMI). Élimine la dépendance au dossier `jepa-adaptation/` et rend les notebooks portables.

### Composants

| Classe / Fonction | Rôle |
|---|---|
| `DiagonalGaussianDistribution` | Distribution postérieure du VAE (μ, log-σ²). Méthodes : `sample()` (reparamétérisation), `kl()` (divergence KL vs N(0,I)), `mode()` (déterministe). Log-variance clampée à [-30, 20]. |
| `nonlinearity(x)` | Activation Swish/SiLU |
| `Normalize(in_channels)` | GroupNorm (32 groupes) |
| `Upsample` / `Downsample` | Redimensionnement spatial 2× (nearest + conv optionnel) |
| `ResnetBlock` | Bloc résiduel avec GroupNorm + Swish + Conv |
| `AttnBlock` | Self-attention multi-tête |
| `Encoder` | Chemin descendant : Conv → ResBlocks → latent |
| `Decoder` | Chemin montant : latent → ResBlocks → Conv → reconstruction |
| `AutoencoderKL` | VAE complet combinant Encoder + Decoder + quant_conv |

### Utilisation dans les notebooks

```python
from medvae_standalone import AutoencoderKL, DiagonalGaussianDistribution
base_ae = AutoencoderKL(ddconfig=DDCONFIG, embed_dim=EMBED_DIM, ckpt_path=PHASE1_CKPT)
```

Ce fichier est importé par **NB4**, **NB5**, **NB6** et **NB7** pour charger les poids
pré-entraînés Stanford (`vae_4x_4c_2D.ckpt`) et construire le cVAE quality-aware.

---

## Fichier utilitaire — `medvae_eval/scripts/masked_psnr.py`

**Objectif :** Définit `MaskedPSNR`, un module PyTorch qui calcule le PSNR uniquement sur
les pixels appartenant au masque vasculaire (polygones de segmentation COCO ARCADE),
au lieu de l'image entière. Utilisé par **NB7**.

---

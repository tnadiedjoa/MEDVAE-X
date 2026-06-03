# Récapitulatif – Pipeline Biais Inductif & Qualité d'Image (ARCADE)

> Ce dossier `theophile_metrics/` contient **6 notebooks** formant un pipeline complet :
> extraction de métriques de qualité No-Reference sur les images coronariennes ARCADE,
> génération d'un biais inductif, modification de l'architecture MedVAE, et évaluation de robustesse.

---

## Vue d'ensemble du pipeline

```
NB1 → Exploration & MSCN
NB2 → Cartes de qualité par patch (8 métriques × G×G)
NB3 → Score unique c ∈ [0,1] → labels_quality.csv
NB4 → QualityAwareAutoencoderKL (FiLM) → quality_aware_vae.pt
NB5 → Évaluation de robustesse (baseline vs cVAE)
quality_metrics.ipynb → Pipeline autonome simplifié (11 métriques NR-IQA + score composite)
```

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

### Sorties

| Fichier | Contenu |
|---------|---------|
| `outputs/01_panorama_sample.png` | Mosaïque de l'échantillon |
| `outputs/01_good_vs_bad.png` | Comparaison image nette vs floue |
| `outputs/01_metrics_distribution.png` | Histogrammes des 3 métriques |
| `outputs/01_mscn_comparison.png` | Effet de la normalisation MSCN |
| `outputs/01_histograms_mscn.png` | Distributions pixel avant/après MSCN |
| `outputs/01_metrics_raw_vs_mscn.png` | Métriques brutes vs post-MSCN |

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
6. **Extraction de features ResNet-18** (optionnel) — Vecteur 512D par image via un ResNet-18 pré-entraîné figé (couche `AdaptiveAvgPool2d`), utilisable comme biais inductif enrichi.
7. **Traitement du dataset complet** — Parallélisé (4 threads). Les patchs « noirs » (bords de la radio, intensité moyenne < 2%) sont exclus. Agrégation des patchs valides en mean/std/max/min par métrique.
8. **Analyse statistique** — Distributions des 8 métriques sur tout le dataset, matrice de corrélation pour identifier les redondances.
9. **Export** — CSV + cartes de qualité `.npy`.

### Sorties

| Fichier | Contenu |
|---------|---------|
| `outputs/patch_metrics_full.csv` | 8 métriques × (mean/std/max/min) pour chaque image |
| `outputs/quality_maps/*.npy` | Carte de qualité `(4, 4, 8)` par image |
| `outputs/resnet18_features.npy` | Matrice (N, 512) de features ResNet-18 |
| `outputs/resnet18_ids.csv` | Correspondance image_id ↔ indice dans la matrice |
| `outputs/02_qmap_demo.png` | Heatmaps de qualité par métrique (démo) |
| `outputs/02_directional_tenengrad.png` | Analyse directionnelle H/V/D1/D2 |
| `outputs/02_full_dataset_distributions.png` | Histogrammes des métriques (dataset complet) |
| `outputs/02_correlation_matrix.png` | Matrice de corrélation entre métriques |
| `outputs/02_extreme_images.png` | Images extrêmes (worst vs best Tenengrad) |

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

Une colonne `split` est ajoutée au chargement des métriques, déduite du chemin de l'image (`seg_train` → `"train"`, `seg_val` → `"val"`). **Toutes les opérations statistiques (clipping, normalisation, PCA, seuils) sont fitted uniquement sur les images TRAIN**, puis appliquées (transform) sur l'ensemble du dataset.

### Deux méthodes de scoring comparées

| Méthode | Principe | Avantage |
|---|---|---|
| **A — Score pondéré** | Combinaison linéaire de 6 features normalisées (min-max **fitted sur TRAIN**) avec poids manuels (tenengrad=0.30, laplacian=0.25, rms_contrast=0.15, entropy_inv=0.15, spatial_homogeneity=0.10, directional_balance=0.05) | Interprétable physiquement |
| **B — PCA (PC1)** | Premier composant principal normalisé à [0, 1] (**PCA fitted sur TRAIN**), signe corrigé par corrélation avec Tenengrad (sur TRAIN) | Objectif, data-driven, pas de choix arbitraire de poids |
| **C — Hybride** (optionnel) | Moyenne de A + B + PC1-ResNet-18 (**PCA ResNet fitted sur TRAIN**) | Plus riche si features ResNet disponibles |

### Sélection automatique du score final

- Si `score_hybrid` disponible → utiliser `score_hybrid`
- Si corrélation A↔B > 0.85 → utiliser `score_pca` (objectif)
- Sinon → utiliser `score_weighted` (plus robuste si peu d'images)

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
- Comparaison poids manuels vs loadings PCA (cohérence vérifiée)

### Sorties

| Fichier | Contenu |
|---------|---------|
| `outputs/labels_quality.csv` | image_id, path, **split**, quality_score, quality_label, score_weighted, score_pca [, score_resnet, score_hybrid] |
| `outputs/03_engineered_features.png` | Distribution des features spatial_homogeneity et directional_balance |
| `outputs/03_score_weighted.png` | Distribution du score pondéré + contribution de chaque feature |
| `outputs/03_pca_biplot.png` | Loadings PC1 + biplot PC1 vs PC2 |
| `outputs/03_score_comparison.png` | Corrélation et différence entre Méthode A et Méthode B |
| `outputs/03_label_distribution.png` | Distribution du score final avec seuils + camembert des labels |
| `outputs/03_gallery_sorted_by_score.png` | Galerie triée par quality_score |
| `outputs/03_examples_per_class.png` | 3 exemples par classe (bad / medium / good) |

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
6. **Boucle d'entraînement** — Warm-up FiLM seul (poids de base gelés), loss ELBO = MSE + KL_weight × KL. 50 steps de démonstration.
7. **Visualisation post-entraînement** — Reconstruction d'une image bad et good à c=0, 0.5, 1. Le score commence à moduler la sortie.
8. **Sauvegarde du checkpoint** — `quality_aware_vae.pt` contenant model_state_dict, ddconfig, embed_dim, history.

### Sorties

| Fichier | Contenu |
|---------|---------|
| `outputs/quality_aware_vae.pt` | Checkpoint du cVAE (modèle + config + historique) |
| `outputs/04_architecture_cvae.png` | Schéma de l'architecture FiLM |
| `outputs/04_conditioning_before_training.png` | Reconstructions à c variés (zero-init) |
| `outputs/04_training_curves.png` | Courbes de loss (total, rec, KL) |
| `outputs/04_conditioning_after_training.png` | Effet du conditionnement post-warm-up |

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

> **Note :** HaarPSI est privilégié car plus fiable que SSIM/PSNR sur les images médicales (Breger et al., 2024).

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

- Après entraînement complet (50 000+ steps) : MSE cVAE < MSE baseline sur les images `bad`, performances comparables sur les `good`.
- Avec 50 steps de démonstration : les deux modèles sont quasi-identiques (attendu).

### Sorties

| Fichier | Contenu |
|---------|---------|
| `outputs/05_evaluation_results.csv` | Métriques par image (baseline + cVAE) sur images réelles |
| `outputs/05_synthetic_results.csv` | Métriques par image × dégradation sur images synthétiques |
| `outputs/05_degradation_grid.png` | Grille des 7 types de dégradations |
| `outputs/05_boxplots.png` | Box plots MSE/SSIM/HaarPSI par classe de qualité |
| `outputs/05_synthetic_metrics.png` | Bar charts baseline vs cVAE par dégradation |
| `outputs/05_gallery_bad.png` | Galerie qualitative sur images bad |
| `outputs/05_gallery_synthetic.png` | Galerie sur images synthétiquement dégradées |

---

## quality_metrics.ipynb — Pipeline autonome simplifié

**Objectif :** Pipeline autonome et léger d'évaluation de qualité NR-IQA, indépendant de la chaîne NB1→NB5. Produit directement un score composite par image.

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
| `outputs/metrics/quality_metrics_scores.csv` | `image, quality_score` pour chaque image |
| `outputs/quality_distributions.png` | Histogrammes des 11 métriques |
| `outputs/quality_correlation.png` | Matrice de corrélation |
| `outputs/quality_comparison.png` | 5 low quality vs 5 high quality |
| `outputs/quality_composite.png` | Distribution du score composite |

---

# Projet IM06 — Récapitulatif

## Contexte et objectif

On travaille sur le dataset **ARCADE** (angiographies coronariennes, 26 classes de
segmentation). L'idée centrale est d'évaluer si **MedVAE** — un autoencodeur médical
pré-entraîné sur des radios thoraciques et mammographies — peut être utilisé comme
encodeur pour compresser les images ARCADE tout en préservant les informations
nécessaires à la segmentation vasculaire.

Pour montrer l'intérêt du fine-tuning de MedVAE sur une nouvelle modalité, on compare
trois pipelines dont le résultat est mesuré par le **Dice score** sur le test set.

---

## Les trois conditions

| Condition | Pipeline | Rôle |
|-----------|----------|------|
| **A** | Image → U-Net complet → Masque | Plafond de référence |
| **B** | Image → MedVAE pré-entraîné (gelé) → Tête segmentation → Masque | Baseline MedVAE généraliste |
| **C** | Image → MedVAE fine-tuné sur ARCADE (gelé) → Tête segmentation → Masque | Contribution principale |

La question centrale : **est-ce que fine-tuner MedVAE sur ARCADE récupère les features
vasculaires que le modèle généraliste (condition B) perdait ?**

---

## Ce qui est fait ✅

### Structure du projet

```
projet/
├── dataset/
│   ├── dataset.py          ✅ existant — ArcadeDataset, get_files_labels, split_files
│   └── dataloaders.py      ✅ existant
│
├── finetune/
│   ├── __init__.py
│   ├── configs/
│   │   └── condition_a.yaml
│   ├── models/
│   │   ├── __init__.py
│   │   └── unet.py
│   ├── losses/
│   │   ├── __init__.py
│   │   └── seg_loss.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── seg_metrics.py
│   ├── trainer/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── plot.py
└── train.py
```

---

### `condition_a.yaml`

Centralise tous les hyperparamètres de l'expérience dans un seul fichier.
L'intérêt est de ne jamais toucher au code pour changer un learning rate ou
un batch size — et d'avoir une trace claire des paramètres de chaque run.
Quand on passera aux conditions B et C, on créera simplement deux nouveaux
fichiers yaml sans modifier le reste.

---

### `unet.py`

On utilise `segmentation_models_pytorch` plutôt que de réimplémenter un U-Net
from scratch. La librairie fournit des architectures éprouvées avec des backbones
pré-entraînés (ici ResNet34 sur ImageNet). Le backbone est adapté automatiquement
pour accepter des images en niveaux de gris (1 canal). Le modèle retourne des
**logits bruts** — le softmax est délibérément absent du modèle car PyTorch
l'intègre directement dans la Cross-Entropy pour des raisons de stabilité numérique.

---

### `seg_loss.py`

On combine deux losses complémentaires :
- La **Cross-Entropy** est stable en début d'entraînement et optimise pixel par pixel
- La **Dice Loss** est robuste au déséquilibre de classes — crucial ici car les pixels
  de fond dominent largement les pixels vasculaires

La loss retourne un tuple `(valeur scalaire, dictionnaire)` — le scalaire pour le
backward, le dictionnaire pour sauvegarder les valeurs détaillées dans l'historique.

---

### `seg_metrics.py`

On utilise `torchmetrics` qui accumule les statistiques sur tous les batches avant
de calculer le Dice final sur l'epoch entière — ce qui est plus précis que de faire
la moyenne des Dice par batch. On calcule le Dice moyen (chiffre résumé pour comparer
A/B/C) et le Dice par classe (pour analyser quelles artères bénéficient du fine-tuning).

---

### `trainer.py`

La boucle d'entraînement est séparée du point d'entrée pour que le même Trainer
puisse être réutilisé pour les conditions B et C sans duplication de code.
Points clés :
- **Early stopping** sur le Dice de validation — évite l'overfitting et le
  gaspillage de temps GPU
- **Gradient clipping** — stabilise l'entraînement sur des structures fines
- **Sauvegarde automatique** du meilleur checkpoint (Dice val maximal)
- **Historique JSON** — toutes les métriques epoch par epoch sont sauvegardées
  localement, sans dépendance à un service externe comme wandb

---

### `plot.py`

Lit les fichiers `history.json` produits par le Trainer et génère des figures
matplotlib. Conçu pour comparer plusieurs conditions sur le même graphe —
la figure de comparaison A/B/C n'apparaît que si on lui passe plusieurs fichiers.

---

### `train.py`

Point d'entrée unique qui assemble tous les modules. Il réutilise directement
`get_files_labels`, `split_files` et `ArcadeDataset` qui existent déjà dans
`dataset/dataset.py` — on évite toute duplication. Seule la construction des
`DataLoader` est dans `train.py` car `dataloaders.py` a des chemins hardcodés
incompatibles avec notre approche yaml.

---

## Ce qu'il reste à faire ❌

### Priorité 1 — Valider et lancer la condition A

 Lancer d'abord un run court (5 epochs) pour s'assurer qu'il n'y a pas d'erreur d'import ou de dimension. Une fois validé,
lancer l'entraînement complet. Le Dice obtenu sur le test set devient le **plafond
de référence** que les conditions B et C devront approcher.

### Priorité 2 — Condition B (MedVAE pré-entraîné)

Deux nouveaux fichiers à créer :

**`finetune/encoder/medvae_encoder.py`** — wrapper autour de la classe `MVAE` du
repo MedVAE. Il charge le modèle pré-entraîné, gèle tous ses poids, et expose une
méthode `encode(image)` qui retourne le latent compressé. Le gel des poids est
essentiel pour que MedVAE joue uniquement le rôle de feature extractor sans être
modifié par la loss de segmentation.

**`finetune/models/seg_head.py`** — décodeur léger (convolutions + upsampling)
qui part du latent compressé et remonte progressivement à la résolution du masque
original. C'est l'équivalent de la moitié décodeur d'un U-Net, mais qui prend
en entrée un latent déjà encodé plutôt qu'une image.

Cette condition donnera le Dice **sans fine-tuning** — probablement inférieur à A
car MedVAE n'a jamais vu d'angiographies.

### Priorité 3 — Fine-tuning MedVAE (condition C)

Le fine-tuning de MedVAE se fait en **deux étapes indépendantes** de la segmentation :

**Stage 1** — fine-tuner l'autoencodeur complet (encodeur + décodeur) sur les images
ARCADE avec les losses de reconstruction de MedVAE (perceptuelle, adversariale,
BiomedCLIP). MedVAE apprend à encoder et reconstruire fidèlement des angiographies.

**Stage 2** — raffiner l'espace latent avec des couches de projection légères guidées
par BiomedCLIP pour mieux capturer les features cliniques.

Une fois fine-tuné, on gèle l'encodeur et on réutilise exactement le même pipeline
que la condition B. Si le Dice de C est significativement supérieur à B, c'est la
preuve que le fine-tuning a bien appris à encoder les features vasculaires.

### Priorité 4 — Analyse des résultats

- Tableau comparatif Dice mean + IoU pour A, B, C
- Dice par classe pour identifier quelles artères bénéficient le plus du fine-tuning
- Visualisations qualitatives (image + masque GT + masque prédit superposés)
- Courbes d'entraînement comparées via `plot.py`

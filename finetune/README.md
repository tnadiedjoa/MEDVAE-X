# Pipeline `finetune/` — récapitulatif technique

Comparaison de quatre pipelines de segmentation d'artères coronariennes sur le
dataset **ARCADE** (26 classes, images rayons-X 512×512 en niveaux de gris). La
question centrale : **une représentation latente compressée par MedVAE permet-elle
une segmentation aussi efficace que le travail directement sur l'image pleine
résolution ?**

Point d'entrée : `python -m finetune.train --config finetune/configs/condition_{a,b,c,d}.yaml`
(toujours depuis la racine du projet). Chaque run crée son dossier
`experiments/runs/<date>_<nom>/` (config exacte, commit, GPU, historique, scores) ;
toutes les expériences et leurs résultats sont dans [`EXPERIMENTS.md`](../EXPERIMENTS.md).

---

## 1. Socle commun à toutes les conditions

Tout ce qui suit est partagé par les conditions A, B, C, D — seul le **modèle**
change d'une condition à l'autre. Le cœur logique vit dans
[`trainer/trainer.py`](trainer/trainer.py).

### 1.1 Données et augmentation ([`dataset.py`](dataset.py))

- Images chargées en niveaux de gris (`convert("L")`), normalisées dans **`[0, 1]`**
  (`/ 255.0`). Le masque est reconstruit à partir des annotations COCO (RLE →
  `category_id` par pixel).
- **Split** (`data.validation: official`, E08) : entraînement sur les **1000**
  images de `seg_train`, sélection du modèle sur le **`seg_val` officiel** (200
  images). Le **test set est un dataset séparé** (phase finale ARCADE, 300 images) —
  jamais vu pendant l'entraînement. L'ancien découpage 80/20 de `seg_train` selon le
  seed reste disponible (`data.validation: split`).
- **Augmentation** (train uniquement, Albumentations, image+masque synchronisés) :
  `HorizontalFlip(0.5)`, `VerticalFlip(0.2)`, `ShiftScaleRotate` (shift 0.05,
  scale 0.1, rotate ±15°, p=0.7), `RandomBrightnessContrast(0.2/0.2, p=0.5)`,
  `GaussNoise` léger (écart-type 3–7 / 255, p=0.3, `data.gauss_noise_std_range`, E07 ;
  sans paramètre, albumentations ≥ 2 ajoute un bruit de ~65–84 / 255), `CLAHE(0.3)`.
  La val/test n'a qu'un `Resize`.
- **Dégradations réalistes** (option, E15, désactivée par défaut) : `data.degradation_aug_p: 0.3`
  applique à 30 % des images une dégradation parmi bruit de Poisson, JPEG (qualité 5–95) et
  flou gaussien (noyau 3–31). Sur A : robustesse aux images dégradées largement rétablie,
  −0.007 de Dice sur images propres (cf. [EXPERIMENTS.md](../EXPERIMENTS.md)).

### 1.2 Fonction de coût ([`losses/seg_loss.py`](losses/seg_loss.py))

`SegLoss` est une **combinaison pondérée de trois termes** :

```
L = dice_weight · L_Dice + ce_weight · L_CE + cl_weight · L_clDice
```

| Terme | Rôle | Détail d'implémentation |
|-------|------|-------------------------|
| **Dice** | recouvrement région-à-région, robuste au déséquilibre de classes | `smp.losses.DiceLoss` mode `multiclass`, `from_logits=True`, `smooth=1e-6` |
| **Cross-Entropy** | classification pixel par pixel, gradients stables | `nn.CrossEntropyLoss` standard sur les 26 classes |
| **clDice** | **préserve la connexité/topologie** des vaisseaux (artères fines) | calculé sur les classes d'avant-plan uniquement (`[:, 1:]`), via squelette morphologique soft |

**Pourquoi clDice ?** Le Dice classique peut être élevé alors qu'une artère est
*coupée en deux* : quelques pixels manquants pénalisent peu le recouvrement mais
brisent la connexité — catastrophique pour des vaisseaux. clDice compare les
**squelettes** (`soft_skel`, obtenu par érosions/dilatations morphologiques
différentiables, `skel_iters=5`) du masque prédit et de la cible, et mesure une
précision/sensibilité topologique. Il pousse le réseau à produire des structures
**continues**.

> **Réglage actuel** : dans les quatre configs, `dice_weight = ce_weight = 0.5` et
> **`cl_weight = 0.0`**. clDice est donc **implémenté mais désactivé** (mis à 0
> pour économiser la mémoire GPU — le calcul du squelette one-hot sur 26 classes
> est coûteux). Pour l'activer : passer `cl_weight` à ~0.2 dans le YAML.

### 1.3 Métriques d'évaluation ([`metrics/seg_metrics.py`](metrics/seg_metrics.py))

Le **score** rapporté n'est *pas* la loss mais des métriques `torchmetrics`
calculées sur les prédictions `argmax` :

Toutes les moyennes sont calculées sur l'ensemble du jeu évalué (pixels cumulés), puis
moyennées sur les **classes présentes** (dans la prédiction ou la vérité terrain).

- **Dice mean** = `MulticlassF1Score(average="macro")` (le Dice est le F1 en
  segmentation), **fond compris**. C'est la métrique de sélection du meilleur modèle.
- **IoU mean** = `MulticlassJaccardIndex(average="macro")`, même moyenne. *(Avant E01,
  `MeanIoU` recevait des masques d'indices au lieu du one-hot attendu : les IoU publiés
  à l'origine sont faux.)*
- **Dice / IoU artères** (`dice_fg_mean`, `iou_fg_mean`, E09) : mêmes moyennes sur les
  25 classes d'artères seules, **fond exclu** (standard ARCADE). Le fond, facile
  (Dice ~0.99), gonfle `dice_mean` d'environ 0.02.
- **Dice / IoU par classe** : vecteurs de 26 valeurs dans `results.json`.
- Un run terminé peut être réévalué avec les métriques actuelles :
  `python -m finetune.evaluate experiments/runs/<run>`.

> ⚠️ Distinction importante : `dice_loss` (dans la loss, dérivable, sur logits) ≠
> `dice_mean` (métrique d'éval, sur `argmax`, non dérivable). Les courbes
> d'historique tracent les deux.

### 1.4 Optimiseur, scheduler et boucle d'entraînement

| Élément | Valeur | Remarque |
|---------|--------|----------|
| **Optimiseur** | `AdamW` | **filtré** : `filter(lambda p: p.requires_grad, ...)` → les params gelés du MedVAE sont **exclus** de l'optimiseur |
| **Weight decay** | `1e-4` | régularisation L2 |
| **Scheduler** | `CosineAnnealingLR` | voir ci-dessous |
| **Gradient clipping** | `clip_grad_norm_ = 1.0` | borne la norme du gradient, stabilise l'entraînement |
| **AMP** | `autocast` + `GradScaler` en **bfloat16** | activé **uniquement si CUDA** — le code tourne aussi en CPU sans erreur |
| **Early stopping** | patience 15 epochs | surveille `dice_mean` de validation |
| **Epochs max** | 100 | |
| **Checkpoint** | `experiments/runs/<run>/best_model_{experiment_name}.pth` | un dossier par run → rien n'est écrasé (non versionné) |
| **Seed** | `experiment.seed` | fixe torch, numpy, random et, via `worker_init_fn`, les augmentations de chaque worker (avant la correction de l'audit, les 4 workers rejouaient la même suite d'augmentations à chaque epoch). `cudnn.deterministic` est activé, mais certaines opérations CUDA (backward de l'interpolation bilinéaire) restent non déterministes : deux runs de même seed peuvent différer légèrement |

#### Le cosine annealing — comment et pourquoi

Le scheduler utilisé est **`CosineAnnealingLR`** (`scheduler.name: "cosine"` dans
tous les YAML), pas à pas **une fois par epoch** (`_step_scheduler`).

**Comment ça marche.** Le learning rate suit une demi-période de cosinus, du LR
initial `η_max` (le `learning_rate` de la config) jusqu'à un plancher `η_min`
(`eta_min: 1e-6`), sur `T_max = 100` epochs :

```
η_t = η_min + ½ (η_max − η_min) · (1 + cos(π · t / T_max))
```

- au début (`t=0`) : `η_t = η_max` → grands pas, exploration rapide ;
- au milieu : décroissance douce, de plus en plus lente ;
- à la fin (`t=T_max`) : `η_t = η_min` → tout petits pas, ajustement fin.

**Pourquoi ce choix.**
1. **Décroissance lisse** plutôt qu'en escalier (vs `StepLR`) : pas de chute
   brutale du LR qui déstabiliserait la convergence.
2. **Grand LR au début** pour sortir vite des mauvaises régions, **petit LR à la
   fin** pour se poser précisément dans un minimum — exactement ce qu'on veut pour
   raffiner des frontières de segmentation pixel-précises.
3. Pas d'hyperparamètre à régler en cours de route (contrairement à
   `ReduceLROnPlateau`, qui est *implémenté en alternative* dans le code mais non
   utilisé par défaut).

> Une alternative `ReduceLROnPlateau` (mode `max` sur le Dice val, `factor 0.5`,
> `patience 7`) est codée dans `_build_scheduler` mais désactivée — il suffit de
> mettre `scheduler.name: "plateau"` pour l'utiliser.

---

## 2. Les conditions

Le rappel spatial MedVAE est essentiel ici : **`medvae_4_1_2d` compresse 4× par
dimension spatiale** (et non 16×). Image `512×512` → latent **`128×128×1`** ; pour
revenir à 512 il faut **2 upsamplings ×2** (`n_upsample: 2`). *(Détail de
nomenclature : le « 4 » du nom = facteur par dimension ; le papier note ce même
modèle `f=16` car 16 = 4² est le facteur surfacique.)*

La normalisation `[0,1] → [-1,1]` exigée par MedVAE (`x*2−1`) est faite **une seule
fois**, dans [`encoder/medvae_encoder.py`](encoder/medvae_encoder.py).

| Cond. | Pipeline | Modèle entraîné | Question scientifique |
|-------|----------|-----------------|-----------------------|
| **A** | U-Net sur image 512×512 | U-Net complet | Référence haute résolution |
| **B** | MedVAE gelé (HF) → tête U-Net sur le latent | tête (7.8 M params) | La compression généraliste suffit-elle ? |
| **C** | MedVAE gelé (fine-tuné ARCADE) → même tête | tête (7.8 M params) | Le fine-tuning du compresseur aide-t-il ? |
| **D** | MedVAE gelé (encode→decode) → U-Net | U-Net complet | S'adapter aux artefacts de compression compense-t-il ? |

### Condition A — référence U-Net ([`models/unet.py`](models/unet.py))

- **U-Net SMP**, encodeur `resnet34` pré-entraîné ImageNet, `in_channels=1`,
  26 classes, sortie en logits bruts (pas de softmax — il est dans la loss).
- Travaille directement sur l'image **512×512** originale.
- `batch_size 8`, `lr 1e-4`. Tous les poids sont entraînables.
- **Rôle** : borne supérieure de référence. Combien peut-on atteindre sans
  aucune compression ?

### Condition B — MedVAE pré-entraîné gelé + tête U-Net sur le latent

- **Encodeur** : `MedVAEEncoder` chargé depuis les poids HuggingFace
  (`checkpoint_path = null`), **entièrement gelé** (`requires_grad=False`, `eval()`).
- **Tête** : `LatentUNetHead` ([`models/seg_head.py`](models/seg_head.py), E04) —
  petit U-Net à la résolution du latent (`128 → 64 → 32 → 16 → 128`, connexions
  skip, `base_channels=64`) pour capter le contexte global nécessaire à
  l'identification des segments, puis 2 blocs `interpolate ×2 + ConvBlock`
  (`n_upsample=2`) jusqu'à 512 et une conv 1×1 vers 26 classes (7.8 M paramètres).
  L'ancienne tête `SegHead` (projection 1×1 puis upsampling, aucune convolution à la
  résolution du latent) reste disponible (`model.architecture: seg_head`).
- `batch_size 4`, **`lr 1e-3`** (E03 : avec 5e-5, la tête restait bloquée sur « tout
  est du fond »). Seule la tête s'entraîne.
- **Rôle** : un compresseur médical *généraliste* (les modèles 2D de MedVAE sont entraînés
  sur des radiographies thoraciques et des mammographies) préserve-t-il assez d'information
  pour segmenter des artères ? B segmente le latent de l'étape 1 de MedVAE (celui de la
  reconstruction), pas la sortie des couches de projection de l'étape 2.

### Condition C — MedVAE fine-tuné ARCADE gelé + tête U-Net sur le latent

- **Architecture identique à B**, au checkpoint près, fourni au lancement :
  ```bash
  python -m finetune.train --config finetune/configs/condition_c.yaml \
    --set encoder.checkpoint_path=experiments/runs/<run medvae_finetune>/best_medvae_finetuned.pth
  # ou : MEDVAE_CKPT=<...>/best_medvae_finetuned.pth sbatch finetune/slurm/train_c.sbatch
  ```
  → l'encodeur charge les poids MedVAE **fine-tunés sur ARCADE** (voir §3), puis
  est gelé comme en B. Sans checkpoint, l'entraînement s'arrête avec une erreur
  (au lieu de tourner silencieusement comme B).
- **Rôle** : isole l'effet du fine-tuning du compresseur. Tout le reste (tête,
  hyperparamètres) étant identique à B, **B vs C** mesure exactement le gain
  apporté par l'adaptation du MedVAE au domaine coronarien.

### Condition D — MedVAE autoencodeur gelé (encode→decode) + U-Net

- **`MedVAEAutoencoder`** : l'image fait le cycle complet
  `[0,1] → [-1,1] → encode → latent 128×128 → decode → [-1,1] → [0,1]`, **gelé**.
  La sortie est une image **512×512 reconstruite** (donc dégradée par la
  compression), de **même interface** que l'originale.
- **U-Net entraînable** (même archi que A) placé *après*, qui apprend à segmenter
  ces images reconstruites.
- `batch_size 8`, `lr 1e-4`. Le MedVAE traite les images par paquets de 2 (son
  attention en 512×512 coûte ~1 Go par image ; sans effet sur le résultat).
- **Rôle** : le U-Net *voit les artefacts de compression dès l'entraînement* et
  peut s'y adapter. **A vs D** mesure le coût pur de la compression à capacité de
  réseau égale.

### Condition A* — variante d'évaluation ([`eval_astar.py`](eval_astar.py))

Pas un entraînement : on prend le **U-Net de la condition A** (entraîné sur images
*originales*) et on l'évalue sur des images **reconstruites par MedVAE**
(`CHECKPOINT_A=experiments/runs/<run A>/best_model_condition_a_unet.pth sbatch finetune/slurm/eval_astar.sbatch`). Mesure le
**décalage de domaine** (domain shift) : un réseau entraîné en pleine résolution
encaisse-t-il la dégradation au test ? À comparer avec D, où le réseau a été
*entraîné* sur les images reconstruites.

### Témoins R et A↓↑ — compression triviale au même taux (E16)

- **R** ([`configs/condition_r.yaml`](configs/condition_r.yaml), `slurm/train_r.sbatch`) : la tête
  de B, même architecture et même entraînement, appliquée à l'image moyennée sur des blocs 4×4
  (128×128×1, autant de valeurs que le latent). **B vs R** dit si le latent MedVAE apporte plus
  qu'une réduction triviale.
- **A↓↑** ([`eval_resampled.py`](eval_resampled.py)) : le U-Net de A, sans réentraînement, sur
  l'image réduite à 128×128 puis ré-agrandie à 512×512 (bilinéaire ou bicubique). **A* vs A↓↑**
  dit si la reconstruction MedVAE préserve plus qu'un rééchantillonnage.

### Garde-fous communs au gel du MedVAE

- Le `train()` est **overridé** dans `MedVAEEncoder` et `MedVAEAutoencoder` :
  même quand le trainer appelle `model.train()`, le MedVAE reste forcé en `eval()`
  (MedVAE n'utilise que des GroupNorm, sans statistiques qui dériveraient, mais le
  mode `eval()` garantit un comportement identique à l'inférence).
- L'`encode`/`forward` du MedVAE est sous `torch.no_grad()`.

---

## 3. Fine-tuning MedVAE — et le rappel des stages 1 & 2 du papier

### 3.1 Les deux stages d'entraînement de MedVAE (papier original)

MedVAE (Varma et al., MIDL 2025) est un **autoencodeur variationnel généraliste**
pré-entraîné sur ~1,05 M d'images médicales, en **deux stages** :

**Stage 1 — autoencodeur de base (qualité de reconstruction).**
L'encodeur et le décodeur sont entraînés end-to-end pour reconstruire l'image, en
combinant **quatre termes** :
1. **Perte perceptuelle** (LPIPS) — similarité dans l'espace de features d'un
   réseau pré-entraîné, plutôt que pixel-à-pixel.
2. **Perte adversariale par patch** (discriminateur PatchGAN) — restaure les
   hautes fréquences / la netteté que la reconstruction L1 seule rend floues.
3. **Consistance d'embedding BiomedCLIP** — pénalité **L2 entre les embeddings
   BiomedCLIP** de l'image d'entrée et de l'image reconstruite, pour conserver les
   features cliniquement pertinentes.
4. **Régularisation KL** — divergence vers une gaussienne standard, **poids très
   faible (1e-6)** pour ne pas dégrader la reconstruction (c'est ce qui en fait un
   *VAE* et rend le latent exploitable/régulier).

**Stage 2 — préservation des features cliniques (raffinement du latent).**
Pour les modèles 2D : **encodeur et décodeur sont gelés**, et de **petites couches
de projection entraînables** sont ajoutées sur le latent. Elles sont optimisées par
une **perte de consistance d'embedding BiomedCLIP** (L2 entre l'embedding de
l'image d'entrée et celui de la représentation latente projetée). Objectif :
garantir que les détails fins, subtils mais cliniquement décisifs, survivent à la
compression — là où un VAE généraliste les lisserait.

> **Pourquoi deux stages ?** Le stage 1 donne un compresseur qui *reconstruit
> bien visuellement* ; le stage 2 force le latent à rester *sémantiquement fidèle
> au contenu clinique*. Reconstruction nette ≠ latent informatif pour une tâche
> médicale : les deux stages couvrent ces deux objectifs distincts.

**Nomenclature des modèles released** : `f` = facteur de sous-échantillonnage
**surfacique**, `C` = canaux latents. Les 4 modèles 2D sont `f16/C1`, `f16/C3`,
`f64/C1`, `f64/C4`. Notre `medvae_4_1_2d` = `f16, C1` (le « 4 » = facteur **par
dimension**, 16 = 4²).

### 3.2 Notre fine-tuning ARCADE ([`finetune_medvae.py`](finetune_medvae.py))

Avant la condition C, on **ré-adapte tout le MedVAE** au domaine coronarien — un
fine-tuning léger et volontairement simple (≠ les stages complexes du papier) :

- **Dataset** : images ARCADE *seules* (sans masque), redimensionnées 512,
  normalisées `[0,255] → [-1,1]` (`/127.5 − 1`). Split 90 % / 10 % de `seg_train`.
- **Tout le MedVAE est dégelé** (`requires_grad=True`, `mvae.train()`) — c'est le
  seul moment où il s'entraîne.
- **Loss = `L1` pure** entre image originale et `decode(encode(image))`. Pas de
  perceptuelle, pas de GAN, pas de KL : on cherche juste à recaler la
  reconstruction sur la distribution des rayons-X coronariens.
- **Optimiseur** : `AdamW`, **`lr = 4.5e-6`** (très petit — fine-tuning, pas
  entraînement *from scratch*), `weight_decay = 0`.
- **Scheduler** : `CosineAnnealingLR`, `T_max = 50` (= nb d'epochs), `eta_min =
  1e-7` — même logique cosinus que §1.4, adaptée à la durée du fine-tuning.
- Batch de 4 traité en 4 micro-batchs de 1 avec accumulation de gradient
  (`micro_batch_size: 1`) pour tenir sur une RTX 3090 (24 Go) ; identique au batch
  entier (L1 moyenne, normalisations par image).
- `grad_clip = 1.0`, **early stopping** patience 10 sur la **L1 de validation**.
- Sauvegarde `best_medvae_finetuned.pth` dans le dossier du run → passé à la
  condition C (`--set encoder.checkpoint_path=...` ou `MEDVAE_CKPT`).

À l'usage (condition C), `MedVAEEncoder._load_finetuned` recharge ces poids
(formats `state_dict` / Lightning / `Phase1Trainer`) : les clés du MedVAE commencent
par `model.` et non `encoder.`, c'est donc le **modèle complet** qui est rechargé
(0 clé manquante, vérifié), puis tout est re-gelé.

---

## 4. Tableau de synthèse des hyperparamètres

| | Cond. A | Cond. B | Cond. C | Cond. D | FT MedVAE |
|---|---|---|---|---|---|
| Modèle entraîné | U-Net r34 | LatentUNetHead | LatentUNetHead | U-Net r34 | MedVAE complet |
| MedVAE | — | gelé (HF) | gelé (FT) | gelé (enc→dec) | **entraîné** |
| Entrée réseau | image 512 | latent 128 | latent 128 | image reconstr. 512 | image 512 |
| `batch_size` | 8 | 4 | 4 | 8 | 4 |
| `learning_rate` | 1e-4 | 1e-3 | 1e-3 | 1e-4 | 4.5e-6 |
| `weight_decay` | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 0 |
| Scheduler | cosine T=100 | cosine T=100 | cosine T=100 | cosine T=100 | cosine T=50 |
| Loss | Dice+CE | Dice+CE | Dice+CE | Dice+CE | L1 |
| Validation | seg_val officiel | seg_val | seg_val | seg_val | 10 % de seg_train |
| Early stop (patience) | 15 (Dice val) | 15 | 15 | 15 | 10 (L1 val) |
| AMP bf16 | ✓ (GPU) | ✓ | ✓ | ✓ | — |

---

## Sources

- [MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders (arXiv:2502.14753)](https://arxiv.org/abs/2502.14753)
- [Version HTML du papier](https://arxiv.org/html/2502.14753v1)
- [Dépôt officiel StanfordMIMI/MedVAE](https://github.com/StanfordMIMI/MedVAE)

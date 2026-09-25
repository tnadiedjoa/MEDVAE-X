# Pipeline `finetune/` — récapitulatif technique

Comparaison de quatre pipelines de segmentation d'artères coronariennes sur le
dataset **ARCADE** (26 classes, images rayons-X 512×512 en niveaux de gris). La
question centrale : **une représentation latente compressée par MedVAE permet-elle
une segmentation aussi efficace que le travail directement sur l'image pleine
résolution ?**

Point d'entrée : `python -m finetune.train --config finetune/configs/condition_{a,b,c,d}.yaml`
(toujours depuis la racine du projet).

---

## 1. Socle commun à toutes les conditions

Tout ce qui suit est partagé par les conditions A, B, C, D — seul le **modèle**
change d'une condition à l'autre. Le cœur logique vit dans
[`trainer/trainer.py`](trainer/trainer.py).

### 1.1 Données et augmentation ([`dataset.py`](dataset.py))

- Images chargées en niveaux de gris (`convert("L")`), normalisées dans **`[0, 1]`**
  (`/ 255.0`). Le masque est reconstruit à partir des annotations COCO (RLE →
  `category_id` par pixel).
- **Split** : 80 % train / 20 % val sur les images d'entraînement
  (`split_dataset`, seed 42). Le **test set est un dataset séparé** (phase finale
  ARCADE) — jamais vu pendant l'entraînement.
- **Augmentation** (train uniquement, Albumentations, image+masque synchronisés) :
  `HorizontalFlip(0.5)`, `VerticalFlip(0.2)`, `ShiftScaleRotate` (shift 0.05,
  scale 0.1, rotate ±15°, p=0.7), `RandomBrightnessContrast(0.2/0.2, p=0.5)`,
  `GaussNoise(0.3)`, `CLAHE(0.3)`. La val/test n'a qu'un `Resize`.

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

- **Dice mean** = `MulticlassF1Score(average="macro")` — le Dice score est
  mathématiquement le F1 en segmentation. `macro` = **moyenne non pondérée sur
  les 26 classes** (chaque artère compte autant, même rare). C'est la métrique de
  sélection du meilleur modèle.
- **IoU mean** = `MeanIoU` (Jaccard).
- **Dice par classe** : vecteur de 26 valeurs, sauvegardé dans `results.json` pour
  l'analyse fine (certaines artères rares tombent à 0 — voir résultats).

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
| **Checkpoint** | `best_model_{experiment_name}.pth` | nommé par expérience → les conditions ne s'écrasent pas |

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

## 2. Les quatre conditions

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
| **B** | MedVAE gelé (HF) → tête | tête légère | La compression généraliste suffit-elle ? |
| **C** | MedVAE gelé (fine-tuné ARCADE) → tête | tête légère | Le fine-tuning du compresseur aide-t-il ? |
| **D** | MedVAE gelé (encode→decode) → U-Net | U-Net complet | S'adapter aux artefacts de compression compense-t-il ? |

### Condition A — référence U-Net ([`models/unet.py`](models/unet.py))

- **U-Net SMP**, encodeur `resnet34` pré-entraîné ImageNet, `in_channels=1`,
  26 classes, sortie en logits bruts (pas de softmax — il est dans la loss).
- Travaille directement sur l'image **512×512** originale.
- `batch_size 8`, `lr 1e-4`. Tous les poids sont entraînables.
- **Rôle** : borne supérieure de référence. Combien peut-on atteindre sans
  aucune compression ?

### Condition B — MedVAE pré-entraîné gelé + tête légère

- **Encodeur** : `MedVAEEncoder` chargé depuis les poids HuggingFace
  (`checkpoint_path = null`), **entièrement gelé** (`requires_grad=False`, `eval()`).
- **Tête** : `SegHead` ([`models/seg_head.py`](models/seg_head.py)) — légère.
  Reçoit le latent `128×128×1`, fait une projection 1×1 vers `base_channels=128`,
  puis **2 blocs `interpolate ×2 + ConvBlock`** (`n_upsample=2`) pour remonter
  `128 → 256 → 512`, et une conv 1×1 finale vers 26 classes.
- `batch_size 4`, `lr 5e-5`. Seule la tête s'entraîne (peu de params).
- **Rôle** : un compresseur médical *généraliste* (entraîné sur ~1M d'images
  toutes modalités) préserve-t-il assez d'information pour segmenter des artères ?

### Condition C — MedVAE fine-tuné ARCADE gelé + tête légère

- **Architecture identique à B**, à **un seul champ** près dans le YAML :
  ```yaml
  encoder:
    checkpoint_path: "experiments/runs/<run medvae_finetune>/best_medvae_finetuned.pth"
  ```
  → l'encodeur charge les poids MedVAE **fine-tunés sur ARCADE** (voir §3), puis
  est gelé comme en B.
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
- `batch_size 8`, `lr 1e-4`.
- **Rôle** : le U-Net *voit les artefacts de compression dès l'entraînement* et
  peut s'y adapter. **A vs D** mesure le coût pur de la compression à capacité de
  réseau égale.

### Condition A* — variante d'évaluation ([`eval_astar.py`](eval_astar.py))

Pas un entraînement : on prend le **U-Net de la condition A** (entraîné sur images
*originales*) et on l'évalue sur des images **reconstruites par MedVAE**. Mesure le
**décalage de domaine** (domain shift) : un réseau entraîné en pleine résolution
encaisse-t-il la dégradation au test ? À comparer avec D, où le réseau a été
*entraîné* sur les images reconstruites.

### Garde-fous communs au gel du MedVAE

- Le `train()` est **overridé** dans `MedVAEEncoder` et `MedVAEAutoencoder` :
  même quand le trainer appelle `model.train()`, le MedVAE reste forcé en `eval()`
  → ses **BatchNorm gardent leurs statistiques fixes** (sinon elles dériveraient
  sur les batchs ARCADE et fausseraient l'encodage).
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
  normalisées `[0,255] → [-1,1]` (`/127.5 − 1`). Split 90 % / 10 %.
- **Tout le MedVAE est dégelé** (`requires_grad=True`, `mvae.train()`) — c'est le
  seul moment où il s'entraîne.
- **Loss = `L1` pure** entre image originale et `decode(encode(image))`. Pas de
  perceptuelle, pas de GAN, pas de KL : on cherche juste à recaler la
  reconstruction sur la distribution des rayons-X coronariens.
- **Optimiseur** : `AdamW`, **`lr = 4.5e-6`** (très petit — fine-tuning, pas
  entraînement *from scratch*), `weight_decay = 0`.
- **Scheduler** : `CosineAnnealingLR`, `T_max = 50` (= nb d'epochs), `eta_min =
  1e-7` — même logique cosinus que §1.4, adaptée à la durée du fine-tuning.
- `grad_clip = 1.0`, **early stopping** patience 10 sur la **L1 de validation**.
- Sauvegarde `best_medvae_finetuned.pth` → renseigné dans `condition_c.yaml`.

À l'usage (condition C), `MedVAEEncoder._load_finetuned` recharge ces poids
(supporte les formats `state_dict` / Lightning / `Phase1Trainer`), ne garde que
l'**encodeur** (`encoder.*`), puis re-gèle tout.

---

## 4. Tableau de synthèse des hyperparamètres

| | Cond. A | Cond. B | Cond. C | Cond. D | FT MedVAE |
|---|---|---|---|---|---|
| Modèle entraîné | U-Net r34 | SegHead | SegHead | U-Net r34 | MedVAE complet |
| MedVAE | — | gelé (HF) | gelé (FT) | gelé (enc→dec) | **entraîné** |
| Entrée réseau | image 512 | latent 128 | latent 128 | image reconstr. 512 | image 512 |
| `batch_size` | 8 | 4 | 4 | 8 | 4 |
| `learning_rate` | 1e-4 | 5e-5 | 5e-5 | 1e-4 | 4.5e-6 |
| `weight_decay` | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 0 |
| Scheduler | cosine T=100 | cosine T=100 | cosine T=100 | cosine T=100 | cosine T=50 |
| Loss | Dice+CE | Dice+CE | Dice+CE | Dice+CE | L1 |
| Early stop (patience) | 15 (Dice val) | 15 | 15 | 15 | 10 (L1 val) |
| AMP bf16 | ✓ (GPU) | ✓ | ✓ | ✓ | — |

---

## Sources

- [MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders (arXiv:2502.14753)](https://arxiv.org/abs/2502.14753)
- [Version HTML du papier](https://arxiv.org/html/2502.14753v1)
- [Dépôt officiel StanfordMIMI/MedVAE](https://github.com/StanfordMIMI/MedVAE)

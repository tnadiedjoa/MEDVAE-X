# Journal d'expériences

Toutes les modifications testées sur le projet, **qu'elles aient amélioré les résultats
ou non**. Chaque modification est comparée à l'état du projet **juste avant** elle,
mesuré dans les mêmes conditions (même GPU, même dataset, même seed).

## Résumé

| # | Axe | Ce qui a été testé | Effet mesuré | Décision |
|---|---|---|---|---|
| E01 | D | Calcul de l'IoU (masques d'indices passés à `MeanIoU`) | IoU publiés faux (0.50 → ~0.33) | corrigé |
| E02 | D | Reproduction sur RTX 3090 | Dice à ±0.006 de l'origine ; B et C ne détectent aucune artère | référence |
| E03 | D | lr de la tête de B/C : 5e-5 → 1e-3 | B 0.039 → 0.097, C 0.039 → 0.091 | gardé |
| E04 | D | Tête U-Net à la résolution du latent (B/C) | B → 0.409, C → 0.398 : le latent est exploitable | gardé |
| E06 | D | 3 seeds par condition | variabilité ±0.01 ; D > A pour les 3 seeds | protocole |
| E07 | D | Bruit d'augmentation ramené à l'intensité voulue | A +0.010 (faible) | gardé |
| E08 | D | Validation sur le seg_val officiel (1000 images d'entraînement) | A +0.030 | gardé |
| E09 | D | Dice / IoU sur les artères seules (fond exclu) | ~0.024 sous le Dice avec fond, classement inchangé | ajouté |
| E10 | D | Nouvel état de référence (E07 + E08) sur A, B, C, D, A* | A 0.452, A* 0.448, B 0.417, C 0.427, D 0.455 ; D ≈ A désormais | référence |
| E11 | A | Robustesse mesurée par rapport à l'image propre | débruite le Poisson (+1.2 à +6.6 dB), ne restaure ni JPEG ni flou | refait |
| E12 | A | Segmentation d'images dégradées (A, A*, D) | à venir | — |
| E13 | B | FiLM refait (256×256, entraînement identique, contrôles sur c) | fine-tuning +7.4 dB ; FiLM ≈ baseline ; c inutilisé (C) ou à peine (A, B : ≤ 0.04 dB) | refait |
| E14 | C | JEPA refait (test exclu du pré-entraînement, linear probe) | en cours | — |

Axes : A robustesse, B conditionnement FiLM, C JEPA, D segmentation (dont les conditions
A, A*, B, C et D, décrites dans le README). Les effets de segmentation sont des Dice sur le test set.

## Protocole

1. **Avant** : le run de référence est celui de l'état courant du projet. S'il n'existe
   pas dans les conditions actuelles, on le lance d'abord.
2. **Modification** : on commite le changement, puis on lance le run (un run lancé avec
   des modifications non commitées est marqué `git_dirty` dans son `meta.json`).
3. **Après** : on compare, on note l'entrée ci-dessous, avec la conclusion.
4. **Si la modification est conservée** : `python scripts/promote_run.py <run>` met à
   jour les résultats officiels (`finetune/results/`), puis README et figures.

Chaque run vit dans `experiments/runs/<date>_<nom>/` : `config.yaml` (config exacte),
`meta.json` (commit, GPU, job Slurm), historique et scores. Les checkpoints `.pth`
y restent mais ne sont pas versionnés.

Lancer un run avec une valeur modifiée, sans toucher aux yaml :

```bash
python -m finetune.train --config finetune/configs/condition_a.yaml \
  --set training.learning_rate=3e-4 --run-name lr3e-4_condition_a
```

**Métriques (segmentation, test set de 300 images).** Dice et IoU par classe, calculés
sur l'ensemble du test set, puis moyennés sur les classes présentes (dans la prédiction
ou la vérité terrain), fond compris. Dice ≥ IoU toujours.

### Modèle d'entrée

```markdown
## EXX — Titre court (AAAA-MM-JJ)

**Hypothèse** : ce qu'on pense améliorer, et pourquoi.
**Modification** : ce qui change (commit `abc1234`).
**Runs** : avant `experiments/runs/...` — après `experiments/runs/...`

| | Dice | IoU |
|---|---|---|
| Avant | | |
| Après | | |

**Conclusion** : gardé / abandonné, et ce qu'on en retient.
```

---

## E01 — Correction du calcul de l'IoU (2026-09-25)

**Constat** : pour les conditions A et D, l'IoU rapporté était supérieur au Dice, ce qui
est impossible (Dice ≥ IoU pour toute classe).

**Cause** : `torchmetrics.segmentation.MeanIoU` attend par défaut des masques one-hot ;
le projet lui passait des masques d'indices de classes, mal interprétés sans erreur.
Sur un cas test où le modèle prédit « fond » partout, il renvoyait 0.000 au lieu de 0.283.

**Modification** (commit `93390f7`) : `MeanIoU` remplacé par `MulticlassJaccardIndex(average="macro")`,
calculé exactement comme le Dice. Test de non-régression : `tests/test_seg_metrics.py`
(échoue sur l'ancien code, passe sur le nouveau).

**Impact** : le Dice est inchangé (son calcul était correct). **Tous les IoU publiés
jusqu'ici sont faux** ; les checkpoints d'origine n'étant pas disponibles, ils ne peuvent
pas être recalculés ; les IoU corrects sont ceux de E02.

**Conclusion** : gardé (correction de bug).

## E02 — Reproduction de l'état actuel sur RTX 3090 (2026-09-25)

**Objectif** : obtenir le point de départ mesuré dans nos conditions. Les résultats
d'origine ont été obtenus sur H100/A100, avec d'autres versions des librairies et une
copie d'ARCADE dont la provenance exacte n'est pas connue ; nous utilisons des RTX 3090,
torch 2.14 et ARCADE depuis Zenodo.

**Code** : commit `10f4048`, configs inchangées.
**Runs** : `experiments/runs/2026-09-25_*_e02_*`

**Adaptations à la 3090 (24 Go), sans effet sur les calculs** : le fine-tuning MedVAE
traite chaque batch de 4 en 4 micro-batchs de 1 avec accumulation de gradient, et la
condition D passe les images dans le MedVAE gelé par paquets de 2. Dans les deux cas le
résultat est mathématiquement identique (loss L1 moyenne, normalisations par image).

| Condition | Dice origine | Dice E02 | IoU E02 (corrigé) | Epochs |
|---|---|---|---|---|
| A — U-Net | 0.433 | **0.436** | 0.330 | 100 |
| A* — U-Net A sur images reconstruites | 0.435 | **0.437** | 0.329 | — (éval.) |
| B — latent MedVAE + tête | 0.038 | **0.039** | 0.039 | 22 (early stop) |
| C — latent MedVAE fine-tuné + tête | 0.039 | **0.039** | 0.039 | 22 (early stop) |
| D — MedVAE → U-Net | 0.451 | **0.445** | 0.335 | 100 |

Fine-tuning MedVAE (prérequis de C) : loss L1 de validation 0.02617 → 0.02408 (epoch
49/50), identique à l'original (0.02613 → 0.02408, epoch 49/50).

**Conclusions**

- **Reproduction réussie** : tous les Dice sont à ±0.006 de l'original malgré le
  changement de GPU, de librairies et de source du dataset. E02 sert de référence
  « avant » pour les expériences suivantes.
- **Les vrais IoU sont ~0.33** pour A, A* et D (contre ~0.50 annoncés, cf. E01).
- **B et C ne détectent aucune artère** : Dice de 0.98 sur le fond et < 0.01 sur les
  25 classes d'artères, aucun progrès en validation, arrêt à l'epoch 22. Ils prédisent
  « fond » partout. La conclusion actuelle du projet (« le latent n'est pas adapté à la
  prédiction dense ») repose donc sur des modèles qui n'ont rien appris : il reste à
  établir si c'est une limite réelle du latent ou un problème d'entraînement.
- **D ≈ A** (0.445 vs 0.436) : l'écart A/D est du même ordre que la variation
  origine/reproduction (0.006), il n'est donc pas significatif sur un seul run.

## E03 — Learning rate de la tête de B et C : 5e-5 → 1e-3 (2026-09-26)

**Diagnostic préalable** (`experiments/diagnostics/diag_latent_b.py`, commit `b7854d8`) :
le latent MedVAE n'est pas en cause. Le bruit du tirage aléatoire est négligeable
(écart-type 0.001 contre 7 pour le signal) et le latent contient autant d'information
sur les vaisseaux qu'une image réduite à la même taille. En revanche, sur 8 images, la
tête de B n'apprend presque rien avec son learning rate (Dice 0.05), mais atteint
Dice 0.72 et 13 artères avec lr = 1e-3.

**Hypothèse** : B et C restent bloqués sur « tout est du fond » parce que leur learning
rate (5e-5) est trop faible pour une tête entraînée depuis zéro.
**Modification** : `training.learning_rate` 5e-5 → 1e-3, rien d'autre (commit `247793b`,
`--set training.learning_rate=1e-3`). C utilise le MedVAE fine-tuné de E02.
**Runs** : avant `*_e02_condition_{b,c}` — après `2026-09-26_025705_e03_lr1e-3_condition_{b,c}`

| | Dice | IoU | Artères détectées (Dice > 0.01) | Epochs |
|---|---|---|---|---|
| B avant (E02) | 0.039 | 0.039 | 0 / 25 | 22 (early stop) |
| **B après** | **0.097** | **0.070** | **12 / 25** | 97 (early stop) |
| C avant (E02) | 0.039 | 0.039 | 0 / 25 | 22 (early stop) |
| **C après** | **0.091** | **0.066** | **12 / 25** | 100 |

**Conclusion** : gardé (learning rate 1e-3 dans les configs de B et C, résultats promus). B et C apprennent enfin (Dice ×2.5, 12 artères détectées sur
les plus grosses, Dice jusqu'à 0.25 par classe), mais restent très loin de A (0.436).

- **Sous-apprentissage** : Dice d'entraînement (~0.09) ≈ Dice de validation (~0.11) ;
  le modèle n'arrive pas à mieux faire même sur les images d'entraînement. La limite
  est la capacité de la tête (aucune convolution à la résolution du latent, champ
  réceptif très petit), pas le sur-apprentissage.
- **C ≈ B** : le fine-tuning de MedVAE sur ARCADE n'apporte rien ici (écart 0.006, du
  même ordre que la variation d'un run à l'autre).
- La conclusion d'origine (« le latent n'est pas adapté à la prédiction dense ») n'est
  pas établie : il faut d'abord tester une tête capable d'exploiter le contexte.

## E04 — Tête U-Net à la résolution du latent pour B et C (2026-09-26)

**Hypothèse** : après E03, B et C sous-apprennent (Dice d'entraînement ≈ validation
≈ 0.09). La tête `SegHead` ne fait aucune convolution à la résolution du latent : chaque
pixel ne voit que son voisinage immédiat, alors qu'identifier un segment d'artère (IVA,
circonflexe…) demande de voir une grande partie de l'image.
**Modification** : nouvelle tête `LatentUNetHead` (commit `388ece7`) : petit U-Net sur le
latent (128 → 64 → 32 → 16 → 128, connexions skip), puis le même upsampling ×4 vers
512×512 ; `base_channels=64`, 7.8 M paramètres (U-Net de A : 24 M). Rien d'autre ne
change (même latent, loss, lr 1e-3). Lancé avec
`--set model.architecture=latent_unet_head --set model.base_channels=64`.
Test préalable sur 8 images (lr 1e-3, 1000 pas) : Dice 0.995 contre 0.749 pour `SegHead`.
**Runs** : avant `*_e03_lr1e-3_condition_{b,c}` — après `2026-09-26_114934_e04_latent_unet_head_condition_{b,c}`

| | Dice | IoU | Artères détectées | Dice train / val (meilleur) |
|---|---|---|---|---|
| B avant (E03) | 0.097 | 0.070 | 12 / 25 | 0.090 / 0.109 |
| **B après** | **0.409** | **0.304** | **21 / 25** | 0.452 / 0.444 |
| C avant (E03) | 0.091 | 0.066 | 12 / 25 | 0.092 / 0.107 |
| **C après** | **0.398** | **0.292** | **22 / 25** | 0.466 / 0.446 |
| *Référence A (U-Net sur l'image)* | *0.436* | *0.330* | *22 / 25* | *0.650 / 0.507* |
| *Référence D (MedVAE → U-Net)* | *0.445* | *0.335* | *23 / 25* | *0.567 / 0.497* |

**Conclusion** : gardé (`latent_unet_head`, `base_channels: 64` dans les configs de B et C, résultats promus). Dice ×4 pour B et C, qui arrivent à 94 % du Dice de A (0.409 contre
0.436) en ne partant que du latent MedVAE compressé ×16.

- **La conclusion d'origine du projet est renversée** : le latent MedVAE *est* exploitable
  pour la segmentation dense des coronaires ; l'échec venait de l'entraînement (E03) et
  de la tête (E04), pas du latent.
- **C ≈ B** encore une fois (0.398 contre 0.409) : fine-tuner MedVAE sur ARCADE n'aide pas.
- **Encore du sous-apprentissage** (train ≈ val ≈ 0.45, contre 0.65 / 0.51 pour A) et
  les courbes plafonnent quand le learning rate atteint son minimum (100 epochs, meilleur
  score aux epochs 90-97) : une tête plus large ou un entraînement plus long pourraient
  encore réduire l'écart avec A.

## E05 — Validation de E04 (2026-09-26)

`LatentUNetHead` (`base_channels: 64`) devient la tête de B et C dans les configs
(commit `e361732`) et les runs E04 sont promus en résultats officiels.

## E06 — Variabilité d'un seed à l'autre (2026-09-26)

**Objectif** : jusqu'ici chaque condition n'avait qu'un run ; sans connaître la
variabilité d'un entraînement à l'autre, on ne peut pas dire si un écart est réel.
**Modification** : aucune ; seeds 43 et 44 pour A, B, C et D (le seed 42 = runs E02 pour
A et D, E04 pour B et C, même configuration). Le seed fixe aussi le découpage
entraînement/validation (80/20 de seg_train) : la variabilité mesurée inclut cet effet.
**Runs** : `2026-09-26_*_e06_seed4{3,4}_condition_{a,b,c,d}` (commit `e361732`)

| Condition | Dice | Dice artères | IoU artères |
|---|---|---|---|
| A — U-Net | 0.424 ± 0.010 | 0.400 ± 0.011 | 0.294 ± 0.007 |
| B — latent MedVAE + tête U-Net | 0.409 ± 0.009 | 0.385 ± 0.009 | 0.276 ± 0.005 |
| C — latent MedVAE fine-tuné + tête U-Net | 0.397 ± 0.005 | 0.372 ± 0.005 | 0.264 ± 0.004 |
| D — MedVAE → U-Net | 0.441 ± 0.003 | 0.418 ± 0.003 | 0.305 ± 0.003 |

(moyenne ± écart-type sur 3 seeds ; « artères » = moyenne sur les 25 classes d'artères,
fond exclu, cf. E09 ; tableau produit par `scripts/summarize_runs.py`)

**Conclusions**

- **La variabilité d'un seed à l'autre est d'environ ±0.01 de Dice** (jusqu'à 0.02 d'écart
  entre deux seeds pour A) : aucun écart plus petit ne doit être interprété sur un run.
- **D bat A pour les 3 seeds** (+0.009, +0.020, +0.024 ; +0.018 en moyenne) : l'avantage de
  D annoncé à l'origine tient, alors qu'il n'était pas démontrable sur un seul run (E02).
- **B < A** (−0.015 en moyenne) et **C < B** pour les 3 seeds : le latent MedVAE seul reste
  un peu en dessous de l'image entière, et fine-tuner MedVAE sur ARCADE dégrade légèrement.

## E07 — Intensité du bruit gaussien d'augmentation (2026-09-26)

**Constat** : `A.GaussNoise(p=0.3)` sans paramètre ajoute, avec albumentations ≥ 2 (2.0.8
installé), un bruit d'écart-type 20 à 44 % du maximum (~65-84 niveaux sur 255) : 30 % des
images d'entraînement sont presque détruites. Avec albumentations 1.x, la même ligne donnait
un bruit d'écart-type ~3-7 niveaux, sans doute l'intention d'origine.
**Modification** : option `data.gauss_noise_std_range` (commit `6dca6a2`), testée avec
`[0.0124, 0.0277]` (écart-type 3.2-7.1 / 255, l'équivalent de l'ancien défaut) sur A.
**Runs** : avant = A de E06 — après `2026-09-26_*_e07_light_noise_seed4{2,3,4}_condition_a`

| A | Dice | Dice artères | IoU artères |
|---|---|---|---|
| Avant (E06) | 0.424 ± 0.010 | 0.400 ± 0.011 | 0.294 ± 0.007 |
| Bruit léger | 0.433 ± 0.008 | 0.410 ± 0.008 | 0.294 ± 0.006 |

Écart apparié par seed : +0.004, +0.007, +0.018 (+0.010 en moyenne).

**Conclusion** : gardé. Le gain est positif pour les 3 seeds mais du même ordre que la
variabilité, et nul en IoU : l'effet est faible. La modification est conservée surtout parce
qu'elle rétablit l'augmentation voulue à l'origine.

## E08 — Validation sur le seg_val officiel (2026-09-26)

**Constat** : ARCADE fournit une validation officielle (`seg_val`, 200 images annotées)
jamais utilisée : la validation était prise sur 20 % de `seg_train`, et le modèle
s'entraînait sur 800 images au lieu de 1000.
**Modification** : option `data.validation: official` (commit `6dca6a2`) : entraînement sur
les 1000 images de `seg_train`, sélection du modèle sur `seg_val`. Le seed ne change plus
le découpage, seulement l'initialisation et l'ordre des batchs. Testée sur A.
**Runs** : avant = A de E06 — après `2026-09-26_*_e08_official_val_seed4{2,3,4}_condition_a`

| A | Dice | Dice artères | IoU artères |
|---|---|---|---|
| Avant (E06) | 0.424 ± 0.010 | 0.400 ± 0.011 | 0.294 ± 0.007 |
| Validation officielle | **0.454 ± 0.018** | **0.431 ± 0.019** | **0.316 ± 0.011** |

Écart apparié par seed : +0.028, +0.015, +0.048 (+0.030 en moyenne).

**Conclusion** : gardé. Gain net et positif pour les 3 seeds, sur toutes les métriques.

## E09 — Métriques sur les artères seules (2026-09-26)

**Modification** : `SegMetrics` calcule aussi `dice_fg_mean` et `iou_fg_mean`, moyennes sur
les 25 classes d'artères présentes (prédites ou réelles), fond exclu, comme le challenge
ARCADE (commit `f3c156c`). `finetune/evaluate.py` réévalue un run terminé avec les
métriques actuelles ; tous les runs E02 à E06 ont été réévalués (Dice identique à 1e-4
près ; D varie de 2e-4 car MedVAE tire son latent au hasard). La classe 12 est absente du
test set ; la catégorie 26 (« stenosis ») n'apparaît jamais dans les annotations de
segmentation, les 26 classes du modèle (fond + 25 segments) sont donc correctes.

**Effet** : le fond, facile (Dice ~0.99), gonflait la moyenne : le Dice sur les artères est
~0.024 plus bas que `dice_mean` pour toutes les conditions (ex. A : 0.424 → 0.400). Le
classement des conditions est inchangé.

## E10 — Nouvel état de référence : E07 + E08 sur A, B, C, D (2026-09-28)

**Modification** : les deux changements validés sur A (validation sur seg_val, E08 ; bruit
d'augmentation léger, E07) appliqués aux configs A à D (commit `f3b6538`), 3 seeds par
condition ; A* = U-Net A du seed 42 appliqué aux reconstructions MedVAE.
**Runs** : avant = E06 (mêmes seeds) — après `2026-09-2{7,8}_*_e10_seed4{2,3,4}_condition_{a,b,c,d}`
et `2026-09-28_*_e10_seed42_condition_astar`

| Condition | Dice avant (E06) | Dice après (E10) | Dice artères | IoU artères | Écart par seed (42 / 43 / 44) |
|---|---|---|---|---|---|
| A — U-Net | 0.424 ± 0.010 | **0.452 ± 0.005** | 0.430 ± 0.005 | 0.308 ± 0.002 | +0.018 / +0.028 / +0.039 |
| A* — U-Net A sur reconstructions | — | 0.448 (seed 42) | 0.425 | 0.303 | −0.006 par rapport à A |
| B — latent MedVAE | 0.409 ± 0.009 | **0.417 ± 0.008** | 0.393 ± 0.008 | 0.277 ± 0.005 | +0.017 / +0.014 / −0.007 |
| C — latent MedVAE fine-tuné | 0.397 ± 0.005 | **0.427 ± 0.002** | 0.403 ± 0.002 | 0.285 ± 0.002 | +0.028 / +0.028 / +0.035 |
| D — MedVAE → U-Net | 0.441 ± 0.003 | **0.455 ± 0.004** | 0.432 ± 0.004 | 0.311 ± 0.003 | +0.015 / +0.015 / +0.011 |

**Conclusions**

- **Toutes les conditions progressent**, surtout A et C (+0.03) ; B le moins (+0.008, un
  seed en baisse). Résultats promus (seed 42 de chaque condition, comme jusqu'ici).
- **D ≈ A** : +0.003 en moyenne (+0.005, +0.006, −0.004), dans la variabilité. L'avantage de
  D mesuré en E06 (+0.018, 3 seeds sur 3) disparaît presque. Hypothèse : avec le bruit
  d'augmentation trop fort d'avant E07, 30 % des images d'entraînement de A étaient presque
  détruites, alors que D les voyait après MedVAE, qui débruite (E11) ; D était donc moins
  pénalisé par ce bruit. A a d'ailleurs gagné deux fois plus que D (+0.028 contre +0.013).
- **C ≥ B pour les 3 seeds** (+0.010 en moyenne), l'inverse de E06 : fine-tuner MedVAE sur
  ARCADE aide peut-être un peu la segmentation depuis le latent, mais l'écart est de l'ordre
  de la variabilité.
- **A* ≈ A** (−0.006 sur un seed) : la compression seule coûte peu.
- La segmentation depuis le latent (B, C) reste ~0.03 en dessous de A (92-94 % de son Dice).

## E11 — Robustesse de MedVAE aux dégradations, mesure corrigée (2026-09-27)

**Constat sur l'étude d'origine** (axe A, `medvae_eval/scripts/run_maskedPSNR_sweep.py`) :
- la métrique principale compare la reconstruction à l'entrée **dégradée** : une image
  floue étant plus facile à reproduire, « les images floues se reconstruisent mieux » est
  presque tautologique et ne dit rien de la robustesse ;
- les images sont ramenées dans [-1, 1] par un min-max propre à chaque image (qui change
  avec la dégradation) : les trois métriques ne sont pas sur la même échelle ;
- le latent est tiré au hasard, le bruit n'est pas seedé, 50 images de seg_train seulement.

**Nouveau protocole** (`medvae_eval/robustness/sweep.py`, commit `34b8619`) : latent
déterministe (moyenne du posterior), échelle fixe [0, 1], bruit seedé par image et par
niveau, 100 images de seg_val, 3 dégradations × 10 niveaux (mêmes plages qu'à l'origine),
IC 95 % par bootstrap sur les images. La question posée est
**Δ = PSNR(propre, reconstruction) − PSNR(propre, entrée dégradée)** : Δ > 0 signifie que
MedVAE rapproche l'image de sa version propre.

**Résultats** (`experiments/robustness/e11_sweep/`, figures `robustness_*.png`) :
MedVAE reconstruit l'image propre à 33.1 dB (vaisseaux : 33.1 dB), ce qui fixe un plafond.

| Dégradation | Δ PSNR image (dB), du plus faible au plus fort niveau | Δ vaisseaux (dB) | Images restaurées |
|---|---|---|---|
| Bruit de Poisson | **+1.2 → +6.6** | +0.9 → +6.8 | 99-100 % |
| JPEG (qualité 95 → 5) | −11.7 → −0.6 | −11.3 → −0.1 | 0-9 % |
| Flou gaussien (noyau 5 → 31) | −2.0 → +0.08 | −2.8 → −0.2 | 3-82 % |

(IC 95 % de largeur < 0.5 dB partout)

**Conclusions**

- **MedVAE débruite réellement le bruit de Poisson**, d'autant plus que le bruit est fort
  (+6.6 dB au niveau le plus bruité) : la conclusion d'origine (« véritable débruiteur »)
  est confirmée, cette fois avec une mesure correcte.
- **Il ne restaure ni le JPEG ni le flou.** Pour le JPEG, son erreur de reconstruction
  propre (plafond à 33 dB) dépasse les artefacts de compression : il dégrade toujours
  l'image. Pour le flou, il ne récupère aucun détail ; la reconstruction suit simplement
  l'entrée floue (Δ ≈ 0 sur l'image, légèrement négatif sur les vaisseaux).
- Le résultat d'origine sur le flou (« les images floues se reconstruisent mieux », r ≈ −0.998)
  venait de la métrique : la courbe orange des figures (reconstruction vs entrée dégradée)
  monte avec le flou, alors que la courbe bleue (vs image propre) descend.

## E13 — Axe B refait : conditionnement FiLM par un score de qualité (2026-09-27)

**Constat sur l'étude d'origine** (notebooks 04 à 07 de `medvae_eval/cvae/`) :
- entraînement et évaluation en **64×64** : à cette résolution, le MedVAE pré-entraîné
  (`vae_4x_4c_2D`) ne reconstruit qu'à **20.7 dB** sur seg_val, contre **34.8 dB** en
  256×256 (runs `e13_film_pretrained_64px` et `e13_film_pretrained`), et les vaisseaux
  font ~1 pixel ;
- loss = MSE moyenne + 1e-6 × KL **sommée** sur le latent : le terme KL n'est pas à la même
  échelle que la MSE (en 256×256, il pèse ~700 fois la MSE et fait remonter l'erreur de
  validation pendant l'entraînement). La « loss ELBO interne » où le cVAE gagnait de 31 à
  45 % mesure donc surtout la KL, pas la reconstruction ;
- le cVAE (2500 pas FiLM seul, puis 5000 pas complets) et la baseline (7500 pas complets)
  ne sont pas entraînés de la même façon ;
- un seul run par modèle, 60 images de test : l'ablation compare deux modèles entraînés
  séparément (vrai c contre c = 0.5), l'écart mêle donc l'usage de c et la variabilité d'un
  entraînement à l'autre, que le test de Wilcoxon sur les images ne couvre pas.

**Nouveau protocole** (`medvae_eval/film/train_film.py`, commits `34b8619` et `da55a67`) :
- 256×256 ; seg_train découpé 900 / 100 (découpage fixe) ; **test = seg_val** (200 images,
  jamais vues ni pour l'entraînement ni pour la sélection) ;
- baseline et FiLM entraînés **à l'identique** : tous les paramètres, 3000 pas, AdamW
  lr 1e-5 cosine, batch 4, loss = MSE + 1e-6 × KL / nombre de pixels (convention
  MedVAE / latent-diffusion) ; sélection sur la MSE de validation ; 3 seeds ;
- FiLM : 19 blocs, têtes initialisées à zéro, +1.70 M paramètres (+3.1 % des 55.3 M de
  `vae_4x_4c_2D`) ;
- **contrôles à l'inférence, sur le même modèle** : c mélangé (permutation fixe des scores
  entre les 200 images) et c constant (moyenne d'entraînement). Si le modèle se sert de c,
  le vrai c doit battre les deux ;
- score c de l'approche C (appris), celle pour laquelle l'étude d'origine concluait que
  c est exploité.

**Runs** : `2026-09-27_*_e13_film_{pretrained,baseline_seed4?,C_seed4?}` ; analyse
`python medvae_eval/film/analyze.py` (commande dans le script) → `experiments/film/e13/`.

| Modèle (test seg_val, 256×256) | PSNR (dB) | PSNR vaisseaux (dB) | SSIM | HaarPSI |
|---|---|---|---|---|
| MedVAE pré-entraîné | 34.81 | 35.57 | 0.949 | 0.939 |
| Baseline fine-tunée (sans c) | 42.261 ± 0.013 | 40.891 ± 0.005 | 0.970 | 0.983 |
| FiLM, vrai c | 42.287 ± 0.018 | 40.915 ± 0.009 | 0.970 | 0.983 |
| FiLM, c mélangé | 42.285 ± 0.017 | 40.914 ± 0.009 | 0.970 | 0.983 |
| FiLM, c constant | 42.300 ± 0.021 | 40.925 ± 0.010 | 0.970 | 0.984 |

| Différence appariée (PSNR, dB) | Moyenne | IC 95 % (images) | Par seed (42 / 43 / 44) |
|---|---|---|---|
| Baseline − pré-entraîné | **+7.45** | [+7.06, +7.85] | +7.44 / +7.46 / +7.44 |
| FiLM (vrai c) − baseline | +0.026 | [+0.015, +0.038] | +0.055 / +0.004 / +0.019 |
| FiLM : vrai c − c mélangé | +0.002 | [−0.007, +0.010] | +0.002 / +0.002 / +0.001 |
| FiLM : vrai c − c constant | −0.013 | [−0.021, −0.006] | −0.015 / −0.015 / −0.009 |

(moyenne ± écart-type sur 3 seeds ; mêmes conclusions sur le PSNR des vaisseaux, le SSIM
et le HaarPSI, cf. `experiments/film/e13/paired.csv`)

**Conclusions**

- **Fine-tuner MedVAE sur ARCADE améliore fortement la reconstruction** : +7.4 dB de PSNR
  (+5.3 dB sur les vaisseaux) par rapport au modèle pré-entraîné, pour les 3 seeds.
- **FiLM ne fait ni mieux ni moins bien que la baseline** : +0.03 dB, positif pour les 3
  seeds mais très faible (< 0.1 dB). Le résultat d'origine (cVAE pire de 21 à 49 % en
  MSE) ne se reproduit pas quand les deux modèles sont entraînés à l'identique.
- **Le modèle ne se sert pas de l'information de qualité** : remplacer le vrai c par celui
  d'une autre image ne change rien (+0.002 dB, IC contenant 0), et un c constant fait même
  très légèrement mieux. Les modulations FiLM apprises dépendent bien de c (|γ(1) − γ(0)|
  jusqu'à 0.05), mais cette dépendance n'aide pas la reconstruction : le petit gain sur la
  baseline vient des paramètres ajoutés (une modulation affine par canal), pas du score.
  La conclusion d'origine « l'approche C exploite c » n'est pas confirmée.

**Vérifications complémentaires (seed 42)** : scores des approches A (pondéré) et B (PCA),
et learning rate 100 fois plus fort pour les couches FiLM (`--film-lr 1e-3`, commit
`8445406`), pour écarter l'idée que c soit ignoré parce que les têtes FiLM apprennent trop
lentement. Runs `*_e13_film_{A,B,C_filmlr1e-3}_seed42` ; analyse `experiments/film/e13_seed42_checks/`.

| Seed 42, PSNR (dB) | FiLM − baseline | vrai c − c mélangé | vrai c − c constant |
|---|---|---|---|
| Score C (référence ci-dessus) | +0.055 | +0.002 [−0.007, +0.010] | −0.015 |
| Score C, lr FiLM 1e-3 | **+0.239** [+0.16, +0.33] | +0.011 [−0.008, +0.031] | −0.001 [−0.018, +0.018] |
| Score A | +0.067 [+0.05, +0.09] | +0.009 [+0.003, +0.016] | +0.006 [+0.002, +0.011] |
| Score B | +0.078 [+0.06, +0.10] | **+0.037** [+0.020, +0.054] | +0.023 [+0.010, +0.036] |

- **Un lr FiLM plus fort améliore FiLM (+0.24 dB), mais pas grâce à c** : le c constant fait
  aussi bien que le vrai. Les couches FiLM servent de paramètres supplémentaires.
- **Avec les scores A et B, le vrai c bat les deux contrôles** (IC hors de 0) : le réseau
  exploite un peu ces scores, construits à partir de métriques de netteté et de contraste
  calculées sur toute l'image (une information globale, alors que ce MedVAE n'a que des
  convolutions). L'effet reste minuscule : ≤ 0.04 dB sur l'image, ≤ 0.012 dB sur les
  vaisseaux, un seul seed.
- Conclusion d'ensemble : **le conditionnement par la qualité n'apporte pas de gain utile à
  la reconstruction**. Le score C (appris) n'est pas utilisé ; A et B le sont très
  légèrement. C'est l'inverse de l'étude d'origine (C exploité, A inversé), qui comparait
  des modèles entraînés séparément sur un seul run.

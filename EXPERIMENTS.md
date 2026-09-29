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
| E05 | D | Validation de E04 (tête U-Net par défaut pour B et C) | — | gardé |
| E06 | D | 3 seeds par condition | variabilité ±0.01 ; D > A pour les 3 seeds | protocole |
| E07 | D | Bruit d'augmentation ramené à l'intensité voulue | A +0.010 (faible) | gardé |
| E08 | D | Validation sur le seg_val officiel (1000 images d'entraînement) | A +0.030 | gardé |
| E09 | D | Dice / IoU sur les artères seules (fond exclu) | ~0.024 sous le Dice avec fond, classement inchangé | ajouté |
| E10 | D | Nouvel état de référence (E07 + E08) sur A, B, C, D, A* | A 0.452, A* 0.448, B 0.417, C 0.427, D 0.455 ; D ≈ A désormais | référence, remplacée par E17 |
| E11 | A | Robustesse mesurée par rapport à l'image propre | réduit le Poisson (+1.2 à +6.6 dB, un filtre 5×5 fait +6.3 à +10.3), ne restaure ni JPEG ni flou | refait |
| E12 | A | Segmentation d'images dégradées (A, A*, D) | le débruitage MedVAE n'aide pas (A* ≤ A) ; E07 a coûté la robustesse de A | mesuré, refait en E17 |
| E13 | B | FiLM refait (256×256, entraînement identique, contrôles sur c) | fine-tuning +7.4 dB ; FiLM ≈ baseline ; score C inutilisable (quasi constant à l'entraînement) | refait |
| E14 | C | JEPA refait (test exclu, probe sur la couche entraînée) | JEPA dégrade la représentation (−0.056) et la reconstruction ; pas de contraction | refait, précisé par E19 |
| E15 | D | Augmentation par dégradations réalistes (Poisson, JPEG, flou) sur A | robustesse rétablie (+0.1 à +0.25 aux fortes dégradations), −0.007 sur images propres | option (non activée par défaut), refait en E17 |
| E16 | D | Témoins de la compression : même tête sur l'image réduite à 128×128 (R) ; U-Net de A sur l'image réduite puis ré-agrandie | R 0.405 ≈ B 0.412 (+0.006, non établi) : le latent ne fait pas mieux qu'une réduction ; en pixels, MedVAE coûte −0.004 contre −0.039 | témoins ajoutés |
| E17 | D | E10, E12 et E15 relancés (augmentations différentes par worker et par epoch, commit figé) | A 0.428, B 0.412, C 0.410, D 0.427 ; C ≈ B (−0.001) ; D ≈ A (0.000) ; augmentation par dégradations : +0.010 sur images propres | remplace E10, E12, E15 |
| E18 | B | FiLM avec les scores A et B sur 3 seeds | le réseau utilise A et B, mais ≤ 0.03 dB ; FiLM − baseline non établi | conclusion B confirmée |
| E19 | C | JEPA sur 3 pré-entraînements (bruit et seeds corrigés, lr du probe choisi sur la validation) | JEPA baisse le probe de son étape 1 (−0.025, 3 seeds sur 3) ; étape 2 ≈ MedVAE (+0.019, non établi) ; rang : pas d'effet net | remplace E14 |

Axes : A robustesse, B conditionnement FiLM, C JEPA, D segmentation (dont les conditions
A, A*, B, C et D, décrites dans le README). Les effets de segmentation sont des Dice sur le test set.

## Protocole

1. **Avant** : le run de référence est celui de l'état courant du projet. S'il n'existe
   pas dans les conditions actuelles, on le lance d'abord.
2. **Modification** : on commite le changement, puis on lance le run (un run lancé avec
   des modifications non commitées est marqué `git_dirty` dans son `meta.json`, et le
   `git diff` est enregistré dans `git_diff.patch`). Depuis l'audit (E16 et suivantes), les
   jobs Slurm sont soumis depuis un worktree figé au commit testé (`~/MEDVAE-X_runs`) : un job
   en file d'attente exécute ainsi exactement ce commit, même si le dépôt évolue entre-temps.
3. **Après** : on compare, on note l'entrée ci-dessous, avec la conclusion. **La décision de
   garder une modification se prend sur la validation (seg_val), pas sur le test** : le test
   ne sert qu'à rapporter le résultat. Jusqu'à E15, les décisions s'appuyaient sur le Dice de
   test (défaut relevé par l'audit du 2026-09-29) ; les conclusions de E03, E04, E07 et E08
   restent valables sur la validation (cf. E17), mais les valeurs de test d'alors ne sont plus
   des estimations tout à fait indépendantes.
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
ou la vérité terrain), fond compris. Dice ≥ IoU toujours. `scripts/summarize_runs.py` affiche aussi le meilleur Dice
de validation (`val_dice`), base des décisions.

**Intervalles de confiance.** Un IC calculé sur les images ne vaut que pour des variantes d'un
même modèle (ex. vrai c contre c mélangé). Pour comparer des modèles entraînés séparément, l'IC
porte sur les seeds (loi de Student sur les écarts par seed) ; avec un seul seed, pas d'IC.

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
49/50), identique à l'original (0.02613 → 0.02408, epoch 49/50). *(0.02617 est la valeur après la
première epoch, pas celle du MedVAE pré-entraîné, qui n'a pas été mesurée.)*

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
- **B < A** (−0.015 en moyenne) et **C < B** pour 2 seeds sur 3 (seed 43 : +0.0004, égalité) : le
  latent MedVAE seul reste un peu en dessous de l'image entière, et fine-tuner MedVAE sur ARCADE
  dégrade légèrement. *(Formulation corrigée après l'audit ; cf. E17 : pas d'effet du fine-tuning.)*

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
qu'elle rétablit l'augmentation voulue à l'origine. *Ajout après E12 : ce gain se paie en
robustesse aux dégradations (Dice 0.34 → 0.02 au bruit de Poisson maximal), cf. E12 et E15.*

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
métriques actuelles ; tous les runs E02 à E06 ont été réévalués (leurs `results.json` ont
été réécrits avec le code du commit `f3c156c`, sans trace dans `meta.json`) (Dice identique à 1e-4
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
- **C > B pour 2 seeds sur 3, égalité au seed 42** (+0.010 en moyenne), l'inverse de E06 :
  fine-tuner MedVAE sur ARCADE aide peut-être un peu la segmentation depuis le latent, mais
  l'écart est de l'ordre de la variabilité. *(Formulation corrigée après l'audit ; E17 ne
  confirme pas cet effet : C − B = −0.001.)*
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
| JPEG (qualité 95 → 5) | −11.7 → −0.5 | −11.3 → −0.1 | 0-9 % |
| Flou gaussien (noyau 5 → 31) | −2.0 → +0.08 | −2.8 → −0.2 | 3-82 % |

(IC 95 % de largeur ≤ 0.53 dB partout)

**Conclusions**

- **MedVAE réduit le bruit de Poisson**, d'autant plus que le bruit est fort (+6.6 dB au
  niveau le plus bruité) ; voir ci-dessous le témoin « filtre simple », qui fait nettement mieux.
- **Il ne restaure ni le JPEG ni le flou.** Pour le JPEG, son erreur de reconstruction
  propre (plafond à 33 dB) dépasse les artefacts de compression : il dégrade toujours
  l'image. Pour le flou, il ne récupère aucun détail ; la reconstruction suit simplement
  l'entrée floue (Δ ≈ 0 sur l'image, légèrement négatif sur les vaisseaux).
- *Ajout après l'audit (2026-09-29) : témoin « filtre simple »* (`medvae_eval/robustness/baselines.py`,
  `experiments/robustness/e11_baselines/`). Sur les mêmes 100 images et le même bruit seedé, un flou
  gaussien 5×5 gagne **+6.3 → +10.3 dB** (vaisseaux +7.1 → +10.9) et un filtre médian 5×5
  **+7.2 → +10.0 dB**, contre +1.2 → +6.6 dB pour MedVAE : MedVAE réduit le bruit de Poisson, mais
  4 à 5 dB de moins qu'un filtre trivial ; « véritable débruiteur » était exagéré. Sur le JPEG le plus
  fort, le médian gagne même un peu (+0.4 dB), MedVAE jamais.
- Le résultat d'origine sur le flou (« les images floues se reconstruisent mieux », r ≈ −0.998)
  venait de la métrique : la courbe orange des figures (reconstruction vs entrée dégradée)
  monte avec le flou, alors que la courbe bleue (vs image propre) descend.

## E12 — Segmentation d'images dégradées : MedVAE rend-il la segmentation plus robuste ? (2026-09-29)

**Question** : E11 montre que MedVAE débruite le bruit de Poisson (+1.2 à +6.6 dB). Est-ce que
cela aide à segmenter des images dégradées ?
**Protocole** (`medvae_eval/robustness/downstream.py`, commits `dccc438` et `91e77bf`) : les 300
images de test sont dégradées (Poisson, JPEG, flou ; mêmes modèles et plages que E11, niveaux
0, 3, 6, 9 sur 10, bruit seedé, mêmes images dégradées pour tous les modèles), puis segmentées
par A, A* (MedVAE puis U-Net de A) et D, pour les 3 seeds de E10. Dice artères.
**Résultats** : `experiments/robustness/e12_downstream/` (`main.md`, `main.png`)

| Dice artères (3 seeds) | A | A* | D |
|---|---|---|---|
| Image propre | 0.429 ± 0.005 | 0.426 ± 0.006 | 0.432 ± 0.004 |
| Poisson, niveaux 0 / 6 / 9 | 0.400 / 0.345 / 0.025 | 0.379 / 0.307 / 0.040 | 0.400 / 0.353 / 0.079 |
| JPEG, niveaux 3 / 6 / 9 | 0.426 / 0.414 / 0.051 | 0.421 / 0.413 / 0.087 | 0.429 / 0.425 / 0.113 |
| Flou, niveaux 3 / 6 / 9 | 0.393 / 0.297 / 0.151 | 0.395 / 0.304 / 0.158 | 0.397 / 0.287 / 0.168 |

- **Le débruitage de MedVAE n'aide pas la segmentation** : sous bruit de Poisson, A* (images
  débruitées par MedVAE) fait *moins bien* que A aux niveaux faibles et moyens (−0.02 à −0.04) ;
  le U-Net de A est plus gêné par les reconstructions d'images bruitées que par le bruit lui-même.
- **D ≈ A** à tous les niveaux faibles et moyens ; aux niveaux extrêmes du Poisson et du JPEG,
  tous s'effondrent (Dice ≤ 0.11), D un peu moins ; le flou le plus fort divise le Dice par deux
  à trois (0.15 à 0.17). *(Formulation corrigée après l'audit : « tous < 0.12 » était faux pour
  le flou.)*
- JPEG : sans effet jusqu'au niveau 6 (qualité ≥ 35), effondrement au niveau 9 (qualité 5).

**Complément : effet du bruit d'augmentation (E07) sur la robustesse.** Même protocole sur
les modèles entraînés avec l'ancien bruit trop fort (A et D de E02/E06) et avec le bruit
corrigé (A de E07), tous avec l'ancien découpage 80/20 (`experiments/robustness/e12_aug_noise/`).

| Dice artères de A (3 seeds) | Bruit fort (E06) | Bruit léger (E07) |
|---|---|---|
| Image propre | 0.400 ± 0.011 | 0.410 ± 0.008 |
| Poisson, niveaux 6 / 9 | 0.379 / **0.340** | 0.341 / **0.021** |
| JPEG, niveau 9 | 0.131 | 0.029 |
| Flou, niveaux 6 / 9 | 0.325 / 0.253 | 0.283 / 0.166 |

- **E07 a un coût caché** : le bruit d'augmentation trop fort, involontaire, rendait A très
  robuste aux dégradations (Dice 0.34 au bruit de Poisson maximal, contre 0.02 avec le bruit
  corrigé). Le gain de +0.010 sur images propres s'est payé en robustesse.
- D'où E15 : une augmentation par dégradations réalistes peut-elle récupérer cette robustesse
  sans perdre le gain sur images propres ?

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
| FiLM (vrai c) − baseline | +0.026 | [+0.015, +0.038] (sur les images, voir note) | +0.055 / +0.004 / +0.019 |
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
- *Notes après l'audit (2026-09-29)* : (1) l'IC « FiLM − baseline » ci-dessus porte sur les
  images, alors qu'il compare des modèles entraînés séparément ; sur les 3 seeds, l'IC 95 %
  (Student) est **[−0.04, +0.09]** : l'écart n'est pas établi. (2) Le score C ne pouvait pas
  être utilisé : sur les 900 images d'entraînement, il vaut 0.92 en moyenne, 79 % au-dessus de
  0.9 (intervalle 5-95 % [0.70, 1.00]), contre 0.73 en moyenne sur le test (26 % sous 0.5). Le
  modèle n'a presque jamais vu de score faible, et sur la plage vue, γ varie d'au plus 0.02.
  D'où E18 : les scores A et B sur 3 seeds.

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

## E14 — Axe C refait : JEPA à l'étape 2, évaluation corrigée (2026-09-29)

**Constats sur l'étude d'origine** (`jepa_adaptation/`, notebook `pretraining_analysis.ipynb`) :
- les images de test d'ARCADE faisaient partie du pré-entraînement (corrigé : 1080 / 120
  images de seg_train + seg_val, test exclu ; commit `dfca730`) ;
- **l'« étape 1 » n'est pas MedVAE** : c'est un autoencodeur de la même famille mais plus
  petit (`ch` 64, `ch_mult` [1, 2, 4, 4] : facteur 8 par côté, latent 64×64×1, 21 M
  paramètres), entraîné **de zéro** sur ARCADE (15 epochs, pertes de reconstruction,
  perceptuelle et adversariale). La comparaison « MedVAE → étape 1 → étape 2 » mélange donc
  architecture, taux de compression (×4 contre ×8 par côté) et données ;
- **les mesures de structure du latent portaient sur une couche jamais entraînée** : la
  sortie `channel_proj(channel_ds(z) + z)`, ignorée par la loss de l'étape 1 et hors du chemin
  de la loss JEPA (qui porte sur les features avant `conv_out`). Le « 90 % de la variance sur
  25 dimensions » décrivait une projection aléatoire du latent ;
- le linear probe prédisait la classe d'artère dominante de chaque image, sur 60 images.

**Nouvelle évaluation** (commits `dfca730`, `a1cf08b`, `8f4086f`), sur les 300 images de test,
avec les modèles réentraînés sans le test : reconstruction (`eval/reconstruction.py`), linear
probe par pixel vaisseau / fond sur la couche entraînée par JEPA (`eval/probe.py` : features
standardisées, convolution 1×1, entraînement sur seg_train, sélection sur seg_val, 3 seeds),
dimensionnalité de cette couche (`eval/feature_rank.py`). Résultats : `experiments/jepa/`.

| | MedVAE (×4, pré-entraîné) | Étape 1 (×8, de zéro) | Étape 2 JEPA |
|---|---|---|---|
| Reconstruction : PSNR / vaisseaux (dB) | 34.8 / 35.5 | 30.2 / 30.9 | **17.8** / 24.5 |
| Linear probe vaisseaux, Dice (lr 1e-2, 100 epochs) | 0.517 ± 0.005 | **0.541 ± 0.005** | 0.485 ± 0.006 |
| Rang effectif des features (canaux) | 251 (512) | 111 (256) | 139 (256) |
| Composantes pour 90 % de la variance | 61 | 22 | 34 |

- **Le probe doit converger** : à 20 epochs (lr 1e-3), les scores étaient 0.39 / 0.43 / 0.39 et
  le meilleur epoch était le dernier *(pour MedVAE et l'étape 1 ; pour l'étape 2 : 16, 18, 16)* ; à 60 epochs 0.45 / 0.50 / 0.45. *(Correction, cf. E19 :
  j'avais écrit « le classement est le même dans les trois réglages » ; c'est faux, l'étape 2
  égale MedVAE à 20 et 60 epochs.)* Le probe sur les 26 segments (Dice ~0.03) et sur le latent à
  1 canal (~0.03) ne distinguent rien : un classifieur linéaire par pixel ne peut pas nommer un
  segment, ni séparer les vaisseaux avec un seul canal.
- **JEPA dégrade la représentation qu'il part améliorer** : −0.056 de Dice par rapport à
  l'étape 1, et en dessous du MedVAE d'origine *(sur un seul pré-entraînement ; E19, sur 3,
  confirme la baisse par rapport à l'étape 1 mais pas l'écart à MedVAE)* ; la reconstruction s'effondre (décodeur figé).
- **Pas de contraction dimensionnelle** : sur la couche entraînée, JEPA *augmente* le rang
  effectif (111 → 139) et le nombre de composantes (22 → 34), ce qui est l'effet attendu de la
  régularisation SIGReg *(hausse non reproductible sur 3 pré-entraînements, cf. E19)*. La conclusion d'origine (« latent plus compact et plus discriminant »)
  n'est pas confirmée.

## E15 — Augmentation par dégradations réalistes (2026-09-29)

**Hypothèse** : E12 montre que A, entraîné avec le bruit d'augmentation corrigé (E07),
s'effondre sur les images fortement dégradées. Montrer au modèle des dégradations d'acquisition
pendant l'entraînement devrait rétablir cette robustesse.
**Modification** : option `data.degradation_aug_p` (commit `91e77bf`) : avec probabilité 0.3,
une dégradation parmi bruit de Poisson (échelle 0.05-1), JPEG (qualité 5-95) et flou gaussien
(noyau 3-31). Testée sur A, rien d'autre ne change (config E10, 3 seeds,
`--set data.degradation_aug_p=0.3`). **Attention** : ce sont les mêmes familles et plages que
l'évaluation de robustesse (E12), qui est donc « dans la distribution » de l'augmentation.
**Runs** : avant = A de E10 — après `2026-09-29_*_e15_degradation_aug_seed4{2,3,4}_condition_a` ;
robustesse `experiments/robustness/e15_downstream/` (`comparison.md`, `comparison.png`)

| Dice artères (3 seeds) | A (E10) | A + dégradations (E15) |
|---|---|---|
| Image propre | 0.430 ± 0.005 | 0.423 ± 0.011 |
| Poisson, niveaux 6 / 9 | 0.345 / 0.025 | **0.400 / 0.266** |
| JPEG, niveau 9 | 0.051 | **0.289** |
| Flou, niveaux 6 / 9 | 0.297 / 0.151 | **0.403 / 0.373** |

Écart apparié sur images propres, par seed : +0.002, −0.012, −0.011 (−0.007 en moyenne ;
IoU artères −0.002).

**Conclusion** : la robustesse est largement rétablie (gains de 0.1 à 0.25 de Dice aux
dégradations fortes), pour un coût sur images propres de l'ordre de la variabilité entre seeds
(2 seeds sur 3 en baisse). Option conservée mais **non activée par défaut** : les résultats
officiels restent ceux de E10 (images propres), et l'activer demanderait de réentraîner B, C
et D pour garder des conditions comparables. À décider selon l'usage visé (robustesse ou score
sur images propres).

## E16 — Témoins de la compression pour la segmentation (2026-09-29)

**Pourquoi** (audit, MAJ-12) : la conclusion « la compression préserve l'information nécessaire à
la segmentation » n'avait pas de témoin. B et C (latent 128×128×1, 16 fois moins de valeurs que
l'image) n'étaient comparés qu'à A (image entière, U-Net trois fois plus gros) ; A* n'était
comparé à aucune compression triviale.
**Témoins** :
- **R** (commit `4b6cf64`, `finetune/configs/condition_r.yaml`) : la tête de B, même architecture et
  même entraînement, appliquée à l'image moyennée sur des blocs 4×4 (128×128×1, autant de valeurs
  que le latent). 3 seeds, depuis le worktree figé ;
- **A↓↑** (commit `04a8e11`, `finetune/eval_resampled.py`) : le U-Net de A, sans réentraînement,
  appliqué à l'image réduite à 128×128 puis ré-agrandie à 512×512 (bilinéaire ou bicubique). C'est
  la compression triviale qui correspond à A*.
**Runs** : `2026-09-29_*_e16_seed4?_condition_r` ; `experiments/diagnostics/e16_resampled.csv`
(modèles A de E17).

| Dice artères (test, 3 seeds) | Moyenne | Écart apparié (IC 95 % sur les seeds) | Par seed |
|---|---|---|---|
| B — latent MedVAE + tête | 0.412 ± 0.005 | | |
| C — latent fine-tuné + tête | 0.410 ± 0.009 | | |
| **R — image réduite à 128×128 + même tête** | 0.405 ± 0.002 | B − R = +0.006 [−0.002, +0.014] | +0.009 / +0.003 / +0.007 |
| | | (validation : B − R = +0.005 [−0.020, +0.030]) | +0.016 / −0.002 / +0.001 |
| A* — U-Net A sur la reconstruction MedVAE | 0.424 ± 0.008 | A* − A = −0.004 [−0.009, +0.002] | −0.003 / −0.002 / −0.006 |
| **A↓↑ — U-Net A sur l'image réduite puis ré-agrandie** (bilinéaire) | 0.389 ± 0.018 | A↓↑ − A = −0.039 [−0.070, −0.008] | −0.049 / −0.042 / −0.025 |

(bicubique : 0.390 ± 0.022 ; A* − A↓↑ = +0.035 [−0.001, +0.071] en Dice, +0.030 [+0.003, +0.058]
en IoU)

**Conclusions**

- **Utilisé comme entrée d'une tête entraînée, le latent MedVAE ne fait guère mieux qu'une
  réduction triviale** : avec la même tête, l'image moyennée à 128×128 atteint presque le Dice du
  latent (B − R = +0.006, en faveur de B pour les 3 seeds sur le test mais IC contenant 0, et pas
  sur la validation). Que B et C atteignent 96 % du Dice de A montre surtout qu'une résolution de
  128×128 suffit à cette tâche avec une tête adaptée ; le latent préserve l'information utile à la
  segmentation autant qu'une réduction ×4, pas nettement plus.
- **Rendu en pixels, MedVAE préserve bien mieux qu'une réduction triviale** : sans réentraînement,
  le U-Net de A perd 0.004 sur les reconstructions MedVAE, mais 0.039 sur l'image réduite puis
  ré-agrandie (3 seeds sur 3), où les vaisseaux fins sont effacés. La reconstruction MedVAE
  restitue à partir de 128×128 valeurs des détails qu'une interpolation ne restitue pas.
- Les deux témoins répondent à deux questions différentes : R est entraîné pour son entrée
  (compression puis apprentissage), A↓↑ ne l'est pas (compression seule, comme A*).

## E17 — E10, E12 et E15 relancés avec les augmentations corrigées (2026-09-29)

**Pourquoi** : l'audit a trouvé qu'avec albumentations 2, chaque worker du DataLoader recevait
une copie du même générateur aléatoire, sans nouvelle graine : les workers tiraient les mêmes
augmentations, et la même suite revenait à chaque epoch. Ce défaut touchait toutes les
expériences de segmentation (E02 à E15). Par ailleurs, les résultats officiels B et D et
plusieurs chiffres publiés venaient de runs `git_dirty`, et les jobs en file exécutaient le code
présent à leur démarrage.
**Modification** : graine des augmentations par worker (`seed_worker_augmentations`, commit
`4b6cf64`, testée dans `tests/test_dataset.py`) ; tous les runs lancés depuis le worktree figé à ce
commit (tous propres).
**Protocole** : config E10 inchangée, 3 seeds par condition ; A* = U-Net de A appliqué aux
reconstructions (3 seeds via la robustesse, seed 42 via `eval_astar`) ; robustesse (protocole
E12) et augmentation par dégradations (E15, p = 0.3) refaites sur ces runs. Décisions sur la
validation.
**Runs** : `2026-09-29_*_e17_seed4?_condition_{a,b,c,d}`, `*_e17_seed42_condition_astar`,
`*_e17_degradation_aug_seed4?_condition_a` ; robustesse `experiments/robustness/e17_downstream/`
(`report_figure.md`, comparaison avec E12 et E15 dans `vs_e12_e15.md`).

| Condition (3 seeds) | Dice de validation | Dice artères (test) | IoU artères (test) | E10, Dice artères |
|---|---|---|---|---|
| A — U-Net | 0.481 ± 0.010 | 0.428 ± 0.008 | 0.309 ± 0.006 | 0.430 ± 0.005 |
| A* — U-Net A sur reconstructions | — | 0.424 ± 0.008 | 0.305 ± 0.006 | 0.425 (seed 42) |
| B — latent MedVAE | 0.473 ± 0.009 | 0.412 ± 0.005 | 0.292 ± 0.004 | 0.393 ± 0.008 |
| C — latent fine-tuné | 0.473 ± 0.006 | 0.410 ± 0.009 | 0.291 ± 0.008 | 0.403 ± 0.002 |
| D — MedVAE → U-Net | 0.488 ± 0.010 | 0.427 ± 0.009 | 0.309 ± 0.005 | 0.432 ± 0.004 |
| A + dégradations (p = 0.3) | 0.489 ± 0.011 | 0.437 ± 0.010 | 0.316 ± 0.008 | 0.423 ± 0.011 (E15) |

| Écart apparié (Dice artères, test) | Moyenne | IC 95 % (seeds) | Par seed | Validation |
|---|---|---|---|---|
| D − A | +0.000 | [−0.041, +0.040] | +0.019 / −0.012 / −0.007 | +0.007 |
| C − B | −0.001 | [−0.036, +0.034] | −0.013 / +0.015 / −0.005 | −0.000 |
| B − A | −0.016 | [−0.049, +0.017] | −0.002 / −0.029 / −0.018 | −0.008 |
| C − A | −0.017 | [−0.028, −0.006] | −0.015 / −0.014 / −0.022 | −0.009 |
| A + dégradations − A | +0.010 | [−0.023, +0.042] | +0.023 / +0.008 / −0.003 | +0.008 (+0.011 / +0.008 / +0.004) |

| Dice artères sur images dégradées | A | A* | D | A + dégradations |
|---|---|---|---|---|
| Image propre | 0.428 | 0.424 | 0.427 | 0.437 |
| Poisson, niveaux 0 / 3 / 6 / 9 | 0.405 / 0.390 / 0.351 / 0.023 | 0.388 / 0.372 / 0.317 / 0.039 | 0.395 / 0.379 / 0.343 / 0.080 | 0.417 / 0.409 / 0.390 / 0.233 |
| JPEG, niveaux 6 / 9 | 0.415 / 0.032 | 0.417 / 0.073 | 0.420 / 0.137 | 0.432 / 0.251 |
| Flou, niveaux 3 / 6 / 9 | 0.390 / 0.314 / 0.206 | 0.394 / 0.321 / 0.210 | 0.393 / 0.288 / 0.181 | 0.423 / 0.398 / 0.348 |

**Conclusions**

- A, C et D reproduisent E10 à ±0.005. B gagne 0.018 (0.393 → 0.412) ; seules la graine des
  augmentations et l'exécution depuis un commit figé ont changé, sans qu'on isole la cause.
- **D ≈ A confirmé** (+0.000) : entraîner sur des reconstructions n'apporte rien de mesurable.
- **Fine-tuner MedVAE sur ARCADE n'a pas d'effet sur la segmentation depuis le latent** (C − B =
  −0.001, signes −/+/− ; même chose sur la validation). La formulation « aide légèrement » de E10
  et du rapport est retirée.
- B et C atteignent 96 % du Dice de A (0.412 et 0.410 contre 0.428), mais le témoin R de E16 fait
  presque aussi bien.
- **A* ≈ A sur 3 seeds** (−0.004, légèrement en dessous pour les 3).
- **Robustesse : les conclusions de E12 tiennent.** Sous bruit de Poisson, A* fait moins bien que
  A aux niveaux 0 à 6 (−0.017, −0.018, −0.034) ; D ≈ A jusqu'au niveau 6 ; au niveau 9, Poisson et
  JPEG font s'effondrer les trois modèles (Dice ≤ 0.14, D un peu moins), le flou divise le Dice
  par deux (0.18 à 0.21).
- **L'augmentation par dégradations ne coûte plus rien sur images propres** : +0.010 sur le test
  (2 seeds sur 3), +0.008 sur la validation (3 seeds sur 3), contre −0.007 en E15. Gains aux
  dégradations : +0.21 (Poisson), +0.22 (JPEG) et +0.14 (flou) au niveau 9, +0.02 à +0.08 au
  niveau 6. D'après le protocole (décision sur la validation), elle pourrait être activée par
  défaut ; cela demanderait de réentraîner B, C et D pour garder des conditions comparables. Elle
  reste une option, non activée par défaut : résultats officiels = E17 sans cette augmentation.
- `downstream.py` associait les runs A et D dans l'ordre des dossiers (A seed 42 avec D seed 44,
  etc.). Les images dégradées ne dépendent pas de cet appariement : les moyennes sont justes ;
  les écarts D − A ci-dessus sont appariés par seed à partir des runs. Corrigé (commit `04a8e11`).
- Résultats promus : seed 42 de A, A*, B, C et D de E17.

## E18 — FiLM avec les scores A et B sur 3 seeds (2026-09-29)

**Pourquoi** : l'audit relève que E13 étudie à 3 seeds le seul score inexploitable (C, quasi
constant à l'entraînement) et que les contrôles A et B n'avaient qu'un seed. Les scores A et B
ont des distributions semblables à l'entraînement (moyennes 0.45 et 0.49) et au test (0.44 et
0.46).
**Protocole** : identique à E13 (`train_film.py`, 256×256, 3000 pas, test = seg_val), baseline,
FiLM-A et FiLM-B pour les seeds 42, 43 et 44, tous au commit `4b6cf64` depuis le worktree figé
(runs `2026-09-29_*_e18_film_{baseline,A,B}_seed4?`, tous propres). Analyse
`experiments/film/e18/` (IC sur les images pour les contrôles d'un même modèle, sur les seeds
pour FiLM − baseline).

| PSNR test (dB) | Baseline | FiLM-A | FiLM-B |
|---|---|---|---|
| Vrai c | 42.258 ± 0.004 | 42.296 ± 0.020 | 42.307 ± 0.021 |
| c mélangé | — | 42.289 | 42.274 |
| c constant | — | 42.291 | 42.288 |

| Écart (PSNR, dB) | Moyenne | IC 95 % | Par seed |
|---|---|---|---|
| FiLM-A : vrai c − c mélangé | +0.007 | [+0.002, +0.014] (images) | +0.009 / +0.006 / +0.007 |
| FiLM-A : vrai c − c constant | +0.005 | [+0.002, +0.009] (images) | +0.006 / +0.005 / +0.005 |
| FiLM-B : vrai c − c mélangé | **+0.033** | [+0.018, +0.049] (images) | +0.037 / +0.031 / +0.032 |
| FiLM-B : vrai c − c constant | +0.020 | [+0.008, +0.031] (images) | +0.023 / +0.018 / +0.019 |
| FiLM-A − baseline | +0.038 | [−0.011, +0.087] (seeds) | +0.061 / +0.029 / +0.025 |
| FiLM-B − baseline | +0.050 | [−0.003, +0.102] (seeds) | +0.074 / +0.039 / +0.036 |

(sur les vaisseaux : FiLM − baseline +0.031 [+0.016, +0.047] avec A, +0.038 [+0.018, +0.057]
avec B)

**Conclusions**

- **Avec les scores A et B, le réseau utilise l'information de qualité**, de façon
  reproductible (vrai c > contrôles pour les 3 seeds), mais l'effet est minuscule : 0.03 dB au
  plus (score B).
- FiLM gagne +0.04 à +0.05 dB sur la baseline, dont au plus 0.02-0.03 dB viennent du score ; le
  reste vient des paramètres ajoutés. L'écart FiLM − baseline n'est pas établi sur l'image
  entière (IC sur les seeds contenant 0), il l'est sur les vaisseaux.
- Conclusion de l'axe B : conditionner MedVAE sur un score de qualité calculé à partir de
  l'image n'apporte pas de gain utile. Le score C n'est pas utilisé (il ne varie presque pas à
  l'entraînement) ; A et B le sont, à peine.
- La baseline de E18 reproduit celle de E13 (42.258 contre 42.261 dB). Comme en E13, le
  meilleur pas est le dernier (3000) pour 8 runs sur 9 : les conclusions valent pour ce budget.

## E19 — JEPA sur 3 pré-entraînements (2026-09-29)

**Pourquoi** : les conclusions de E14 reposaient sur un seul pré-entraînement par étape, et
l'audit relève deux erreurs de ma part dans leur rédaction : le classement du probe n'était
**pas** le même dans les trois réglages (à 20 et 60 epochs, l'étape 2 égalait ou dépassait
MedVAE : 0.393 contre 0.391 à 20 epochs, 0.454 contre 0.452 à 60), et le réglage du probe avait été choisi en
regardant le test. S'y ajoutaient deux défauts du pipeline : le bruit gaussien d'augmentation
de `jepa_adaptation` utilisait la valeur par défaut d'albumentations 2 (écart-type 0.2-0.44 sur
[0, 1], bien plus fort que voulu), et, comme pour la segmentation, les augmentations se
répétaient d'un worker et d'un epoch à l'autre ; enfin la validation de l'étape 2 (qui choisit
`best.pt`) tirait des masques aléatoires différents à chaque appel.
**Modification** (commit `4b6cf64`) : bruit ramené à l'écart-type voulu (0.012-0.028), graine
des augmentations par worker, masques de validation tirés avec un générateur fixe, options
`--seed` et `--run-name` (runs traçables dans `experiments/runs/`).
**Protocole** : étape 1 puis étape 2 (15 epochs chacune, comme E14) pour les seeds 42, 43 et 44,
depuis le worktree figé ; puis, pour chaque modèle, linear probe vaisseaux / fond (100 epochs,
lr 1e-3 et 1e-2, 3 seeds de probe), reconstruction et rang effectif sur les 300 images de test.
**Le lr du probe est choisi sur la validation** (seg_val) : 1e-2 pour les trois modèles.
MedVAE, pré-entraîné et fixe, n'a qu'un modèle (3 seeds de probe).
**Runs** : `experiments/runs/2026-09-29_*_e19_jepa_stage{1,2}_seed4?` (tous propres) ;
résultats `experiments/jepa/e19/`.

| Test (moyenne ± écart-type sur 3 pré-entraînements) | MedVAE (×4, pré-entraîné) | Étape 1 (×8, de zéro) | Étape 2 JEPA |
|---|---|---|---|
| Linear probe vaisseaux, Dice (lr 1e-2) | 0.517 ± 0.005 (seeds de probe) | **0.561 ± 0.006** | 0.536 ± 0.011 |
| IoU vaisseaux | 0.349 | **0.390** | 0.366 |
| Dice de validation du probe | 0.570 | **0.604** | 0.569 |
| Reconstruction : PSNR / vaisseaux (dB) | 34.8 / 35.5 | 31.1 / 31.7 | **18.6** / 22.9 |
| Rang effectif des features (canaux) | 251 (512) | 147 (152 / 153 / 136) | 156 (157 / 150 / 161) |
| Composantes pour 90 % de la variance | 61 | 46 (51 / 53 / 35) | 50 (50 / 44 / 55) |

| Écart apparié par pré-entraînement (Dice du probe, test) | Moyenne | IC 95 % (seeds) | Par seed |
|---|---|---|---|
| Étape 2 − étape 1 | **−0.025** | [−0.045, −0.005] | −0.019 / −0.022 / −0.034 |
| Étape 1 − MedVAE | +0.044 | [+0.029, +0.058] | +0.039 / +0.050 / +0.041 |
| Étape 2 − MedVAE | +0.019 | [−0.008, +0.045] | +0.021 / +0.028 / +0.007 |

(avec lr 1e-3 : étape 2 − étape 1 = −0.030, 3 seeds sur 3 ; même sens sur la validation :
−0.035 [−0.056, −0.014])

**Conclusions**

- **JEPA baisse la qualité de la représentation qu'il part améliorer** : −0.025 de Dice du probe
  par rapport à l'étape 1 dont il part, pour les 3 pré-entraînements, sur le test comme sur la
  validation. Le sens de E14 est confirmé, l'ampleur est deux fois plus faible (−0.056 en E14).
- **L'étape 2 n'est pas en dessous de MedVAE** (+0.019, IC contenant 0) : la phrase de E14
  « en dessous du MedVAE d'origine » est retirée.
- L'étape 1 dépasse MedVAE (+0.044), mais les deux modèles diffèrent par l'architecture
  (256 contre 512 canaux), le taux de compression (×8 contre ×4 par côté) et les données
  (ARCADE contre radiographies et mammographies) : cet écart ne dit rien de JEPA et on ne
  l'attribue à aucun de ces facteurs.
- **Rang effectif : pas d'effet net de JEPA** (+9, IC [−27, +45], 2 seeds sur 3 en hausse). La
  hausse 111 → 139 de E14 ne se retrouve pas de façon reproductible ; ce qui tient, c'est
  l'absence de contraction dimensionnelle.
- La reconstruction s'effondre à l'étape 2 (31.1 → 18.6 dB) pour les 3 seeds : le décodeur,
  figé, ne suit pas l'encodeur modifié par JEPA.
- Les niveaux ont changé depuis E14 (étape 1 : 0.541 → 0.561, rang 111 → 147 ; étape 2 :
  0.485 → 0.536). Entre les deux, le bruit d'augmentation, la graine des workers, la validation
  de l'étape 2 et les seeds ont changé en même temps : on n'attribue pas ces écarts à l'une de
  ces causes. E19 remplace E14 pour les chiffres de l'axe C.

# Journal d'expériences

Toutes les modifications testées sur le projet, **qu'elles aient amélioré les résultats
ou non**. Chaque modification est comparée à l'état du projet **juste avant** elle,
mesuré dans les mêmes conditions (même GPU, même dataset, même seed).

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

**Conclusion** : Dice ×4 pour B et C, qui arrivent à 94 % du Dice de A (0.409 contre
0.436) en ne partant que du latent MedVAE compressé ×16.

- **La conclusion d'origine du projet est renversée** : le latent MedVAE *est* exploitable
  pour la segmentation dense des coronaires ; l'échec venait de l'entraînement (E03) et
  de la tête (E04), pas du latent.
- **C ≈ B** encore une fois (0.398 contre 0.409) : fine-tuner MedVAE sur ARCADE n'aide pas.
- **Encore du sous-apprentissage** (train ≈ val ≈ 0.45, contre 0.65 / 0.51 pour A) et
  les courbes plafonnent quand le learning rate atteint son minimum (100 epochs, meilleur
  score aux epochs 90-97) : une tête plus large ou un entraînement plus long pourraient
  encore réduire l'écart avec A.

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

**Modification** : `MeanIoU` remplacé par `MulticlassJaccardIndex(average="macro")`,
calculé exactement comme le Dice. Test de non-régression : `tests/test_seg_metrics.py`
(échoue sur l'ancien code, passe sur le nouveau).

**Impact** : le Dice est inchangé (son calcul était correct). **Tous les IoU publiés
jusqu'ici sont faux** ; les checkpoints d'origine n'étant pas disponibles, ils ne peuvent
pas être recalculés : les IoU corrects viendront de E02.

**Conclusion** : gardé (correction de bug).

## E02 — Reproduction de l'état actuel sur RTX 3090 (en cours)

**Objectif** : obtenir le point de départ mesuré dans nos conditions. Les résultats
d'origine ont été obtenus sur H100/A100, avec d'autres versions des librairies et une
copie d'ARCADE dont la provenance exacte n'est pas connue ; nous utilisons des RTX 3090,
torch 2.14 et ARCADE depuis Zenodo.

**Résultats d'origine** (Dice correct, IoU faux, cf. E01) :

| Condition | Dice | IoU (faux) |
|---|---|---|
| A — U-Net | 0.433 | 0.499 |
| A* — U-Net A sur images reconstruites | 0.435 | 0.497 |
| B — latent MedVAE + tête | 0.038 | 0.005 |
| C — latent MedVAE fine-tuné + tête | 0.039 | 0.000 |
| D — MedVAE → U-Net | 0.451 | 0.495 |

**Adaptations à la 3090 (24 Go), sans effet sur les calculs** : le fine-tuning MedVAE
traite chaque batch de 4 en 4 micro-batchs de 1 avec accumulation de gradient, et la
condition D passe les images dans le MedVAE gelé par paquets de 2. Dans les deux cas le
résultat est mathématiquement identique (loss L1 moyenne, normalisations par image).

**Observation à vérifier** : un Dice de 0.038 correspond à un modèle qui prédit « fond »
partout (≈ 1/26) ; B et C semblent ne rien avoir appris.

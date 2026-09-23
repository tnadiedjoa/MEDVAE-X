#!/bin/bash
# Télécharge le dataset ARCADE (Zenodo, CC0) et le range dans l'arborescence
# attendue par le code (celle du challenge d'origine) :
#
#   $ARCADE_ROOT/
#   ├── dataset_phase_1/
#   │   ├── segmentation_dataset/{seg_train,seg_val}/{images,annotations/seg_*.json}
#   │   └── stenosis_dataset/{sten_train,sten_val}/{images,annotations/sten_*.json}
#   └── dataset_final_phase/
#       ├── test_case_segmentation/{images,annotations/instances_default.json}
#       └── test_case_stenosis/{images,annotations/instances_default.json}
#
# Usage (depuis la racine du repo) :
#   bash scripts/download_arcade.sh
#   ARCADE_ROOT=/autre/chemin bash scripts/download_arcade.sh
#   ARCADE_ZIP=/chemin/arcade.zip bash scripts/download_arcade.sh   # zip déjà téléchargé

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCADE_ROOT="${ARCADE_ROOT:-$REPO_ROOT/data/arcade}"
ZENODO_URL="https://zenodo.org/api/records/10390295/files/arcade.zip/content"
ZIP_MD5="c5b1973ade06f7dff210f878161e1a76"

if [ -d "$ARCADE_ROOT/dataset_phase_1" ]; then
    echo "ARCADE déjà présent dans $ARCADE_ROOT — rien à faire."
    exit 0
fi

mkdir -p "$ARCADE_ROOT"
TMP_DIR="$(mktemp -d "$ARCADE_ROOT/.tmp_download.XXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

ZIP="${ARCADE_ZIP:-$TMP_DIR/arcade.zip}"
if [ ! -f "$ZIP" ]; then
    echo "Téléchargement d'ARCADE (~450 Mo) depuis Zenodo..."
    curl -L --fail --progress-bar -o "$ZIP" "$ZENODO_URL"
fi

echo "Vérification du checksum..."
echo "$ZIP_MD5  $ZIP" | md5sum --check --quiet

echo "Extraction..."
unzip -q "$ZIP" -d "$TMP_DIR"
SRC="$TMP_DIR/arcade"

# move_split <source zenodo> <destination> <nom json zenodo> <nom json attendu>
move_split() {
    mkdir -p "$ARCADE_ROOT/$2/annotations"
    mv "$SRC/$1/images" "$ARCADE_ROOT/$2/images"
    mv "$SRC/$1/annotations/$3" "$ARCADE_ROOT/$2/annotations/$4"
}

move_split syntax/train   dataset_phase_1/segmentation_dataset/seg_train  train.json seg_train.json
move_split syntax/val     dataset_phase_1/segmentation_dataset/seg_val    val.json   seg_val.json
move_split syntax/test    dataset_final_phase/test_case_segmentation      test.json  instances_default.json
move_split stenosis/train dataset_phase_1/stenosis_dataset/sten_train     train.json sten_train.json
move_split stenosis/val   dataset_phase_1/stenosis_dataset/sten_val       val.json   sten_val.json
move_split stenosis/test  dataset_final_phase/test_case_stenosis          test.json  instances_default.json

echo "ARCADE prêt dans $ARCADE_ROOT"

#!/usr/bin/env bash
set -euo pipefail

URL="https://api.insee.fr/melodi/file/DS_BTS_SAL_EQTP_SEX_PCS/DS_BTS_SAL_EQTP_SEX_PCS_2024_CSV_FR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/raw"
ZIP_PATH="$DEST_DIR/DS_BTS_SAL_EQTP_SEX_PCS_2024_CSV_FR.zip"

mkdir -p "$DEST_DIR"
echo "Téléchargement du dataset Insee (salaires EQTP, secteur privé)..."
curl -sL -o "$ZIP_PATH" "$URL"
unzip -o "$ZIP_PATH" -d "$DEST_DIR"
echo "OK : fichiers extraits dans $DEST_DIR"

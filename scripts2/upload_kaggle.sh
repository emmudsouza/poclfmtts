#!/usr/bin/env bash
# Upload a latent cache to Kaggle as a dataset (zips first for a fast single-file upload).
# Prereq: ~/.kaggle/kaggle.json present (Kaggle -> Settings -> API -> Create New Token).
# Usage: scripts2/upload_kaggle.sh data/distill_cache pocketlfm-distill-cache "PocketLFM distill cache (azelma)"
set -euo pipefail

CACHE_DIR="${1:?usage: upload_kaggle.sh <cache_dir> <slug> [title]}"
SLUG="${2:?provide a dataset slug, e.g. pocketlfm-distill-cache}"
TITLE="${3:-$SLUG}"

TOKEN="$HOME/.kaggle/kaggle.json"
[ -f "$TOKEN" ] || { echo "ERROR: $TOKEN missing. Create it: Kaggle -> Settings -> API -> Create New Token."; exit 1; }
USER="$(python -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")"

STAGE="$(mktemp -d)"
NAME="$(basename "$CACHE_DIR")"
echo "zipping $CACHE_DIR ..."
( cd "$(dirname "$CACHE_DIR")" && zip -r -q "$STAGE/$NAME.zip" "$NAME" )
cat > "$STAGE/dataset-metadata.json" <<EOF
{"title": "$TITLE", "id": "$USER/$SLUG", "licenses": [{"name": "CC0-1.0"}]}
EOF

echo "uploading to kaggle as $USER/$SLUG ..."
kaggle datasets create -p "$STAGE"
echo "done -> https://www.kaggle.com/datasets/$USER/$SLUG"
echo "(to update later: kaggle datasets version -p \"$STAGE\" -m 'update')"
rm -rf "$STAGE"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_HTML="$ROOT_DIR/outputs/maps/county_index_map.html"
DEPLOY_DIR="$ROOT_DIR/vercel_deploy"
TARGET_HTML="$DEPLOY_DIR/index.html"

if [[ ! -f "$SOURCE_HTML" ]]; then
  echo "Map file not found: $SOURCE_HTML" >&2
  echo "Run: .venv/bin/python build_food_insecurity_map.py" >&2
  exit 1
fi

mkdir -p "$DEPLOY_DIR"
cp "$SOURCE_HTML" "$TARGET_HTML"

printf 'Synced map to %s\n' "$TARGET_HTML"

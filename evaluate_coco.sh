#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python}"
: "${COCO_ROOT:?Set COCO_ROOT to the COCO2017 directory}"
RUN_DIR="${RUN_DIR:-results/coco_$(date +%Y%m%d_%H%M%S)}"
printf 'Output: %s\n' "$RUN_DIR"
for model in s m; do
  "$PYTHON" -u evaluate.py --model "$model" \
    --images "$COCO_ROOT/val2017" \
    --annotations "$COCO_ROOT/annotations/person_keypoints_val2017.json" \
    --output "$RUN_DIR/$model"
done
printf 'Both full evaluations completed: %s\n' "$RUN_DIR"

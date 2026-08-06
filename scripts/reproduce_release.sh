#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VOICE_CODE_BENCH_VENV:-"$ROOT/.venv"}"
PYTHON_BIN="$VENV_DIR/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  python3 -m venv "$VENV_DIR"
fi

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -e "${ROOT}[figures]"

"$PYTHON_BIN" -c "from pathlib import Path; from voice_code_bench.io import read_metadata; print(f'metadata rows: {len(read_metadata(Path(\"$ROOT/data/metadata.jsonl\")))}')"

mkdir -p "$ROOT/baselines/scores/entities" "$ROOT/baselines/scores/wer"

"$VENV_DIR/bin/vcb-score-entities" \
  --dataset-root "$ROOT" \
  --metadata "$ROOT/data/metadata.jsonl" \
  --entity-matches-dir "$ROOT/baselines/predictions" \
  --output-dir "$ROOT/baselines" \
  --scores-dir "$ROOT/baselines/scores/entities" \
  --output-csv "$ROOT/baselines/results.csv"

"$VENV_DIR/bin/vcb-score-wer" \
  --dataset-root "$ROOT" \
  --metadata "$ROOT/data/metadata.jsonl" \
  --transcripts-dir "$ROOT/baselines/predictions" \
  --output-dir "$ROOT/baselines" \
  --scores-dir "$ROOT/baselines/scores/wer" \
  --output-csv "$ROOT/baselines/results.csv"

"$VENV_DIR/bin/vcb-make-paper-figures" --dataset-root "$ROOT" --model-set paper

echo "Reproduced baselines/results.csv and paper/figures/wer_entity_scatter.pdf"

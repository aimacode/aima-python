#!/usr/bin/env bash
# Build the JupyterLite proof-of-concept site (see README.md).
#
# Usage: ./build.sh [OUTPUT_DIR]
#   OUTPUT_DIR defaults to lite/_output. CI passes docs/_build/html/lite so the
#   companion is published under the GitHub Pages site at /lite/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUTPUT_DIR="${1:-$HERE/_output}"
# resolve to an absolute path (jupyter lite build runs from $HERE)
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

# 1. Build an aima wheel from the repo root and drop it next to the lite config
#    so piplite can install it offline (deps=False, since TF/keras/cv2/cvxopt
#    are not available in Pyodide).
rm -f "$HERE"/aima-*.whl
python -m build --wheel --outdir "$HERE" "$ROOT"

# 2. Build the static JupyterLite site.
cd "$HERE"
jupyter lite build --output-dir "$OUTPUT_DIR"

echo
echo "Built JupyterLite site at: $OUTPUT_DIR"
echo "Serve it locally with:"
echo "    python -m http.server -d \"$OUTPUT_DIR\" 8000"

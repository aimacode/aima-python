# aima-python in the browser (JupyterLite proof-of-concept)

This folder builds a **JupyterLite** site that runs a few `aima` notebooks
entirely in the browser via [Pyodide](https://pyodide.org) — no server, no
local install. It is the first concrete step of the "Python web companion"
discussed in [issue #1072](https://github.com/aimacode/aima-python/issues/1072).

When deployed, it lives next to the API docs on GitHub Pages:

- API docs: <https://aimacode.github.io/aima-python/>
- Try in browser: <https://aimacode.github.io/aima-python/lite/>

## What runs (and what does not)

Pyodide ships scientific-Python wheels (numpy, scipy, matplotlib, networkx,
pandas, …), so the *lightweight* parts of `aima` work in the browser. The
heavy native dependencies in `requirements.txt` — TensorFlow/Keras, OpenCV
(`cv2`), `cvxopt`, `qpsolvers` — are **not** available in Pyodide, so notebooks
that need them (deep learning, parts of perception, LP-based game theory)
cannot run here. The `aima` wheel is therefore installed with `deps=False` and
the demo notebooks are restricted to modules that import only Pyodide-provided
packages.

The proof-of-concept ships these notebooks:

- `content/search.ipynb` — BFS / A* on the Romania map (`aima.search`, numpy only).
- `content/games.ipynb` — minimax / alpha-beta on Tic-Tac-Toe (`aima.games`, numpy only).
- `content/logic.ipynb` — propositional model checking / DPLL and first-order
  forward chaining (`aima.logic`, needs `networkx`).
- `content/csp.ipynb` — map colouring with AC-3 / backtracking and N-queens with
  min-conflicts (`aima.csp`, needs `sortedcontainers`).
- `content/planning.ipynb` — STRIPS planning (spare tire, air cargo) with
  GraphPlan (`aima.planning`, needs `networkx` + `sortedcontainers`).
- `content/probability.ipynb` — exact (enumeration / variable elimination) and
  approximate (likelihood weighting) inference on the burglary network
  (`aima.probability`, numpy only).

`networkx` and `sortedcontainers` are pure-Python and ship with Pyodide; they are
preloaded for the kernel in `jupyter-lite.json`. `content/Welcome.ipynb` shows the
one-cell install pattern used by every notebook.

Note: `aima.logic`/`aima.csp` pull in `aima.agents`, whose only GUI dependency
(`ipythonblocks`) is imported lazily, so these modules import cleanly in Pyodide
without it.

## Build locally

```bash
cd lite
./build.sh          # builds the aima wheel + runs `jupyter lite build`
python -m http.server -d _output 8000   # then open http://localhost:8000
```

`build.sh` installs the build toolchain from `requirements-lite.txt`, builds an
`aima` wheel from the repo root, and bundles it into the JupyterLite site so the
notebooks can `piplite.install("aima", deps=False)` offline.

## Status / next steps

This is a **proof of concept** (issue #1072 stays open until it is a complete
companion). Remaining work:

- Verify in-browser execution across browsers (Pyodide runs only in a real
  browser, so this cannot be checked in headless CI — the CI job only proves the
  static site *builds*).
- Port any remaining lightweight notebooks (e.g. mdp, learning's non-Keras parts)
  once their in-browser behaviour is confirmed.
- Decide whether to grow this into a full MyST / Jupyter Book textbook companion.

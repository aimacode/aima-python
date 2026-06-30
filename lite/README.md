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

The companion ships a notebook per Pyodide-compatible module:

- `content/agents.ipynb` — reflex vs. model-based agents in the vacuum world (`aima.agents`).
- `content/search.ipynb` — BFS / A* on the Romania map (`aima.search`).
- `content/games.ipynb` — minimax / alpha-beta on Tic-Tac-Toe (`aima.games`).
- `content/csp.ipynb` — map colouring (AC-3 / backtracking) and N-queens (min-conflicts) (`aima.csp`).
- `content/logic.ipynb` — propositional model checking / DPLL and first-order forward chaining (`aima.logic`).
- `content/planning.ipynb` — STRIPS planning (spare tire, air cargo) with GraphPlan (`aima.planning`).
- `content/probability.ipynb` — exact and approximate inference on the burglary network (`aima.probability`).
- `content/mdp.ipynb` — value / policy iteration on the 4x3 grid world (`aima.mdp`).
- `content/reinforcement_learning.ipynb` — Q-learning on the grid world (`aima.reinforcement_learning`).
- `content/learning.ipynb` — decision tree and naive Bayes on an inline dataset (`aima.learning`).
- `content/knowledge.ipynb` — current-best-hypothesis learning (`aima.knowledge`).
- `content/nlp.ipynb` — probabilistic CYK parsing (`aima.nlp`).
- `content/text.ipynb` — unigram / bigram language models (`aima.text`).
- `content/game_theory.ipynb` — Nash equilibria, zero-sum games, Shapley value (`aima.game_theory`).

`content/Welcome.ipynb` shows the one-cell install pattern used by every notebook.

Most modules need only numpy; `aima.logic`/`aima.planning` need `networkx`,
`aima.csp`/`aima.planning` need `sortedcontainers` (both pure-Python and shipped
with Pyodide — preloaded for the kernel in `jupyter-lite.json`). `game_theory.ipynb`
additionally `piplite.install("scipy")`s for its linear-program solver.

A few modules stay out of the companion because they need native wheels Pyodide
does not provide: `deep_learning` (TensorFlow/Keras), `perception` (OpenCV), and
the SVM path in `learning` (`cvxopt`/`qpsolvers`).

To make the importable modules load cleanly in the browser, two optional
dependencies are now imported lazily rather than at module top level: the
`ipythonblocks` GUI dependency in `aima.agents` (so `logic`/`csp`/`planning`/...
import without it) and `qpsolvers` in `aima.learning` (needed only by SVM). The
example datasets `learning` builds at import time are also guarded so the module
imports when the `aima-data` files are absent.

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
  static site *builds*; the notebook code is separately validated against a
  numpy/networkx/sortedcontainers/scipy environment that mirrors Pyodide).
- Decide whether to grow this into a full MyST / Jupyter Book textbook companion.

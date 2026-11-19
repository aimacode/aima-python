#!/usr/bin/env python3
"""Run every companion notebook in a real headless browser (Pyodide WASM).

This is the in-browser counterpart to the build: `build.sh` proves the static
site *builds*; this proves every notebook's code actually *runs* in Pyodide.

It serves the locally-built site (`lite/_output`, so run `./build.sh` first),
loads real Pyodide in headless Chromium, installs the freshly-built `aima` wheel
from the bundled offline index, and executes each notebook's real code cells
(the `piplite` install cell included, via a micropip-backed shim). `ipython` is
loaded to mirror the JupyterLite kernel, which always provides it.

With --lab NAME [NAME ...] it also opens those notebooks in the JupyterLite
app itself, runs all their cells in the real kernel, checks that no cell fails
and that every widget renders, presses Play and Step on the step-through players
and clicks the tiles of the 8-puzzle game, the regions of the map-coloring game,
the squares of the tic-tac-toe game and of the wumpus world (see
aima/widgets.py), and saves a screenshot of each widget in lite/_screenshots.

Setup (one-off):
    pip install playwright nbformat
    python -m playwright install chromium

Usage:
    cd lite && ./build.sh && python verify_browser.py [--lab search]
"""
import argparse
import asyncio
import functools
import glob
import http.server
import os
import pathlib
import threading

import nbformat
from playwright.async_api import async_playwright

HERE = pathlib.Path(__file__).resolve().parent
SITE = HERE / "_output"
CONTENT = HERE / "content"
SHOTS = HERE / "_screenshots"
PORT = 8765
PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js"

# notebooks first, in a sensible reading order; any extras are appended
PREFERRED = ["Welcome", "agents", "search", "games", "csp", "logic", "planning",
             "probability", "mdp", "reinforcement_learning", "learning",
             "knowledge", "nlp", "text", "game_theory"]

HTML = """<!doctype html><html><head><meta charset="utf-8">
<script src="{pyodide}"></script></head><body><script>
window.pyReady = false; window.bootErr = ""; window.OUT = [];
async function boot() {{
  try {{
    window.pyodide = await loadPyodide();
    pyodide.setStdout({{batched: (s) => window.OUT.push(s)}});
    pyodide.setStderr({{batched: (s) => window.OUT.push(s)}});
    // ipython mirrors the JupyterLite kernel (aima.agents imports IPython.display)
    await pyodide.loadPackage(["micropip", "numpy", "scipy", "networkx", "ipython"]);
    // fetch the bundled wheel and install from the Pyodide FS (micropip's by-URL
    // path is unreliable on pyodide 0.27); still exercises the real wheel + WASM
    const resp = await fetch("{wheel_url}");
    if (!resp.ok) throw new Error("wheel fetch HTTP " + resp.status);
    pyodide.FS.writeFile("/tmp/{wheel_name}", new Uint8Array(await resp.arrayBuffer()));
    await pyodide.runPythonAsync(`
import micropip
await micropip.install("sortedcontainers")
import sys, types
_pip = types.ModuleType("piplite")
async def _install(pkgs, deps=True, **kw):
    import micropip
    names = [pkgs] if isinstance(pkgs, str) else list(pkgs)
    for n in names:
        if n == "aima":
            await micropip.install("emfs:/tmp/{wheel_name}", deps=False)
        elif n not in ("scipy", "numpy", "networkx", "sortedcontainers"):
            await micropip.install(n)
_pip.install = _install
sys.modules["piplite"] = _pip
`);
    window.pyReady = true;
  }} catch (e) {{ window.bootErr = e.toString(); }}
}}
boot();
</script></body></html>"""


def serve(directory):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


async def run_cell(page, code):
    return await page.evaluate(
        """async (code) => {
            window.OUT = [];
            try {
                await pyodide.runPythonAsync(code, {globals: window.__ns});
                return {ok: true, out: window.OUT.join('')};
            } catch (e) {
                return {ok: false, out: window.OUT.join('') + '\\n' + e.toString()};
            }
        }""", code)


async def check_lab(browser, name):
    """Run a notebook in the JupyterLite app, exercise its widgets and save
    screenshots of them. Return a list of problems (empty if all went well)."""
    problems = []
    page = await browser.new_page(viewport={"width": 1280, "height": 1000})
    await page.goto(f"http://127.0.0.1:{PORT}/lab/index.html?path={name}.ipynb")
    await page.wait_for_selector(".jp-Notebook .jp-CodeCell", timeout=120000)
    await page.wait_for_timeout(3000)
    await page.locator(".lm-MenuBar-itemLabel", has_text="Run").first.click()
    await page.locator(".lm-Menu-itemLabel", has_text="Run All Cells").first.click()
    # wait until every code cell has an execution count, i.e. "[n]:" and not "[*]:"
    await page.wait_for_function(
        """() => { const p = [...document.querySelectorAll('.jp-CodeCell .jp-InputArea-prompt')];
                   return p.length > 0 && p.every(e => /\\[\\d+\\]/.test(e.textContent)); }""",
        timeout=600000, polling=1000)
    await page.wait_for_timeout(3000)
    errors = await page.evaluate(
        """() => [...document.querySelectorAll('.jp-OutputArea-output')].map(e => e.innerText)
                 .filter(t => /Traceback|Error displaying widget|\\w+Error:/.test(t))""")
    problems += [f"cell output: {e.strip()[-300:]}" for e in errors]

    SHOTS.mkdir(exist_ok=True)
    players = page.locator(".aima-player")
    for i in range(await players.count()):
        player = players.nth(i)
        await player.scroll_into_view_if_needed()
        await player.get_by_role("button", name="Step").click()
        await page.wait_for_timeout(1500)
        stepped = int(await player.locator(".widget-readout").first.inner_text())
        await player.get_by_role("button", name="Play").click()
        await page.wait_for_timeout(3500)  # about 3 steps at the default 1 step per second
        played = int(await player.locator(".widget-readout").first.inner_text())
        if stepped != 1 or played < 3:
            problems.append(f"player {i}: Step went to {stepped}, Play reached {played}")
        await player.screenshot(path=str(SHOTS / f"{name}-player-{i}.png"))
    games = page.locator(".aima-eight-puzzle-game")
    for i in range(await games.count()):
        game = games.nth(i)
        await game.scroll_into_view_if_needed()
        tiles = game.locator(".widget-gridbox button")
        labels = [(await tiles.nth(j).inner_text()).strip() for j in range(9)]
        blank = labels.index("")
        far = next(j for j in range(9) if abs(j // 3 - blank // 3) + abs(j % 3 - blank % 3) > 1)
        near = next(j for j in range(9) if abs(j // 3 - blank // 3) + abs(j % 3 - blank % 3) == 1)
        message = game.locator(".widget-html").last
        await tiles.nth(far).click()
        await page.wait_for_timeout(1500)
        refused = await message.inner_text()
        await tiles.nth(near).click()
        await page.wait_for_timeout(1500)
        moved = await message.inner_text()
        after = [(await tiles.nth(j).inner_text()).strip() for j in range(9)]
        if ("cannot move" not in refused or not moved.startswith("Moves: 1")
                or after[near] != "" or after[blank] != labels[near]):
            problems.append(f"8-puzzle game {i}: {refused!r}, then {moved!r} with tiles {after}")
        await game.screenshot(path=str(SHOTS / f"{name}-game-{i}.png"))
    maps = page.locator(".aima-map-coloring-game")
    for i in range(await maps.count()):
        game = maps.nth(i)
        await game.scroll_into_view_if_needed()
        region = game.locator(".widget-hbox button")
        await region.nth(0).click()  # the first region, in the default color
        await page.wait_for_timeout(1500)
        await region.nth(1).click()  # a neighbor of the first region, same color
        await page.wait_for_timeout(1500)
        message = await game.locator(".widget-html").last.inner_text()
        if "cannot be" not in message:
            problems.append(f"map coloring game {i}: {message!r}")
        await game.screenshot(path=str(SHOTS / f"{name}-map-game-{i}.png"))
    boards = page.locator(".aima-tic-tac-toe-game")
    for i in range(await boards.count()):
        game = boards.nth(i)
        await game.scroll_into_view_if_needed()
        squares = game.locator(".widget-gridbox button")
        await squares.nth(0).click()
        await page.wait_for_timeout(3000)
        message = await game.locator(".widget-html").last.inner_text()
        marks = [(await squares.nth(j).inner_text()).strip() for j in range(9)]
        await squares.nth(0).click()  # taken
        await page.wait_for_timeout(1500)
        taken = await game.locator(".widget-html").last.inner_text()
        if marks.count("X") != 1 or marks.count("O") != 1 or "looked at" not in message \
                or "taken" not in taken:
            problems.append(f"tic-tac-toe game {i}: {marks}, {message!r}, {taken!r}")
        await game.screenshot(path=str(SHOTS / f"{name}-tic-tac-toe-{i}.png"))
    worlds = page.locator(".aima-wumpus-world-game")
    for i in range(await worlds.count()):
        game = worlds.nth(i)
        await game.scroll_into_view_if_needed()
        squares = game.locator(".widget-gridbox button")  # top row first, as on the grid
        for index in (13, 12, 8):  # [2,1], [1,1], [1,2] on a 4x4 grid
            await squares.nth(index).click()
            await page.wait_for_timeout(2000)
        labels = [(await squares.nth(j).inner_text()).strip() for j in range(16)]
        if labels[4] != "wumpus!" or labels[14] != "pit!" or labels[9] != "safe":
            problems.append(f"wumpus world {i}: {labels}")
        await game.screenshot(path=str(SHOTS / f"{name}-wumpus-{i}.png"))
    await page.screenshot(path=str(SHOTS / f"{name}-page.png"), full_page=True)
    await page.close()
    return problems


def notebook_order():
    names = [p.stem for p in CONTENT.glob("*.ipynb")]
    ordered = [n for n in PREFERRED if n in names]
    return ordered + sorted(n for n in names if n not in PREFERRED)


async def main(lab=()):
    if not SITE.exists():
        raise SystemExit("lite/_output not found — run ./build.sh first")
    wheels = glob.glob(str(SITE / "pypi" / "aima-*.whl"))
    if not wheels:
        raise SystemExit("no aima wheel in lite/_output/pypi — run ./build.sh first")
    wheel_name = os.path.basename(wheels[0])

    html = HTML.format(pyodide=PYODIDE, wheel_url=f"pypi/{wheel_name}", wheel_name=wheel_name)
    (SITE / "verify.html").write_text(html)
    serve(SITE)

    results = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        page.on("pageerror", lambda e: print("PAGEERROR:", e))
        await page.goto(f"http://127.0.0.1:{PORT}/verify.html")
        try:
            await page.wait_for_function("window.pyReady === true", timeout=180000)
        except Exception:
            raise SystemExit("boot failed: " + await page.evaluate("() => window.bootErr"))
        print(f"Pyodide booted, {wheel_name} installed.\n")

        for name in notebook_order():
            await page.evaluate("() => { if (window.__ns) window.__ns.destroy();"
                                " window.__ns = pyodide.globals.get('dict')(); }")
            nb = nbformat.read(str(CONTENT / f"{name}.ipynb"), as_version=4)
            ok, out = True, ""
            for cell in nb.cells:
                if cell.cell_type != "code":
                    continue
                r = await run_cell(page, cell.source)
                out += r["out"]
                if not r["ok"]:
                    ok = False
                    break
            results[name] = ok
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
            if not ok:
                print("\n".join("        " + l for l in out.strip().splitlines()[-8:]))

        for name in lab:
            problems = await check_lab(browser, name)
            results[f"{name} (JupyterLite)"] = not problems
            print(f"  {'PASS' if not problems else 'FAIL'}  {name} in the JupyterLite app")
            for problem in problems:
                print("        " + problem)
        await browser.close()

    passed = sum(results.values())
    print(f"\n{passed}/{len(results)} notebook runs succeeded in a real browser")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lab", nargs="+", default=[], metavar="NAME",
                        help="also run these notebooks in the JupyterLite app and check their widgets")
    asyncio.run(main(parser.parse_args().lab))

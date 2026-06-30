#!/usr/bin/env python3
"""Run every companion notebook in a real headless browser (Pyodide WASM).

This is the in-browser counterpart to the build: `build.sh` proves the static
site *builds*; this proves every notebook's code actually *runs* in Pyodide.

It serves the locally-built site (`lite/_output`, so run `./build.sh` first),
loads real Pyodide in headless Chromium, installs the freshly-built `aima` wheel
from the bundled offline index, and executes each notebook's real code cells
(the `piplite` install cell included, via a micropip-backed shim). `ipython` is
loaded to mirror the JupyterLite kernel, which always provides it.

Setup (one-off):
    pip install playwright nbformat
    python -m playwright install chromium

Usage:
    cd lite && ./build.sh && python verify_browser.py
"""
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


def notebook_order():
    names = [p.stem for p in CONTENT.glob("*.ipynb")]
    ordered = [n for n in PREFERRED if n in names]
    return ordered + sorted(n for n in names if n not in PREFERRED)


async def main():
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
        await browser.close()

    passed = sum(results.values())
    print(f"\n{passed}/{len(results)} notebooks ran successfully in a real browser")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())

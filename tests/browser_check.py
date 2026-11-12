"""End-to-end check of the web demo in headless Chrome.

    python -m http.server 8000 -d docs &
    python tests/browser_check.py

For each model (Arabic, Latin) it
1. feeds real test-set images through the in-browser ONNX model and compares
   with the labels (parity with the Python evaluation), and
2. draws a few rough synthetic letters with the mouse API and prints what the
   page predicts, which exercises the stroke -> image preprocessing.
"""
import sys
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets import SPECS

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/"

# Rough stroke shapes in unit coordinates (x right, y down). Dots are tiny strokes.
dot = lambda x, y: [(x, y), (x + 0.005, y + 0.005)]
ARABIC = {
    "alef": [[(0.5, 0.2), (0.5, 0.8)]],
    "lam": [[(0.6, 0.15), (0.6, 0.7), (0.5, 0.82), (0.36, 0.82), (0.28, 0.7), (0.28, 0.6)]],
    "beh": [[(0.8, 0.42), (0.8, 0.55), (0.7, 0.62), (0.3, 0.62), (0.2, 0.55), (0.2, 0.42)], dot(0.5, 0.78)],
    "noon": [[(0.72, 0.4), (0.72, 0.62), (0.62, 0.74), (0.4, 0.74), (0.3, 0.62), (0.3, 0.4)], dot(0.51, 0.3)],
    "reh": [[(0.6, 0.35), (0.6, 0.55), (0.5, 0.72), (0.3, 0.8)]],
}
_C = [(0.68, 0.32), (0.56, 0.24), (0.42, 0.26), (0.32, 0.38), (0.3, 0.52), (0.36, 0.66), (0.48, 0.72), (0.6, 0.7), (0.68, 0.62)]
_O = [(0.5, 0.3), (0.38, 0.34), (0.32, 0.48), (0.36, 0.64), (0.5, 0.7), (0.64, 0.64), (0.68, 0.48), (0.62, 0.34), (0.5, 0.3)]
_U = [(0.32, 0.32), (0.32, 0.58), (0.4, 0.7), (0.56, 0.7), (0.66, 0.58), (0.66, 0.32)]
_S = [(0.64, 0.3), (0.5, 0.24), (0.38, 0.3), (0.38, 0.42), (0.5, 0.48), (0.62, 0.54), (0.62, 0.66), (0.5, 0.72), (0.36, 0.66)]
LATIN = {
    "L l": [[(0.38, 0.2), (0.38, 0.78), (0.68, 0.78)]],
    "T t": [[(0.25, 0.22), (0.75, 0.22)], [(0.5, 0.22), (0.5, 0.8)]],
    "V v": [[(0.28, 0.22), (0.5, 0.8), (0.72, 0.22)]],
    "C c": [_C],
    "Ç ç": [_C, [(0.5, 0.72), (0.52, 0.8), (0.46, 0.88)]],
    "O o": [_O],
    "Ö ö": [_O, dot(0.42, 0.16), dot(0.58, 0.16)],
    "U u": [_U],
    "Ü ü": [_U, dot(0.4, 0.18), dot(0.58, 0.18)],
    "S s": [_S],
    "Ş ş": [_S, [(0.5, 0.72), (0.52, 0.8), (0.46, 0.88)]],
    "I ı": [[(0.5, 0.25), (0.5, 0.78)]],
    "İ i": [[(0.5, 0.36), (0.5, 0.8)], dot(0.5, 0.2)],
}

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_context(viewport={"width": 1100, "height": 980}, locale="en-US").new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(URL)
    ok = True
    for script, shapes in (("arabic", ARABIC), ("latin", LATIN)):
        spec = SPECS[script]
        page.evaluate("s => harf.setScript(s)", script)
        page.wait_for_function("s => document.documentElement.dataset.ready === s", arg=script, timeout=30000)
        print(f"\n[{script}] {page.inner_text('#status')}")

        x, y = spec.load_test("data")
        idx = np.random.default_rng(0).choice(len(y), 320, replace=False)
        preds = page.evaluate(
            """async ([imgs, s]) => { const out = []; for (const im of imgs) { const p = await harf.classify(Float32Array.from(im), s); out.push(p.indexOf(Math.max(...p))); } return out; }""",
            [(x[idx].reshape(len(idx), -1) / 255.0).round(4).tolist(), script],
        )
        acc = (np.array(preds) == y[idx]).mean()
        print(f"in-browser accuracy on {len(idx)} test images: {acc:.4f}")
        ok &= acc > 0.9

        box = page.locator("#pad").bounding_box()
        hits = 0
        for name, strokes in shapes.items():
            if page.is_enabled("#clear"):
                page.click("#clear")
            for s in strokes:
                pts = [(box["x"] + u * box["width"], box["y"] + v * box["height"]) for u, v in s]
                page.mouse.move(*pts[0])
                page.mouse.down()
                for q in pts[1:]:
                    page.mouse.move(*q, steps=6)
                page.mouse.up()
            page.wait_for_timeout(250)
            got, conf = page.inner_text("#glyph" if script == "latin" else "#name"), page.inner_text("#conf")
            want = name.split()[0] if script == "latin" else name
            hits += got == want
            print(f"  drew {name:5s} -> {got:5s} ({conf})")
            page.screenshot(path=f"/tmp/demo_{script}_{spec.names[[g for g in spec.glyphs].index(name)] if script == 'latin' else name}.png")
        print(f"synthetic drawings recognised: {hits}/{len(shapes)}")
    print("\nconsole errors:", errors or "none")
    browser.close()
    assert ok and not errors

"""Latin letters for English and Turkish: EMNIST Letters plus synthesised Turkish diacritics.

EMNIST Letters gives 26 classes of real handwriting with upper and lower case
merged. No public handwriting set covers the Turkish letters ç, ğ, İ/i, ö, ş, ü,
so those six classes are built here by drawing a randomised cedilla, breve or
dot(s) onto real EMNIST base letters (c, g, I, o, s, u) and re-normalising the
result the way EMNIST normalises every glyph. Real dotted i samples found in
the EMNIST "I" class are moved to the İ/i class.
"""
from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

BASE_URL = "https://github.com/ivarreukers/emnist-letters/raw/master/"
FILES = {
    "train_x": "emnist-letters-train-images-idx3-ubyte.gz",
    "train_y": "emnist-letters-train-labels-idx1-ubyte.gz",
    "test_x": "emnist-letters-test-images-idx3-ubyte.gz",
    "test_y": "emnist-letters-test-labels-idx1-ubyte.gz",
}

# 32 classes in Turkish alphabet order, with q, w, x slotted in where English has them.
# (display glyphs, ascii name, EMNIST base letter, diacritic or None)
CLASSES = [
    ("A a", "a", "a", None), ("B b", "b", "b", None), ("C c", "c", "c", None), ("Ç ç", "c-cedilla", "c", "cedilla"),
    ("D d", "d", "d", None), ("E e", "e", "e", None), ("F f", "f", "f", None), ("G g", "g", "g", None),
    ("Ğ ğ", "g-breve", "g", "breve"), ("H h", "h", "h", None), ("I ı", "dotless-i", "i", None), ("İ i", "dotted-i", "i", "dot"),
    ("J j", "j", "j", None), ("K k", "k", "k", None), ("L l", "l", "l", None), ("M m", "m", "m", None),
    ("N n", "n", "n", None), ("O o", "o", "o", None), ("Ö ö", "o-umlaut", "o", "dots"), ("P p", "p", "p", None),
    ("Q q", "q", "q", None), ("R r", "r", "r", None), ("S s", "s", "s", None), ("Ş ş", "s-cedilla", "s", "cedilla"),
    ("T t", "t", "t", None), ("U u", "u", "u", None), ("Ü ü", "u-umlaut", "u", "dots"), ("V v", "v", "v", None),
    ("W w", "w", "w", None), ("X x", "x", "x", None), ("Y y", "y", "y", None), ("Z z", "z", "z", None),
]
GLYPHS = [c[0] for c in CLASSES]
NAMES = [c[1] for c in CLASSES]
SYNTHETIC = [i for i, c in enumerate(CLASSES) if c[3]]
I_DOTLESS, I_DOTTED = NAMES.index("dotless-i"), NAMES.index("dotted-i")


def download(root: str | Path = "data") -> Path:
    root = Path(root) / "emnist"
    root.mkdir(parents=True, exist_ok=True)
    for f in FILES.values():
        if not (root / f).exists():
            print(f"Downloading {f} ...")
            urllib.request.urlretrieve(BASE_URL + f, root / f)
    return root


def load_emnist(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns upright images uint8 [N,28,28] and labels 0..25 (a..z)."""
    with gzip.open(root / FILES[f"{split}_x"]) as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).transpose(0, 2, 1)
    with gzip.open(root / FILES[f"{split}_y"]) as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64) - 1
    return x, y


def has_dot(img: np.ndarray) -> bool:
    """True if the glyph has a small separate blob above its main body (a dotted i)."""
    lab, n = ndimage.label(img > 90)
    if n != 2:
        return False
    sizes = ndimage.sum(np.ones_like(lab), lab, [1, 2])
    small, big = (1, 2) if sizes[0] < sizes[1] else (2, 1)
    if sizes[small - 1] > 0.45 * sizes[big - 1]:
        return False
    cy_small = ndimage.center_of_mass(lab == small)[0]
    top_big = np.where((lab == big).any(1))[0][0]
    return cy_small < top_big + 1


def _stroke(canvas: np.ndarray, pts: np.ndarray, width: float) -> None:
    """Draws an anti-aliased polyline (or a dot, for one point) onto canvas in place."""
    h, w = canvas.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if len(pts) == 1:
        pts = np.vstack([pts, pts + 1e-3])
    d = np.full((h, w), 1e9, np.float32)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        vx, vy = x1 - x0, y1 - y0
        t = np.clip(((xx - x0) * vx + (yy - y0) * vy) / (vx * vx + vy * vy + 1e-9), 0, 1)
        d = np.minimum(d, np.hypot(xx - (x0 + t * vx), yy - (y0 + t * vy)))
    np.maximum(canvas, np.clip(width / 2 + 0.5 - d, 0, 1), out=canvas)


def normalise(canvas: np.ndarray, side: float = 22.0) -> np.ndarray:
    """Crops the ink, scales its longer side to `side` px and centres it in 28x28 (as EMNIST does)."""
    ys, xs = np.where(canvas > 0.15)
    crop = canvas[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    k = side / max(crop.shape)
    nh, nw = max(1, round(crop.shape[0] * k)), max(1, round(crop.shape[1] * k))
    small = F.interpolate(torch.from_numpy(crop)[None, None], size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)[0, 0].numpy()
    out = np.zeros((28, 28), np.float32)
    oy, ox = (28 - nh) // 2, (28 - nw) // 2
    out[oy: oy + nh, ox: ox + nw] = small
    return out


def add_diacritic(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Draws a handwritten-looking diacritic onto a 28x28 uint8 glyph; returns uint8 28x28."""
    S = 3  # work at 3x resolution so the mark has smooth edges after downscaling
    big = F.interpolate(torch.from_numpy(img.astype(np.float32) / 255)[None, None], scale_factor=S, mode="bilinear", align_corners=False)[0, 0].numpy()
    canvas = np.zeros((28 * S + 60, 28 * S + 40), np.float32)
    canvas[30: 30 + 28 * S, 20: 20 + 28 * S] = big
    ys, xs = np.where(canvas > 0.35)
    top, bot, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    wdt, hgt = right - left + 1, bot - top + 1
    # Stroke width of the mark at 3x scale, randomised around the typical EMNIST pen width.
    sw = float(np.clip(rng.normal(8.0, 1.2), 5.5, 11))
    j = lambda s: rng.normal(0, s)

    if kind in ("dots", "dot", "breve"):
        band = canvas[top: top + max(3, hgt // 4)] > 0.35
        cols = np.where(band.any(0))[0]
        cx = (cols.min() + cols.max()) / 2 if kind == "dot" else (left + right) / 2
        gap = rng.uniform(4, 10)
        if kind == "dot":
            y = top - gap - sw / 2 + j(1)
            x = cx + j(2.5)
            _stroke(canvas, np.array([[x, y], [x + j(1.5), y + j(1.5)]]), sw * rng.uniform(0.95, 1.3))
        elif kind == "dots":
            sep = max(wdt * rng.uniform(0.22, 0.36), sw * 1.15)
            tilt = j(1.5)
            for sgn in (-1, 1):
                x, y = cx + sgn * sep + j(1.5), top - gap - sw / 2 + sgn * tilt + j(1)
                dash = rng.uniform(0, 3.5) if rng.random() < 0.4 else 0.01  # some people write short dashes
                _stroke(canvas, np.array([[x, y - dash], [x + j(0.8), y + dash]]), sw * rng.uniform(0.9, 1.2))
        else:  # breve: a shallow bowl
            half = max(wdt * rng.uniform(0.2, 0.34), sw * 1.3)
            depth = rng.uniform(4, 9)
            t = np.linspace(-1, 1, 9)
            x = cx + j(2.5) + half * t
            y = top - gap - sw / 2 - depth * (t ** 2) + j(1) + t * j(1.2)
            _stroke(canvas, np.stack([x, y], 1), sw * rng.uniform(0.7, 0.95))
    else:  # cedilla: a small hook hanging from the bottom of the letter
        band = canvas[bot - max(3, hgt // 6): bot + 1] > 0.35
        cols = np.where(band.any(0))[0]
        cx = (cols.min() + cols.max()) / 2 + j(2)
        L = rng.uniform(9, 15)
        if rng.random() < 0.5:   # comma-like tick
            pts = np.array([[cx, bot - 2], [cx - L * 0.15 + j(1), bot + L * 0.5], [cx - L * 0.55 + j(1.5), bot + L]])
        else:                    # the classic "5"-shaped hook
            pts = np.array([[cx, bot - 2], [cx + j(1), bot + L * 0.35], [cx + L * 0.4, bot + L * 0.6],
                            [cx + L * 0.25, bot + L * 0.95], [cx - L * 0.35 + j(1.5), bot + L * 1.05]])
        _stroke(canvas, pts, sw * rng.uniform(0.65, 0.9))

    canvas = ndimage.gaussian_filter(canvas, 1.2)
    canvas[30: 30 + 28 * S, 20: 20 + 28 * S] = np.maximum(canvas[30: 30 + 28 * S, 20: 20 + 28 * S], big)
    return (normalise(canvas) * 255).clip(0, 255).astype(np.uint8)


def build_split(root: Path, split: str, per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Builds the 32-class dataset for one split. Cached as npz next to the raw files."""
    cache = root / f"latin_{split}_{per_class}_{seed}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x"], d["y"]
    rng = np.random.default_rng(seed)
    x, y = load_emnist(root, split)
    letters = "abcdefghijklmnopqrstuvwxyz"
    pools = {c: x[y == k] for k, c in enumerate(letters)}
    dotted = np.array([has_dot(im) for im in pools["i"]])
    i_dotted, i_plain = pools["i"][dotted], pools["i"][~dotted]
    print(f"[{split}] EMNIST 'i' class: {dotted.sum()} dotted, {(~dotted).sum()} dotless")

    def take(pool, n):
        return pool if len(pool) <= n else pool[rng.choice(len(pool), n, replace=False)]

    xs, ys = [], []
    for cls, (_, name, base, mark) in enumerate(CLASSES):
        if mark is None:
            imgs = take(i_plain if name == "dotless-i" else pools[base], per_class)
        elif mark == "dot":
            real = take(i_dotted, min(per_class // 2, len(i_dotted)))
            synth = np.stack([add_diacritic(im, "dot", rng) for im in take(i_plain, per_class - len(real))])
            imgs = np.concatenate([real, synth])
        else:
            imgs = np.stack([add_diacritic(im, mark, rng) for im in take(pools[base], per_class)])
        xs.append(imgs)
        ys.append(np.full(len(imgs), cls, np.int64))
    x, y = np.concatenate(xs), np.concatenate(ys)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Mild affine jitter and stroke-width change for 28x28 batches."""
    n = x.size(0)
    ang = (torch.rand(n) - 0.5) * 2 * np.deg2rad(10)
    scale = 1 + (torch.rand(n) - 0.5) * 0.24
    shear = (torch.rand(n) - 0.5) * 0.3
    tx, ty = (torch.rand(n) - 0.5) * 0.16, (torch.rand(n) - 0.5) * 0.16
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin + shear, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    r = torch.rand(n)
    thick, thin = r < 0.2, r > 0.7
    if thick.any():
        x[thick] = F.max_pool2d(x[thick], 3, 1, 1)
    if thin.any():
        x[thin] = -F.max_pool2d(-x[thin], 3, 1, 1)
    return x.clamp(0, 1)

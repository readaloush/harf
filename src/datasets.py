"""One interface over the two datasets so training, evaluation and export share code."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import data as ahcd
from . import latin


@dataclass
class Spec:
    name: str            # file prefix: results/<name>.pt, docs/<name>.onnx
    size: int            # input side in pixels
    glyphs: list[str]    # what to show for each class
    names: list[str]     # ascii class names
    augment: Callable
    load_train: Callable[[str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]  # x_tr, y_tr, x_val, y_val
    load_test: Callable[[str], tuple[np.ndarray, np.ndarray]]
    synthetic: list[int]  # class indices whose samples are (partly) synthesised


def _ahcd_train(root):
    x, y = ahcd.load_split(ahcd.download(root), "train")
    tr, va = ahcd.holdout_split(len(y))
    return x[tr], y[tr], x[va], y[va]


def _latin_train(root, per_class=2000, seed=0):
    x, y = latin.build_split(latin.download(root), "train", per_class, seed)
    idx = np.random.default_rng(seed).permutation(len(y))
    cut = len(y) // 10
    return x[idx[cut:]], y[idx[cut:]], x[idx[:cut]], y[idx[:cut]]


SPECS = {
    "arabic": Spec("harfnet", 32, [g for g, _ in ahcd.LETTERS], [n for _, n in ahcd.LETTERS], ahcd.augment_batch,
                   _ahcd_train, lambda root: ahcd.load_split(root, "test"), []),
    "latin": Spec("harfnet_latin", 28, latin.GLYPHS, latin.NAMES, latin.augment_batch,
                  _latin_train, lambda root: latin.build_split(latin.download(root), "test", 800, 1), latin.SYNTHETIC),
}

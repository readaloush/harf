"""Loading and augmentation for the Arabic Handwritten Characters Dataset (AHCD).

The dataset is not redistributed in this repo. `download()` fetches the
original CSV archive from the authors' GitHub repository.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

URL = (
    "https://github.com/mloey/Arabic-Handwritten-Characters-Dataset/raw/master/"
    "Arabic%20Handwritten%20Characters%20Dataset%20CSV.zip"
)
FILES = {
    "train_x": "csvTrainImages 13440x1024.csv",
    "train_y": "csvTrainLabel 13440x1.csv",
    "test_x": "csvTestImages 3360x1024.csv",
    "test_y": "csvTestLabel 3360x1.csv",
}

# Class index -> (Arabic letter, transliteration). AHCD labels are 1..28 in alphabet order.
LETTERS = [
    ("ا", "alef"), ("ب", "beh"), ("ت", "teh"), ("ث", "theh"), ("ج", "jeem"),
    ("ح", "hah"), ("خ", "khah"), ("د", "dal"), ("ذ", "thal"), ("ر", "reh"),
    ("ز", "zain"), ("س", "seen"), ("ش", "sheen"), ("ص", "sad"), ("ض", "dad"),
    ("ط", "tah"), ("ظ", "zah"), ("ع", "ain"), ("غ", "ghain"), ("ف", "feh"),
    ("ق", "qaf"), ("ك", "kaf"), ("ل", "lam"), ("م", "meem"), ("ن", "noon"),
    ("ه", "heh"), ("و", "waw"), ("ي", "yeh"),
]


def download(root: str | Path = "data") -> Path:
    root = Path(root)
    if all((root / f).exists() for f in FILES.values()):
        return root
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading AHCD to {root}/ ...")
    with urllib.request.urlopen(URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(root)
    return root


def load_split(root: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (images uint8 [N, 32, 32], labels int64 [N] in 0..27)."""
    root = Path(root)
    cache = root / f"{split}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["x"], d["y"]
    x = np.loadtxt(root / FILES[f"{split}_x"], delimiter=",", dtype=np.uint8)
    y = np.loadtxt(root / FILES[f"{split}_y"], delimiter=",", dtype=np.int64) - 1
    # The CSV stores each image column-major (MATLAB order), so transpose to get upright letters.
    x = x.reshape(-1, 32, 32).transpose(0, 2, 1).copy()
    np.savez_compressed(cache, x=x, y=y)
    return x, y


def holdout_split(n: int, val_blocks: int = 6):
    """Train/validation split by contiguous blocks instead of at random.

    The training CSV is ordered in 60 blocks of 224 images (28 letters x 8),
    and neighbouring samples come from the same handwriting. Holding out the
    last `val_blocks` blocks whole keeps near-duplicates of a validation image
    out of the training set, which a random split would not.
    """
    cut = n - val_blocks * 224
    idx = np.arange(n)
    return idx[:cut], idx[cut:]


class AHCD(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool = False):
        self.x = torch.from_numpy(x).float().div(255.0).unsqueeze(1)  # [N,1,32,32]
        self.y = torch.from_numpy(y)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Random affine jitter + stroke-width change, applied to a whole batch.

    Kept mild on purpose: several Arabic letters differ only by the number or
    position of dots, so strong distortion or any flipping would destroy labels.
    """
    n = x.size(0)
    ang = (torch.rand(n) - 0.5) * 2 * np.deg2rad(12)
    scale = 1 + (torch.rand(n) - 0.5) * 0.3
    shear = (torch.rand(n) - 0.5) * 0.3
    tx, ty = (torch.rand(n) - 0.5) * 0.25, (torch.rand(n) - 0.5) * 0.25
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack(
        [torch.stack([cos, -sin + shear, tx], 1), torch.stack([sin, cos, ty], 1)], 1
    )
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

    # Stroke width: dilate (max-pool) or erode (min-pool) a random third of the batch each.
    r = torch.rand(n)
    thick, thin = r < 0.33, r > 0.75
    if thick.any():
        x[thick] = torch.nn.functional.max_pool2d(x[thick], 3, 1, 1)
    if thin.any():
        x[thin] = -torch.nn.functional.max_pool2d(-x[thin], 2, 1, 1)[..., :32, :32]
    return x.clamp(0, 1)

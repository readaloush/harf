"""Evaluate the best checkpoint on the test set and write metrics and figures.

    python -m src.evaluate --dataset arabic
    python -m src.evaluate --dataset latin
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .datasets import SPECS
from .model import HarfNet, count_params

INK, MUTED, ACCENT = "#1b1f24", "#6b7280", "#0f766e"


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = HarfNet(num_classes=ck.get("classes", 28), width=ck["width"])
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(SPECS), default="arabic")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    out = Path(a.out)
    spec = SPECS[a.dataset]
    NAMES, K = spec.names, len(spec.names)
    # Tick labels: transliterated names for Arabic, the lowercase letter itself for Latin.
    TICKS = NAMES if a.dataset == "arabic" else [g.split()[-1] for g in spec.glyphs]

    model, ck = load_model(out / f"{spec.name}.pt")
    x, y = spec.load_test(a.data)
    xt = torch.from_numpy(x).float().div(255).unsqueeze(1)
    with torch.no_grad():
        logits = torch.cat([model(b) for b in xt.split(2048)])
    pred = logits.argmax(1).numpy()
    top3 = (logits.topk(3, 1).indices.numpy() == y[:, None]).any(1).mean()

    cm = np.zeros((K, K), int)
    np.add.at(cm, (y, pred), 1)
    acc = float((pred == y).mean())
    recall = cm.diagonal() / cm.sum(1)
    precision = cm.diagonal() / np.maximum(cm.sum(0), 1)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((int(off[i, j]), NAMES[i], NAMES[j]) for i in range(K) for j in range(K) if off[i, j]), reverse=True)[:8]

    metrics = {
        "test_accuracy": round(acc, 4),
        "test_top3_accuracy": round(float(top3), 4),
        "test_images": int(len(y)),
        "errors": int((pred != y).sum()),
        "parameters": count_params(model),
        "best_epoch": ck["epoch"],
        "val_accuracy": round(ck["val_acc"], 4),
        "per_class": {NAMES[i]: {"precision": round(float(precision[i]), 3), "recall": round(float(recall[i]), 3)} for i in range(K)},
        "most_confused": [{"true": t, "predicted": q, "count": c} for c, t, q in pairs],
    }
    if spec.synthetic:
        syn = np.isin(y, spec.synthetic)
        metrics["real_handwriting_accuracy"] = round(float((pred[~syn] == y[~syn]).mean()), 4)
        metrics["real_handwriting_images"] = int((~syn).sum())
        metrics["synthetic_diacritic_accuracy"] = round(float((pred[syn] == y[syn]).mean()), 4)
        metrics["synthetic_diacritic_images"] = int(syn.sum())
    (out / f"{spec.name}_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=1))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

    # Confusion matrix (row-normalised)
    fig, ax = plt.subplots(figsize=(8.2, 7.4), dpi=160)
    norm = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(K), TICKS, rotation=90 if a.dataset == "arabic" else 0)
    ax.set_yticks(range(K), TICKS)
    for i in range(K):
        for j in range(K):
            if cm[i, j] and i != j:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6.5, color="#b91c1c")
    ax.set_xlabel("Predicted letter")
    ax.set_ylabel("True letter")
    ax.set_title(f"{'AHCD' if a.dataset == 'arabic' else 'EMNIST + Turkish'} test set: {acc:.2%} accuracy, {metrics['errors']:,} errors in {len(y):,} images", loc="left", color=INK, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Share of true class")
    fig.text(0.01, 0.01, "Red numbers: misclassified image counts", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out / f"{spec.name}_confusion.png")

    # Training curves
    hist = json.loads((out / f"{spec.name}_history.json").read_text())
    ep = [h["epoch"] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3), dpi=160)
    a1.plot(ep, [h["train_loss"] for h in hist], color=INK, lw=1.8)
    a1.set_title("Training loss (label-smoothed cross-entropy)", loc="left", fontsize=10, color=INK)
    a1.set_xlabel("Epoch")
    a2.plot(ep, [h["train_acc"] for h in hist], color=MUTED, lw=1.5, label="train (augmented)")
    a2.plot(ep, [h["val_acc"] for h in hist], color=ACCENT, lw=2, label="validation")
    a2.set_ylim(0.8, 1.0)
    a2.set_title("Accuracy", loc="left", fontsize=10, color=INK)
    a2.set_xlabel("Epoch")
    a2.legend(frameon=False)
    for ax in (a1, a2):
        ax.grid(axis="y", color="#e5e7eb", lw=0.6)
    fig.tight_layout()
    fig.savefig(out / f"{spec.name}_curves.png")


if __name__ == "__main__":
    main()

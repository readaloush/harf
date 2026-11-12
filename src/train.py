"""Train HarfNet.

    python -m src.train --dataset arabic --epochs 30    # AHCD, 28 Arabic letters
    python -m src.train --dataset latin --epochs 12     # EMNIST + Turkish, 33 Latin letters

Model selection uses a validation split held out of the training set;
the test set is only touched by src/evaluate.py.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import AHCD
from .datasets import SPECS
from .model import HarfNet, count_params


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(SPECS), default="arabic")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    spec = SPECS[a.dataset]
    xt, yt, xv, yv = spec.load_train(a.data)
    train_dl = DataLoader(AHCD(xt, yt), batch_size=a.batch, shuffle=True, drop_last=True)
    val_dl = DataLoader(AHCD(xv, yv), batch_size=512)
    print(f"{a.dataset}: train {len(yt)}  val {len(yv)}  classes {len(spec.names)}")

    model = HarfNet(num_classes=len(spec.names), width=a.width)
    print(f"HarfNet: {count_params(model):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(train_dl))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    history, best = [], 0.0
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, run_loss, run_correct, seen = time.time(), 0.0, 0, 0
        for xb, yb in train_dl:
            xb = spec.augment(xb)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(yb)
            run_correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)
        val_acc = accuracy(model, val_dl)
        history.append({"epoch": epoch, "train_loss": run_loss / seen, "train_acc": run_correct / seen, "val_acc": val_acc})
        flag = ""
        if val_acc >= best:
            best = val_acc
            torch.save({"state_dict": model.state_dict(), "width": a.width, "classes": len(spec.names), "epoch": epoch, "val_acc": val_acc}, out / f"{spec.name}.pt")
            flag = " *"
        print(f"epoch {epoch:3d}  loss {run_loss/seen:.4f}  train {run_correct/seen:.4f}  val {val_acc:.4f}  {time.time()-t0:.0f}s{flag}", flush=True)

    (out / f"{spec.name}_history.json").write_text(json.dumps(history, indent=1))
    print(f"best val acc {best:.4f}")


if __name__ == "__main__":
    main()

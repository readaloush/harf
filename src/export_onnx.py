"""Export the checkpoint to ONNX for the in-browser demo and check parity.

    python -m src.export_onnx --dataset arabic
    python -m src.export_onnx --dataset latin

The exported graph takes a [N,1,S,S] float image (S = 32 Arabic, 28 Latin) in 0..1 and returns
class probabilities (softmax is baked in so the web page needs no math).
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .datasets import SPECS
from .evaluate import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(SPECS), default="arabic")
    p.add_argument("--data", default="data")
    a = p.parse_args()
    spec = SPECS[a.dataset]
    a.out = f"docs/{spec.name}.onnx"

    model, _ = load_model(f"results/{spec.name}.pt")
    wrapped = torch.nn.Sequential(model, torch.nn.Softmax(dim=1)).eval()
    torch.onnx.export(
        wrapped, torch.zeros(1, 1, spec.size, spec.size), a.out, input_names=["image"], output_names=["probs"],
        dynamic_axes={"image": {0: "n"}, "probs": {0: "n"}}, opset_version=17, dynamo=False,
    )

    x, y = spec.load_test(a.data)
    xt = torch.from_numpy(x).float().div(255).unsqueeze(1)
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    onnx_probs = sess.run(None, {"image": xt.numpy()})[0]
    with torch.no_grad():
        torch_probs = wrapped(xt).numpy()
    print(f"wrote {a.out} ({Path(a.out).stat().st_size/1e6:.2f} MB)")
    print(f"max |torch - onnx| = {np.abs(onnx_probs - torch_probs).max():.2e}")
    print(f"ONNX test accuracy = {(onnx_probs.argmax(1) == y).mean():.4f}")


if __name__ == "__main__":
    main()

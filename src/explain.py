"""
Phase 6a - explainability: what does the image detector look at, and what does an
adversarial perturbation do to it?

For one detected stego image per embedding method we show

  1. the stego image
  2. Grad-CAM of the "stego" logit on the last conv block (8x8, upsampled)
  3. the attack's perturbation, |adv - stego|, scaled so a 1-grey-level change is
     clearly visible (the attack is invisible at 1 grey level; this column makes
     it visible)
  4. Grad-CAM of the "stego" logit on the adversarial file, and the detector's
     stego probability before -> after

Grad-CAM (Selvaraju et al.): weight each channel of a late activation map by the
mean gradient of the class score with respect to that channel, sum, ReLU.

    python3 explain.py --eps 1
"""

import os
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import attacks as A
from dataset_loader import load_manifest, filter_manifest, ImageStegoDataset
from limit_analysis import load_detector

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "results", "phase5")
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def grad_cam(model, x, cls=1):
    """Grad-CAM for class `cls` on model.g5's output. Returns [B,H,W] in [0,1]."""
    store = {}

    def hook(_m, _inp, out):
        out.retain_grad()
        store["a"] = out

    h = model.g5.register_forward_hook(hook)
    try:
        for p in model.parameters():
            p.requires_grad_(True)       # hooks need a graph; weights are not updated
        logits = model(x)
        model.zero_grad()
        logits[:, cls].sum().backward()
        a = store["a"]
        w = a.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * a).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                                              align_corners=False)[:, 0]
        mx = cam.flatten(1).max(1)[0].clamp_min(1e-12).view(-1, 1, 1)
        out = (cam / mx).detach()
    finally:
        h.remove()
        for p in model.parameters():
            p.requires_grad_(False)
    return out


def pick_examples(model, seed=0):
    df = filter_manifest(load_manifest(), "image", "test")
    st = df[(df.label == "stego") & (df.payload_rate == 0.4)]
    picks = []
    for method in ("lsb", "lsb_matching", "dct_qim"):
        g = st[st.method == method].sample(frac=1, random_state=seed + len(picks) * 7)
        ds = ImageStegoDataset(g.head(30))
        for i in range(len(ds)):
            x = ds[i][0].unsqueeze(0)
            with torch.no_grad():
                if model(x).argmax(1).item() == 1:
                    picks.append((method, g.iloc[i], x))
                    break
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    model = load_detector("image_cnn")
    ex = pick_examples(model, args.seed)
    fig, axes = plt.subplots(len(ex), 4, figsize=(12, 3.1 * len(ex)), facecolor=SURFACE)
    titles = ["stego image", "Grad-CAM, clean", f"perturbation |adv - stego| (eps={args.eps:g})",
              "Grad-CAM, adversarial"]
    for r, (method, row, x) in enumerate(ex):
        x_adv = A.pgd(model, x, args.eps, A.IMAGE_UNITS, steps=args.steps)
        with torch.no_grad():
            p0 = torch.softmax(model(x), 1)[0, 1].item()
            p1 = torch.softmax(model(x_adv), 1)[0, 1].item()
        cam0 = grad_cam(model, x)[0].numpy()
        cam1 = grad_cam(model, x_adv)[0].numpy()
        diff = ((x_adv - x).abs() * 255)[0, 0].numpy()
        changed = float((diff >= 0.5).mean())
        panels = [(x[0, 0].numpy(), "gray", (0, 1)), (cam0, "Blues", (0, 1)),
                  (diff, "Blues", (0, max(1.0, args.eps))), (cam1, "Blues", (0, 1))]
        for c, (img, cmap, (lo, hi)) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], color=INK, fontsize=9.5, loc="left")
        axes[r, 0].set_ylabel(method, color=INK, fontsize=10, fontweight="bold")
        axes[r, 1].set_xlabel(f"P(stego) = {p0:.2f}", color=INK2, fontsize=9)
        axes[r, 2].set_xlabel(f"{100 * changed:.0f}% of pixels changed", color=INK2, fontsize=9)
        axes[r, 3].set_xlabel(f"P(stego) = {p1:.2f}", color=INK2, fontsize=9)
    fig.suptitle("What the image detector attends to, before and after a 1-grey-level attack",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig_gradcam.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print("saved", path)


if __name__ == "__main__":
    main()
"""
Phase 6b - defense: adversarial fine-tuning of the image detector.

Standard adversarial training (Madry et al.), adapted to this threat model. The
attacker only perturbs stego files, so for every batch we

  1. take the stego samples, craft PGD examples against the CURRENT weights that
     try to make the detector say "cover" (BatchNorm in eval mode for the attack),
  2. keep their label as "stego",
  3. update on [clean batch + adversarial stego], BatchNorm in train mode.

The detector is then asked to call a stego file stego even after the perturbation
that fooled its previous self. Fine-tuning starts from the Phase 3 checkpoint.

IMPORTANT about scale: robust training is expensive (each step costs roughly
(pgd_steps + 2)x a normal step) and normally needs many epochs. The default here
is a small CPU-sized budget. If the result is only a partial recovery, that is a
statement about the budget as much as about the defense; run with larger
--steps on a GPU (see the Colab notes) for the full-strength version.

    python3 adv_train.py --steps 80 --eps 2
"""

import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import attacks as A
from dataset_loader import load_manifest, filter_manifest, ImageStegoDataset
from limit_analysis import load_detector, build_attack_set, predict
from models import ImageStegoCNN

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_ROOT, "models")
OUT = os.path.join(_ROOT, "results", "phase5")


def robust_eval(model, data, eps_list, steps=8, bs=24):
    """Clean balanced accuracy and balanced accuracy under white-box PGD, using the
    same 'attack only what is detected' protocol as limit_analysis."""
    model.eval()
    cover_recall = float((predict(model, data["x_cover"], bs) == 0).float().mean())
    clean = predict(model, data["x_stego"], bs)
    det = clean == 1
    n_all, x = len(data["x_stego"]), data["x_stego"][det]
    out = {"clean_balanced": 0.5 * (cover_recall + float(det.float().mean())),
           "cover_recall": cover_recall, "clean_tpr": float(det.float().mean())}
    for eps in eps_list:
        adv = torch.cat([A.pgd(model, x[i:i + bs], eps, A.IMAGE_UNITS, steps=steps)
                         for i in range(0, len(x), bs)]) if len(x) else x
        still = int((predict(model, adv, bs) == 1).sum()) if len(x) else 0
        out[f"pgd_eps{eps}_balanced"] = 0.5 * (cover_recall + still / n_all)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80, help="optimizer steps")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eps", type=float, default=2.0, help="training attack budget, grey levels")
    ap.add_argument("--pgd-steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=40)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = ImageStegoCNN()
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "image_cnn_allmethods.pt"),
                                     map_location="cpu"))

    train = filter_manifest(load_manifest(), "image", "train")
    labels = (train.label != "cover").astype(int).values
    w = (1.0 / np.bincount(labels))[labels]
    sampler = WeightedRandomSampler(w, num_samples=args.steps * args.batch_size, replacement=True)
    loader = DataLoader(ImageStegoDataset(train), batch_size=args.batch_size, sampler=sampler)

    val = build_attack_set("image", 16, 72, seed=100)    # disjoint seed from the sweeps
    eval_eps = [1, 2, 4]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    for p in model.parameters():
        p.requires_grad_(True)
    hist = []
    base = robust_eval(model, val, eval_eps)
    print("before:", {k: round(v, 3) for k, v in base.items()}, flush=True)
    hist.append(dict(step=0, **base))

    t0 = time.time()
    for step, (x, y, _) in enumerate(loader, start=1):
        stego = y == 1
        model.eval()
        x_adv = A.pgd(model, x[stego], args.eps, A.IMAGE_UNITS, steps=args.pgd_steps)
        model.train()
        xb = torch.cat([x, x_adv])
        yb = torch.cat([y, torch.ones(len(x_adv), dtype=torch.long)])
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"step {step}/{args.steps} loss={loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            r = robust_eval(model, val, eval_eps)
            print(f"  eval@{step}:", {k: round(v, 3) for k, v in r.items()}, flush=True)
            hist.append(dict(step=step, **r))

    for p in model.parameters():
        p.requires_grad_(False)
    os.makedirs(OUT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "image_cnn_advtrained.pt"))
    pd.DataFrame(hist).to_csv(os.path.join(OUT, "advtrain_history.csv"), index=False)
    print("saved models/image_cnn_advtrained.pt")


if __name__ == "__main__":
    main()
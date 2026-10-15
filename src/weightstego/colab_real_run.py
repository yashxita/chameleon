"""
Phase 8 - real-model run (Colab GPU).

Trains several real CIFAR-10 CNNs (so accuracy is a genuine measurement, not a synthetic
proxy), embeds a benign quantization-surviving payload (qim_int8), then measures:

  1. DETECTION on real weights: can the carrier-vs-control feature separate clean from stego
     on held-out checkpoints, above the quantizer-only baseline.
  2. SANITIZER UTILITY (the point of this run): the targeted half-step sanitizer removes the
     payload at ~zero accuracy cost. The blunt random-bit-flip baseline is swept up until it
     removes the payload as well, and we report the CIFAR accuracy it costs to get there.

Run on Colab with a GPU runtime:
    !pip -q install torch torchvision scikit-learn pandas
    !python colab_real_run.py --n-models 6 --epochs 3

Needs embed.py and features.py in the same directory.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from embed import (embed_qim_int8, decode_qim_int8, carrier_tensors,
                   _int8_scale, _select_int8_carriers, _payload_bits)
from features import model_features, split_columns

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(_ROOT, "results", "phase8")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def loaders(bs=256):
    tf = T.Compose([T.ToTensor()])
    root = os.path.join(_ROOT, "data", "cifar")
    tr = tv.datasets.CIFAR10(root, train=True, download=True, transform=tf)
    te = tv.datasets.CIFAR10(root, train=False, download=True, transform=tf)
    return (torch.utils.data.DataLoader(tr, bs, shuffle=True, num_workers=2),
            torch.utils.data.DataLoader(te, bs, num_workers=2))


class Net(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c, c * 2, 3, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c * 2, c * 2, 3, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4))
        self.h = nn.Linear(c * 2 * 16, 10)

    def forward(self, x):
        return self.h(self.f(x).flatten(1))


def train_one(seed, tr, epochs):
    torch.manual_seed(seed)
    m = Net().to(DEV)
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    lf = nn.CrossEntropyLoss()
    m.train()
    for _ in range(epochs):
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad(); lf(m(x), y).backward(); opt.step()
    return {k: v.cpu() for k, v in m.state_dict().items()}


@torch.no_grad()
def accuracy(sd, te):
    m = Net().to(DEV); m.load_state_dict(sd); m.eval()
    ok = n = 0
    for x, y in te:
        x = x.to(DEV)
        ok += (m(x).argmax(1).cpu() == y).sum().item(); n += len(y)
    return ok / n


def fake_int8(sd, min_numel=4096):
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sd.items()}
    for k in carrier_tensors(out, min_numel):
        t = out[k].float(); s = _int8_scale(t)
        out[k] = (torch.round(t / s).clamp(-127, 127) * s).to(out[k].dtype)
    return out


def ber(sd, bits, bpt):
    r = decode_qim_int8(sd, bits_per_tensor=bpt)
    k = min(len(bits), len(r))
    return float((bits[:k] != r[:k]).mean())


def sanitize_halfstep(sd, bpt, min_numel=4096):
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sd.items()}
    for k in carrier_tensors(out, min_numel):
        flat = out[k].reshape(-1).float(); s = _int8_scale(flat)
        codes = torch.round(flat / s).clamp(-127, 127)
        idx = _select_int8_carriers(codes, bpt)
        v = codes[idx]
        codes[idx] = (v + torch.where(v >= 0, torch.ones_like(v), -torch.ones_like(v))).clamp(-127, 127)
        out[k] = (codes * s).reshape(out[k].shape).to(out[k].dtype)
    return out


def sanitize_randflip(sd, frac, min_numel=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sd.items()}
    for k in carrier_tensors(out, min_numel):
        flat = out[k].reshape(-1).float(); s = _int8_scale(flat)
        codes = torch.round(flat / s).clamp(-127, 127)
        mask = torch.rand(codes.numel(), generator=g) < frac
        step = torch.where(codes >= 0, torch.ones_like(codes), -torch.ones_like(codes))
        codes[mask] = (codes[mask] + step[mask]).clamp(-127, 127)
        out[k] = (codes * s).reshape(out[k].shape).to(out[k].dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-models", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bits", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tr, te = loaders()

    base = [train_one(100 + i, tr, args.epochs) for i in range(args.n_models)]
    nt = len(carrier_tensors(base[0]))
    payload = _payload_bits(args.bits * nt, 7)

    # ---- detection on real weights (held-out checkpoint) ----
    feat, lab, split = [], [], []
    cut = max(1, int(0.6 * len(base)))
    for i, sd in enumerate(base):
        for variant, s in [("clean", sd), ("clean_int8", fake_int8(sd)),
                            ("stego", embed_qim_int8(sd, bits_per_tensor=args.bits, seed=7)),
                            ("stego_int8", fake_int8(embed_qim_int8(sd, bits_per_tensor=args.bits, seed=7)))]:
            feat.append(model_features(s, topk=args.bits))
            lab.append(0 if variant.startswith("clean") else 1)
            split.append("train" if i < cut else "test")
    F = pd.DataFrame(feat); F["label"] = lab; F["split"] = split
    base_cols, full_cols = split_columns([c for c in F.columns if c not in ("label", "split")])
    det = []
    for view, cols in [("baseline", base_cols), ("full", full_cols)]:
        clf = RandomForestClassifier(300, class_weight="balanced", random_state=0)
        trs = F[F.split == "train"]; tes = F[F.split == "test"]
        clf.fit(trs[cols], trs.label)
        p = clf.predict_proba(tes[cols])[:, 1]
        det.append(dict(view=view,
                        balanced=round(balanced_accuracy_score(tes.label, (p >= 0.5).astype(int)), 3),
                        auc=round(roc_auc_score(tes.label, p), 3)))
    pd.DataFrame(det).to_csv(os.path.join(OUT, "real_detect.csv"), index=False)
    print("DETECTION (held-out checkpoint):"); print(pd.DataFrame(det).to_string(index=False))

    # ---- sanitizer utility: real CIFAR accuracy vs payload removal ----
    rows = []
    for i, sd in enumerate(base):
        stego = embed_qim_int8(sd, bits_per_tensor=args.bits, seed=7)
        acc_stego = accuracy(stego, te)
        hs = sanitize_halfstep(stego, args.bits)
        rows.append(dict(model=i, method="halfstep", frac=np.nan,
                         ber=round(ber(hs, payload, args.bits), 3),
                         acc=round(accuracy(hs, te), 3), acc_drop=round(acc_stego - accuracy(hs, te), 3)))
        # sweep random-bit-flip until it removes the payload as well (BER >= 0.9)
        for frac in [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]:
            rf = sanitize_randflip(stego, frac)
            b = ber(rf, payload, args.bits); a = accuracy(rf, te)
            rows.append(dict(model=i, method="randflip", frac=frac, ber=round(b, 3),
                             acc=round(a, 3), acc_drop=round(acc_stego - a, 3)))
            if b >= 0.9:
                break
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "real_sanitize.csv"), index=False)
    print("\nSANITIZER (mean over models):")
    hs = R[R.method == "halfstep"]
    rf_kill = R[(R.method == "randflip") & (R.ber >= 0.9)].groupby("model").first()
    print(f"  halfstep         : BER {hs.ber.mean():.2f}  acc_drop {hs.acc_drop.mean():.3f}")
    if len(rf_kill):
        print(f"  randflip to kill : BER {rf_kill.ber.mean():.2f}  acc_drop {rf_kill.acc_drop.mean():.3f}  "
              f"(needs frac ~{rf_kill.frac.mean():.2f})")
    else:
        print("  randflip never reached BER>=0.9 within the swept fractions")


if __name__ == "__main__":
    main()

"""
Phase 8 - build the model zoo (the Chameleon "dataset" step, for weights).

Produces many INDEPENDENT base checkpoints, then for each one writes a clean copy and a
stego copy per embedding method, optionally quantized the same way. A manifest records
everything and the split is by BASE checkpoint, so train and test never share a base model
(the review's key control: thousands of weights from one model are not independent samples).

Two sources:
  synthetic  : trains small CNNs from different seeds on a synthetic task. Fully offline,
               for the sandbox smoke test.
  torchvision: loads several real pretrained small models. Use on Colab (needs download).

Quantization ("fake int8", per-tensor symmetric round-to-grid then back to float) is applied
to BOTH clean and stego variants, so the detector cannot win just by spotting quantization.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from embed import embed, carrier_tensors   # noqa: E402  (run from src/weightstego)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(_ROOT, "results", "phase8", "zoo")


class SmallCNN(nn.Module):
    def __init__(self, c=32, nclass=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c, c * 2, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Linear(c * 2 * 16, nclass)

    def forward(self, x):
        return self.head(self.net(x).flatten(1))


def _synth_batch(rng, n=128):
    y = rng.integers(0, 4, size=n)
    x = rng.normal(0, 1, size=(n, 3, 16, 16)).astype(np.float32)
    for i in range(n):            # give each class a faint mean shift so training has signal
        x[i, y[i] % 3] += 0.5
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.int64))


def train_one(seed, steps=120):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = SmallCNN()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss()
    m.train()
    for _ in range(steps):
        x, y = _synth_batch(rng)
        opt.zero_grad(); loss = lossf(m(x), y); loss.backward(); opt.step()
    return m.state_dict()


def fake_int8(state_dict, min_numel=4096):
    sd = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state_dict.items()}
    for k in carrier_tensors(sd, min_numel):
        t = sd[k].to(torch.float32)
        scale = t.abs().max().clamp_min(1e-12) / 127.0
        sd[k] = (torch.round(t / scale).clamp(-127, 127) * scale).to(sd[k].dtype)
    return sd


def _save(sd, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(sd, path)


def build(args):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    bases = []
    if args.source == "synthetic":
        for i in range(args.n_models):
            sd = train_one(seed=1000 + i, steps=args.steps)
            bases.append((f"synth{i:03d}", sd))
    else:
        import torchvision.models as tvm
        names = ["resnet18", "mobilenet_v2", "shufflenet_v2_x1_0", "efficientnet_b0",
                 "resnet34", "densenet121", "mnasnet1_0", "regnet_y_400mf"]
        for nm in names[:args.n_models]:
            sd = getattr(tvm, nm)(weights="DEFAULT").state_dict()
            bases.append((nm, sd))

    methods = ["lsb", "qim_parity", "qim_int8"]
    for base_id, sd in bases:
        variants = [("clean", None, sd)]
        for mth in methods:
            variants.append((mth, mth, embed(sd, mth, bits_per_tensor=args.bits, seed=7)))
        for quant in (["none", "int8"] if args.quant else ["none"]):
            for vname, method, vsd in variants:
                out = fake_int8(vsd) if quant == "int8" else vsd
                path = os.path.join(OUT, f"{base_id}__{vname}__{quant}.pt")
                _save(out, path)
                rows.append(dict(base_id=base_id, variant=vname,
                                 method=(method or "none"), quant=quant,
                                 label=0 if vname == "clean" else 1,
                                 path=os.path.relpath(path, _ROOT)))
    df = pd.DataFrame(rows)
    # split by BASE checkpoint
    rng = np.random.default_rng(0)
    uniq = sorted(df.base_id.unique())
    rng.shuffle(uniq)
    cut = max(1, int(0.6 * len(uniq)))
    train_bases = set(uniq[:cut])
    df["split"] = np.where(df.base_id.isin(train_bases), "train", "test")
    df.to_csv(os.path.join(OUT, "manifest.csv"), index=False)
    print(f"wrote {len(df)} model files from {len(uniq)} bases "
          f"({cut} train / {len(uniq)-cut} test) to {OUT}")
    print(df.groupby(['split','variant','quant']).size())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "torchvision"], default="synthetic")
    ap.add_argument("--n-models", type=int, default=12)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--bits", type=int, default=256)
    ap.add_argument("--quant", action="store_true")
    build(ap.parse_args())

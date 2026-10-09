"""
Phase 4 - cross-modal transfer: does what a detector learns about hidden data in
images carry over to audio, and the other way round?

Design
------
ONE convolutional trunk (the ImageStegoCNN conv stack: ABS, tanh/ReLU, average
pooling, classifier) fed by two modality-specific front ends that contain NO
learned parameters:

  image  x[B,1,256,256] -> KV high-pass (x255) -> TLU(3)
  audio  x[B,1,32000]   -> second difference (x32768) -> per-clip standardisation
                           -> TLU(3) -> reshape to [B,1,160,200]

Because the front ends are parameter-free, a trunk trained on images has seen no
audio-fitted weights at all, so "zero-shot" transfer is genuinely zero-shot.

Why the per-clip standardisation: the audio second difference is in int16 units
and for a full-scale waveform sampled at 16 kHz it is in the thousands, so a
TLU at +-3 would clamp every sample to +-3 and leave a constant signal. Dividing
by the clip's own residual std (a fixed, parameter-free operation) puts the audio
residual on the same scale as the image residual before the clamp. This is the
one addition to the design in the README and is stated here so it is not silent.

Conditions (each reports the TARGET modality's test metrics)
-------------------------------------------------------------
  image_only / audio_only      trunk trained on one modality, tested on it (controls)
  zeroshot_<src>_to_<dst>      the source-trained trunk tested, untouched, on dst
  bnrecal_<src>_to_<dst>       as above after re-estimating ONLY the BatchNorm running
                               statistics on unlabeled dst training data
  finetune_<src>_to_<dst>      source-trained trunk fully fine-tuned on dst labels
  joint                        one trunk trained on both modalities, tested on each

    python3 cross_modal.py --device cuda --epochs 12
    python3 cross_modal.py --smoke        # CPU plumbing check
"""

import os
import copy
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

from dataset_loader import load_manifest, filter_manifest, ImageStegoDataset, AudioStegoDataset
from models import ImageStegoCNN, AudioHighPassResidual, TLU

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "results", "phase4")
AUDIO_SHAPE = (160, 200)          # 160 * 200 = 32000 samples


class CrossModalCNN(ImageStegoCNN):
    """ImageStegoCNN's trunk with a second, parameter-free audio front end."""

    def __init__(self, tlu_threshold=3.0):
        super().__init__(tlu_threshold)
        self.audio_hpf = AudioHighPassResidual()

    def front(self, x, modality):
        if modality == "image":
            return self.tlu(self.hpf(x))
        r = self.audio_hpf(x)                                     # [B,1,T]
        r = r / (r.std(dim=-1, keepdim=True) + 1e-6)              # per-clip standardisation
        r = self.tlu(r)
        return r.view(x.shape[0], 1, *AUDIO_SHAPE)

    def forward(self, x, modality="image"):
        x = self.front(x, modality)
        x = self.g5(self.g4(self.g3(self.g2(self.g1(x)))))
        return self.classifier(self.pool(x).flatten(1))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_loader(modality, split, bs, train, max_samples=None, seed=0, num_workers=0):
    df = filter_manifest(load_manifest(), modality, split)
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=seed).reset_index(drop=True)
    DS = ImageStegoDataset if modality == "image" else AudioStegoDataset
    ds = DS(df)
    if train:
        lab = (df.label != "cover").astype(int).values
        w = (1.0 / np.bincount(lab))[lab]
        sampler = WeightedRandomSampler(w, num_samples=len(df), replacement=True)
        return DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=num_workers)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, modality, loader, device):
    model.eval()
    logits, ys, ms = [], [], []
    for x, y, m in loader:
        logits.append(model(x.to(device), modality).cpu())
        ys.append(y)
        ms += list(m)
    lg, y, ms = torch.cat(logits), torch.cat(ys).numpy(), np.array(ms)
    p = lg.argmax(1).numpy()
    s = torch.softmax(lg, 1)[:, 1].numpy()
    cover = float((p[y == 0] == 0).mean())
    stego = float((p[y == 1] == 1).mean())
    out = dict(balanced=0.5 * (cover + stego), cover_recall=cover, stego_recall=stego,
               auc=float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan"))
    for m in sorted(set(ms) - {"none"}):
        out[f"recall_{m}"] = float((p[(ms == m) & (y == 1)] == 1).mean())
    return out


def train_model(model, sources, val_loaders, epochs, lr, device, log, tag):
    """sources: list of (modality, train_loader). Steps alternate across modalities
    so a joint model sees both in equal measure. Keeps the checkpoint with the best
    mean validation balanced accuracy over val_loaders."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.5)
    crit = nn.CrossEntropyLoss()
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        its = [iter(l) for _, l in sources]
        n_steps = min(len(l) for _, l in sources)
        loss_sum = 0.0
        for _ in range(n_steps):
            for (mod, _), it in zip(sources, its):
                x, y, _m = next(it)
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = crit(model(x, mod), y)
                loss.backward()
                opt.step()
                loss_sum += loss.item()
        sched.step()
        vals = {m: evaluate(model, m, l, device) for m, l in val_loaders.items()}
        score = float(np.mean([v["balanced"] for v in vals.values()]))
        log(f"  [{tag}] epoch {ep}/{epochs} loss={loss_sum / max(1, n_steps * len(sources)):.4f} "
            + " ".join(f"val_{m}={v['balanced']:.3f}" for m, v in vals.items()))
        if score > best:
            best, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def recalibrate_bn(model, modality, loader, device, max_batches=40):
    """Re-estimate ONLY the BatchNorm running statistics on (unlabeled) target data."""
    model = copy.deepcopy(model)
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None                    # cumulative average
    model.train()
    for i, (x, _y, _m) in enumerate(loader):
        if i >= max_batches:
            break
        model(x.to(device), modality)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--finetune-epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--finetune-lr", type=float, default=5e-5)
    ap.add_argument("--max-train-samples", type=int, default=6000)
    ap.add_argument("--max-eval-samples", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.finetune_epochs, args.batch_size = 1, 1, 8
        args.max_train_samples, args.max_eval_samples = 64, 64

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    os.makedirs(OUT, exist_ok=True)
    log = lambda s: print(s, flush=True)
    bs = args.batch_size

    tr = {m: make_loader(m, "train", bs, True, args.max_train_samples, args.seed) for m in ("image", "audio")}
    va = {m: make_loader(m, "val", bs, False, args.max_eval_samples, args.seed) for m in ("image", "audio")}
    te = {m: make_loader(m, "test", bs, False, args.max_eval_samples, args.seed) for m in ("image", "audio")}
    unl = {m: make_loader(m, "train", bs, False, 40 * bs, args.seed) for m in ("image", "audio")}

    rows = []

    def record(cond, target, model):
        r = evaluate(model, target, te[target], dev)
        rows.append(dict(condition=cond, target=target, **r))
        log(f"{cond:28s} -> {target:5s} balanced={r['balanced']:.3f} auc={r['auc']:.3f} "
            f"cover={r['cover_recall']:.3f} stego={r['stego_recall']:.3f}")
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "cross_modal_results.csv"), index=False)

    # --- single-modality controls ------------------------------------------------
    models = {}
    for mod in ("image", "audio"):
        log(f"\n== training {mod}-only trunk ==")
        m = CrossModalCNN().to(dev)
        m = train_model(m, [(mod, tr[mod])], {mod: va[mod]}, args.epochs, args.lr, dev, log, f"{mod}_only")
        models[mod] = m
        record(f"{mod}_only", mod, m)

    # --- transfer in both directions --------------------------------------------
    for src, dst in (("image", "audio"), ("audio", "image")):
        log(f"\n== transfer {src} -> {dst} ==")
        record(f"zeroshot_{src}_to_{dst}", dst, models[src])
        record(f"bnrecal_{src}_to_{dst}", dst, recalibrate_bn(models[src], dst, unl[dst], dev))
        ft = copy.deepcopy(models[src])
        ft = train_model(ft, [(dst, tr[dst])], {dst: va[dst]}, args.finetune_epochs,
                         args.finetune_lr, dev, log, f"finetune_{src}_to_{dst}")
        record(f"finetune_{src}_to_{dst}", dst, ft)

    # --- joint --------------------------------------------------------------------
    log("\n== joint training ==")
    jm = CrossModalCNN().to(dev)
    jm = train_model(jm, [("image", tr["image"]), ("audio", tr["audio"])], va, args.epochs,
                     args.lr, dev, log, "joint")
    for mod in ("image", "audio"):
        record("joint", mod, jm)

    log("\nsaved " + os.path.join(OUT, "cross_modal_results.csv"))


if __name__ == "__main__":
    main()
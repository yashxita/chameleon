"""
Phase 5 - surrogate image detector for the black-box transfer attack.

A realistic attacker does not have the defender's weights. They train their own
detector on whatever stego/cover data they can make, craft adversarial files
against THAT, and hope the result transfers. So the surrogate is deliberately a
different architecture from ImageStegoCNN (no ABS, no TLU, ReLU + max pooling,
different widths) and is trained on a different random subset of the training
split with a different seed.

    python3 surrogate.py --epochs 6 --max-train-samples 3000
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset_loader import load_manifest, filter_manifest, ImageStegoDataset
from models import HighPassResidual

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_ROOT, "models")
RESULTS_DIR = os.path.join(_ROOT, "results", "phase5")


class SurrogateImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hpf = HighPassResidual()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 12), block(12, 24), block(24, 48),
                                      block(48, 96), block(96, 96))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(96, 48), nn.ReLU(inplace=True),
                                        nn.Dropout(0.2), nn.Linear(48, 2))

    def forward(self, x):
        x = self.hpf(x)
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(x)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    for x, y, _ in loader:
        preds.append(model(x).argmax(1))
        labels.append(y)
    p, y = torch.cat(preds), torch.cat(labels)
    cover = float((p[y == 0] == 0).float().mean())
    stego = float((p[y == 1] == 1).float().mean())
    return 0.5 * (cover + stego), cover, stego


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-train-samples", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    df = load_manifest()
    train = filter_manifest(df, "image", "train")
    train = train.sample(min(args.max_train_samples, len(train)), random_state=args.seed)
    val = filter_manifest(df, "image", "val").sample(600, random_state=0)

    labels = (train.label != "cover").astype(int).values
    w = (1.0 / np.bincount(labels))[labels]
    sampler = WeightedRandomSampler(w, num_samples=len(train), replacement=True)
    tl = DataLoader(ImageStegoDataset(train), batch_size=args.batch_size, sampler=sampler)
    vl = DataLoader(ImageStegoDataset(val), batch_size=args.batch_size)

    model = SurrogateImageCNN()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best, best_state, hist = -1.0, None, []
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y, _ in tl:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            n += 1
        bal, cov, ste = evaluate(model, vl)
        print(f"epoch {ep}/{args.epochs}  loss={loss_sum / n:.4f}  val_bal={bal:.3f} "
              f"cover={cov:.3f} stego={ste:.3f}", flush=True)
        hist.append(dict(epoch=ep, loss=loss_sum / n, val_balanced=bal,
                         val_cover_recall=cov, val_stego_recall=ste))
        if bal > best:
            best, best_state = bal, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "surrogate_image_cnn.pt"))
    pd.DataFrame(hist).to_csv(os.path.join(RESULTS_DIR, "surrogate_train_history.csv"), index=False)
    print(f"saved surrogate (best val balanced acc {best:.3f})")


if __name__ == "__main__":
    main()
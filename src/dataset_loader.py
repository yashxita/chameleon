"""
Phase 3 - Dataset loaders.

Reads results/manifest.csv (built in Phase 2) and exposes PyTorch Dataset
classes for images and audio. Supports filtering by embedding method,
which is what lets us do the held-out-method "blindness" test: train on
methods A+B, evaluate on method C which the model has never seen.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import soundfile as sf

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(_PROJECT_ROOT, "results", "manifest.csv")


def load_manifest():
    df = pd.read_csv(MANIFEST_PATH)
    return df


def filter_manifest(df, modality, split, methods=None, exclude_methods=None):
    """methods: list of method names to include (covers are always included
    regardless, since 'cover' has method == 'none').
    exclude_methods: list of method names to exclude (for held-out tests)."""
    sub = df[(df.modality == modality) & (df.split == split)].copy()
    if methods is not None:
        sub = sub[(sub.method == "none") | (sub.method.isin(methods))]
    if exclude_methods is not None:
        sub = sub[~sub.method.isin(exclude_methods)]
    return sub.reset_index(drop=True)


class ImageStegoDataset(Dataset):
    """Returns (tensor[1,H,W] float in [0,1], label int 0=cover/1=stego, method str)."""

    def __init__(self, manifest_df):
        self.df = manifest_df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row.path), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        label = 0 if row.label == "cover" else 1
        return tensor, label, row.method


class AudioStegoDataset(Dataset):
    """Returns (tensor[1, T] float, label int 0=cover/1=stego, method str).
    Signals are zero-padded/truncated to a fixed length so batches can
    stack (all our generated clips are already the same length, but this
    guards against edge cases)."""

    def __init__(self, manifest_df, fixed_len=32000):
        self.df = manifest_df.reset_index(drop=True)
        self.fixed_len = fixed_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig, sr = sf.read(row.path, dtype="float32")
        if len(sig) < self.fixed_len:
            sig = np.pad(sig, (0, self.fixed_len - len(sig)))
        elif len(sig) > self.fixed_len:
            sig = sig[:self.fixed_len]
        tensor = torch.from_numpy(sig).unsqueeze(0)  # [1, T]
        label = 0 if row.label == "cover" else 1
        return tensor, label, row.method


if __name__ == "__main__":
    df = load_manifest()
    print(f"Manifest loaded: {len(df)} rows")

    train_img = filter_manifest(df, "image", "train")
    print(f"Image train (all methods): {len(train_img)} samples, "
          f"{(train_img.label == 'cover').sum()} cover / {(train_img.label == 'stego').sum()} stego")

    held_out = filter_manifest(df, "image", "test", exclude_methods=["dct_qim"])
    print(f"Image test excluding dct_qim: {len(held_out)} samples, methods present: "
          f"{sorted(held_out.method.unique())}")

    ds = ImageStegoDataset(train_img)
    x, y, m = ds[0]
    print(f"Sample 0: tensor shape {x.shape}, label {y}, method '{m}'")

    train_aud = filter_manifest(df, "audio", "train")
    print(f"\nAudio train (all methods): {len(train_aud)} samples")
    ads = AudioStegoDataset(train_aud)
    x, y, m = ads[0]
    print(f"Sample 0: tensor shape {x.shape}, label {y}, method '{m}'")
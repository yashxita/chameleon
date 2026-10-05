"""
Phase 3 - Dataset loaders.

Reads results/manifest.csv (built in Phase 2) and exposes PyTorch Dataset
classes for images and audio. Supports filtering by embedding method,
which is what lets us do the held-out-method "blindness" test: train on
methods A+B, evaluate on method C which the model has never seen.

Path portability
----------------
build_dataset.py writes ABSOLUTE paths into manifest.csv. Those paths were
correct on the machine that generated the dataset and nowhere else, which
breaks two things: copying or moving the project directory makes every row
silently point back at the original location (so you train on the old
copy's files without noticing), and nobody else can run the project at all
without regenerating the entire dataset first -- which contradicts the
reproducibility the README promises.

Manifest rows are always <project root>/data/..., so we re-root them here
against this checkout's own project root. Doing it in load_manifest() fixes
every consumer at once: both CNN datasets and classical_baseline.py.
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


def _reroot_path(stored_path):
    """Re-root one absolute manifest path onto THIS checkout.

    Stored paths look like
        C:\\...\\chameleon-main\\data/covers/images\\cover_00000.png
    with mixed separators, because build_dataset.py joins a literal
    "data/covers/images" onto an OS-native root. We keep everything from
    the last "data/" segment onward and re-join it to _PROJECT_ROOT, so one
    manifest works in any checkout on any OS.
    """
    p = str(stored_path).replace("\\", "/")
    idx = p.rfind("/data/")
    if idx == -1:
        # Not a layout we recognise. A relative path we can still resolve;
        # anything else we leave alone rather than guess at.
        if os.path.isabs(p):
            return stored_path
        return os.path.normpath(os.path.join(_PROJECT_ROOT, p))
    rel = p[idx + 1:]  # "data/covers/images/cover_00000.png"
    return os.path.normpath(os.path.join(_PROJECT_ROOT, rel))


def load_manifest(check_exists=True):
    """Load the manifest with every path re-rooted onto this checkout.

    check_exists catches the failure this function exists to prevent: a
    manifest that loads fine but points at files that aren't here. Better a
    loud error up front than a training run that dies 40 minutes in, or
    worse, quietly reads another copy of the project.
    """
    df = pd.read_csv(MANIFEST_PATH)
    df["path"] = df["path"].map(_reroot_path)

    if check_exists:
        missing = df.loc[~df["path"].map(os.path.exists)]
        if len(missing):
            raise FileNotFoundError(
                f"{len(missing)} of {len(df)} manifest files are missing from this checkout.\n"
                f"  project root : {_PROJECT_ROOT}\n"
                f"  first missing: {missing.iloc[0]['path']}\n"
                f"Regenerate the dataset with:\n"
                f"  python data_generation.py && python build_dataset.py"
            )
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
    print(f"Project root: {_PROJECT_ROOT}")
    df = load_manifest()
    print(f"Manifest loaded: {len(df)} rows, all files present")
    print(f"Example resolved path: {df.iloc[0]['path']}")

    train_img = filter_manifest(df, "image", "train")
    print(f"\nImage train (all methods): {len(train_img)} samples, "
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

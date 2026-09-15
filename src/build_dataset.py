"""
Phase 2 - Full dataset assembly.

For each cover (image or audio), we generate stego versions using every
embedding method at every payload rate. We then build a manifest CSV
(label = cover/stego, method, rate, split) and enforce a strict
cover-level train/val/test split so that no cover (or its stego variants)
crosses split boundaries. This matters for steganalysis specifically:
if a cover's stego version is in train and the same cover (clean) is in
test, the model can partly "recognize the image" rather than learn real
stego statistics, inflating results.
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import soundfile as sf

import os as _os
from embedding_image import EMBED_METHODS, psnr
from embedding_audio import EMBED_METHODS_AUDIO, snr_db

# paths are relative to this script's location (src/), so this works no
# matter where the repo is cloned
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
IMG_COVER_DIR = _os.path.join(_PROJECT_ROOT, "data/covers/images")
IMG_STEGO_DIR = _os.path.join(_PROJECT_ROOT, "data/stego/images")
AUD_COVER_DIR = _os.path.join(_PROJECT_ROOT, "data/covers/audio")
AUD_STEGO_DIR = _os.path.join(_PROJECT_ROOT, "data/stego/audio")
RESULTS_DIR = _os.path.join(_PROJECT_ROOT, "results")

PAYLOAD_RATES_IMAGE = [0.1, 0.2, 0.4]
PAYLOAD_RATES_AUDIO = [0.05, 0.1, 0.2]

SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
SEED = 7


def _assign_splits(n_items, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_items)
    n_train = int(n_items * SPLIT_RATIOS["train"])
    n_val = int(n_items * SPLIT_RATIOS["val"])
    splits = np.empty(n_items, dtype=object)
    splits[idx[:n_train]] = "train"
    splits[idx[n_train:n_train + n_val]] = "val"
    splits[idx[n_train + n_val:]] = "test"
    return splits


def build_image_dataset():
    os.makedirs(IMG_STEGO_DIR, exist_ok=True)
    cover_files = sorted(glob.glob(os.path.join(IMG_COVER_DIR, "*.png")))
    n = len(cover_files)
    splits = _assign_splits(n)

    rng = np.random.default_rng(SEED + 1)
    rows = []
    quality_log = []

    for cover_path, split in zip(cover_files, splits):
        cover_name = os.path.basename(cover_path)
        cover_img = np.array(Image.open(cover_path))

        rows.append({
            "modality": "image", "filename": cover_name, "path": cover_path,
            "label": "cover", "method": "none", "payload_rate": 0.0, "split": split,
        })

        for method_name, fn in EMBED_METHODS.items():
            for rate in PAYLOAD_RATES_IMAGE:
                stego_img = fn(cover_img, rate, rng)
                stego_name = f"{cover_name[:-4]}__{method_name}__r{rate}.png"
                stego_path = os.path.join(IMG_STEGO_DIR, stego_name)
                Image.fromarray(stego_img, mode="L").save(stego_path)

                q = psnr(cover_img, stego_img)
                quality_log.append({"modality": "image", "cover": cover_name,
                                     "method": method_name, "rate": rate, "quality_db": q})

                rows.append({
                    "modality": "image", "filename": stego_name, "path": stego_path,
                    "label": "stego", "method": method_name, "payload_rate": rate, "split": split,
                })

    return pd.DataFrame(rows), pd.DataFrame(quality_log)


def build_audio_dataset():
    os.makedirs(AUD_STEGO_DIR, exist_ok=True)
    cover_files = sorted(glob.glob(os.path.join(AUD_COVER_DIR, "*.wav")))
    n = len(cover_files)
    splits = _assign_splits(n, seed=SEED + 2)

    rng = np.random.default_rng(SEED + 3)
    rows = []
    quality_log = []

    for cover_path, split in zip(cover_files, splits):
        cover_name = os.path.basename(cover_path)
        cover_sig, sr = sf.read(cover_path)

        rows.append({
            "modality": "audio", "filename": cover_name, "path": cover_path,
            "label": "cover", "method": "none", "payload_rate": 0.0, "split": split,
        })

        for method_name, fn in EMBED_METHODS_AUDIO.items():
            for rate in PAYLOAD_RATES_AUDIO:
                if method_name == "echo_hiding":
                    stego_sig = fn(cover_sig, rate, rng, sr=sr)
                else:
                    stego_sig = fn(cover_sig, rate, rng)
                stego_name = f"{cover_name[:-4]}__{method_name}__r{rate}.wav"
                stego_path = os.path.join(AUD_STEGO_DIR, stego_name)
                sf.write(stego_path, stego_sig, sr, subtype="PCM_16")

                q = snr_db(cover_sig, stego_sig)
                quality_log.append({"modality": "audio", "cover": cover_name,
                                     "method": method_name, "rate": rate, "quality_db": q})

                rows.append({
                    "modality": "audio", "filename": stego_name, "path": stego_path,
                    "label": "stego", "method": method_name, "payload_rate": rate, "split": split,
                })

    return pd.DataFrame(rows), pd.DataFrame(quality_log)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Building image stego dataset...")
    img_df, img_quality = build_image_dataset()
    print(f"  {len(img_df)} total image entries "
          f"({(img_df.label=='cover').sum()} covers, {(img_df.label=='stego').sum()} stego)")

    print("Building audio stego dataset...")
    aud_df, aud_quality = build_audio_dataset()
    print(f"  {len(aud_df)} total audio entries "
          f"({(aud_df.label=='cover').sum()} covers, {(aud_df.label=='stego').sum()} stego)")

    full_df = pd.concat([img_df, aud_df], ignore_index=True)
    full_df.to_csv(os.path.join(RESULTS_DIR, "manifest.csv"), index=False)

    quality_df = pd.concat([img_quality, aud_quality], ignore_index=True)
    quality_df.to_csv(os.path.join(RESULTS_DIR, "embedding_quality.csv"), index=False)

    print("\nSplit distribution (image):")
    print(img_df[img_df.label == "cover"].split.value_counts())
    print("\nSplit distribution (audio):")
    print(aud_df[aud_df.label == "cover"].split.value_counts())

    print("\nQuality summary by method (dB, higher = more imperceptible):")
    print(quality_df.groupby(["modality", "method"]).quality_db.agg(["mean", "min", "max"]))

    print(f"\nManifest saved to {RESULTS_DIR}/manifest.csv")
    print(f"Quality log saved to {RESULTS_DIR}/embedding_quality.csv")

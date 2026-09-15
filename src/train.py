"""
Phase 3 - CNN training script.

Trains ImageStegoCNN / AudioSpectrogramCNN / AudioWaveformCNN and reports
accuracy broken down by embedding method on the test set, which is what
lets us check the "blind" claim: if `--exclude-method` is set, that method
never appears during training, so its test accuracy tells us how well the
model generalizes to a completely unseen embedding algorithm.

Usage examples:
    python3 train.py --modality image --model image_cnn --epochs 5
    python3 train.py --modality image --model image_cnn --epochs 5 --exclude-method dct_qim
    python3 train.py --modality audio --model audio_spectrogram --epochs 5
    python3 train.py --modality audio --model audio_waveform --epochs 5 --exclude-method echo_hiding
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset_loader import load_manifest, filter_manifest, ImageStegoDataset, AudioStegoDataset
from models import ImageStegoCNN, AudioSpectrogramCNN, AudioWaveformCNN

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")


def make_weighted_sampler(df):
    """Since covers are outnumbered ~9:1 by stego, use a weighted sampler
    so each training batch sees roughly balanced classes (undersampling
    the way classical_baseline.py does would throw away 90% of our
    generated stego data; a weighted sampler keeps all of it while still
    balancing what the model actually sees per batch)."""
    labels = (df.label != "cover").astype(int).values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(df), replacement=True)


def get_model(name):
    if name == "image_cnn":
        return ImageStegoCNN()
    if name == "audio_spectrogram":
        return AudioSpectrogramCNN()
    if name == "audio_waveform":
        return AudioWaveformCNN()
    raise ValueError(f"unknown model {name}")


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_methods = [], [], []
    with torch.no_grad():
        for x, y, methods in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            all_methods.extend(methods)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_methods = np.array(all_methods)

    overall_acc = (all_preds == all_labels).mean()

    # cover/stego are heavily imbalanced in val/test (covers are ~10% of
    # samples), so raw accuracy can look good while the model just
    # defaults to predicting "stego" for almost everything. Balanced
    # accuracy (average of the two classes' recall) is the honest metric
    # and is what we use for checkpoint selection.
    cover_mask = all_labels == 0
    stego_mask = all_labels == 1
    cover_recall = (all_preds[cover_mask] == all_labels[cover_mask]).mean() if cover_mask.sum() else float("nan")
    stego_recall = (all_preds[stego_mask] == all_labels[stego_mask]).mean() if stego_mask.sum() else float("nan")
    balanced_acc = np.nanmean([cover_recall, stego_recall])

    per_method = {}
    for m in sorted(set(all_methods)):
        mask = all_methods == m
        per_method[m] = (all_preds[mask] == all_labels[mask]).mean()

    metrics = {
        "overall_acc": overall_acc,
        "balanced_acc": balanced_acc,
        "cover_recall": cover_recall,
        "stego_recall": stego_recall,
    }
    return metrics, per_method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["image", "audio"], required=True)
    parser.add_argument("--model", choices=["image_cnn", "audio_spectrogram", "audio_waveform"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--exclude-method", type=str, default=None,
                         help="embedding method to hold out of training entirely, for the blindness test")
    parser.add_argument("--max-train-samples", type=int, default=None,
                         help="optional cap on training set size, for quick smoke tests")
    parser.add_argument("--tag", type=str, default=None,
                         help="run name for saved checkpoint/results, defaults to model+exclude info")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_manifest()
    exclude = [args.exclude_method] if args.exclude_method else None
    train_df = filter_manifest(df, args.modality, "train", exclude_methods=exclude)
    val_df = filter_manifest(df, args.modality, "val")  # validate on all methods
    test_df = filter_manifest(df, args.modality, "test")  # test on all methods (incl. held-out)

    if args.max_train_samples is not None and len(train_df) > args.max_train_samples:
        train_df = train_df.sample(args.max_train_samples, random_state=0).reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Excluded method (held out of training): {args.exclude_method}")

    DatasetCls = ImageStegoDataset if args.modality == "image" else AudioStegoDataset
    train_ds = DatasetCls(train_df)
    val_ds = DatasetCls(val_df)
    test_ds = DatasetCls(test_df)

    sampler = make_weighted_sampler(train_df)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = get_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_bal_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x, y, _methods in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        scheduler.step()

        train_loss = running_loss / max(1, n_batches)
        val_metrics, val_per_method = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_acc={val_metrics['overall_acc']:.4f}  "
              f"val_balanced_acc={val_metrics['balanced_acc']:.4f}  "
              f"val_cover_recall={val_metrics['cover_recall']:.4f}  "
              f"val_stego_recall={val_metrics['stego_recall']:.4f}  "
              f"val_per_method={ {k: round(v,3) for k,v in val_per_method.items()} }")
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        # checkpoint on BALANCED accuracy, not raw accuracy: with a ~90/10
        # stego/cover test split, raw accuracy rewards a model that just
        # defaults to predicting "stego" for almost everything
        if val_metrics["balanced_acc"] > best_val_bal_acc:
            best_val_bal_acc = val_metrics["balanced_acc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  -> new best (val_balanced_acc={val_metrics['balanced_acc']:.4f}), checkpointing")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best checkpoint (val_balanced_acc={best_val_bal_acc:.4f}) for final evaluation")

    test_metrics, test_per_method = evaluate(model, test_loader, device)
    print(f"\nFinal test accuracy: {test_metrics['overall_acc']:.4f}  "
          f"(balanced: {test_metrics['balanced_acc']:.4f})")
    print(f"  cover recall: {test_metrics['cover_recall']:.4f}  "
          f"stego recall: {test_metrics['stego_recall']:.4f}")
    for m, acc in sorted(test_per_method.items()):
        flag = "  <-- HELD OUT (blind generalization test)" if m == args.exclude_method else ""
        print(f"  {m:15s} acc={acc:.4f}{flag}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = args.tag or f"{args.model}_{'excl_' + args.exclude_method if args.exclude_method else 'allmethods'}"

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{tag}.pt"))

    pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, f"train_history_{tag}.csv"), index=False)

    test_rows = [{"tag": tag, "modality": args.modality, "model": args.model,
                   "excluded_method": args.exclude_method, "method": m, "accuracy": acc}
                  for m, acc in test_per_method.items()]
    test_rows.append({"tag": tag, "modality": args.modality, "model": args.model,
                        "excluded_method": args.exclude_method, "method": "OVERALL", "accuracy": test_metrics["overall_acc"]})
    test_rows.append({"tag": tag, "modality": args.modality, "model": args.model,
                        "excluded_method": args.exclude_method, "method": "BALANCED", "accuracy": test_metrics["balanced_acc"]})
    results_path = os.path.join(RESULTS_DIR, f"test_results_{tag}.csv")
    pd.DataFrame(test_rows).to_csv(results_path, index=False)
    print(f"\nSaved model to models/{tag}.pt")
    print(f"Saved results to results/test_results_{tag}.csv")


if __name__ == "__main__":
    main()
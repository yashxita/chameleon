"""
Phase 3 - CNN training script.

Trains ImageStegoCNN / AudioSpectrogramCNN / AudioWaveformCNN and reports
accuracy broken down by embedding method on the test set, which is what
lets us check the "blind" claim: if `--exclude-method` is set, that method
never appears during training, so its test accuracy tells us how well the
model generalizes to a completely unseen embedding algorithm.

Stability notes (why the defaults are what they are)
----------------------------------------------------
An earlier version of this script produced a model that scored 83.5%
overall test accuracy while getting only 8.3% of covers right, i.e. it had
collapsed to answering "stego" for almost everything and was being
rewarded for it by the ~90/10 stego/cover ratio in the test set. Its
validation accuracy also swung wildly between epochs (0.79 -> 0.16 -> 0.39
-> 0.19). Four things address that:

  1. Checkpoint selection uses BALANCED accuracy (the mean of cover recall
     and stego recall), so an always-stego model scores 0.5 and can never
     win the checkpoint.
  2. Default learning rate is 1e-4, not 1e-3. The steganalysis signal is a
     very low-amplitude residual; 1e-3 with Adam overshoots it and is the
     main source of the epoch-to-epoch thrashing.
  3. Gradient-norm clipping, for the same reason.
  4. BatchNorm momentum is lowered (--bn-momentum, default 0.01) so the
     running statistics used at eval time average over many more batches.
     At batch_size 16-32 with a signal this subtle, the PyTorch default of
     0.1 makes eval-time BN statistics noisy enough on its own to move
     validation accuracy by tens of points between epochs.

We also report AUC, which is threshold-independent and therefore the
honest headline number under this much class imbalance, and seed every
source of randomness (--seed) so runs reproduce exactly, matching the
guarantee Phase 2 data generation already makes.

Usage examples:
    python3 train.py --modality image --model image_cnn --epochs 20
    python3 train.py --modality image --model image_cnn --epochs 20 --exclude-method dct_qim
    python3 train.py --modality audio --model audio_spectrogram --epochs 20
    python3 train.py --modality audio --model audio_waveform --epochs 20 --exclude-method echo_hiding
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

from dataset_loader import load_manifest, filter_manifest, ImageStegoDataset, AudioStegoDataset
from models import ImageStegoCNN, AudioSpectrogramCNN, AudioWaveformCNN

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")


def set_seed(seed):
    """Seed every source of randomness the training loop touches. Phase 2
    already guarantees the dataset itself is reproducible; this extends the
    same guarantee to training, which also makes the instability we were
    chasing reproducible rather than a moving target."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_weighted_sampler(df, generator=None):
    """Since covers are outnumbered ~9:1 by stego, use a weighted sampler
    so each training batch sees roughly balanced classes (undersampling
    the way classical_baseline.py does would throw away 90% of our
    generated stego data; a weighted sampler keeps all of it while still
    balancing what the model actually sees per batch)."""
    labels = (df.label != "cover").astype(int).values
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)
    if (class_counts == 0).any():
        # only one class present -- nothing to balance, sample uniformly
        return None
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(df),
                                 replacement=True, generator=generator)


def get_model(name, bn_momentum=None):
    if name == "image_cnn":
        model = ImageStegoCNN()
    elif name == "audio_spectrogram":
        model = AudioSpectrogramCNN()
    elif name == "audio_waveform":
        model = AudioWaveformCNN()
    else:
        raise ValueError(f"unknown model {name}")

    # Lower BatchNorm momentum => eval-time running statistics are averaged
    # over many more batches => far less epoch-to-epoch eval noise. See the
    # stability notes at the top of this file.
    if bn_momentum is not None:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.momentum = bn_momentum
    return model


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels, all_methods = [], [], [], []
    with torch.no_grad():
        for x, y, methods in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            all_methods.extend(methods)

    all_probs = np.array(all_probs)
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

    # AUC is threshold-independent, so unlike accuracy it cannot be gamed
    # by a model that simply shifts its decision boundary toward the
    # majority class. Worth reporting alongside balanced accuracy.
    if cover_mask.sum() and stego_mask.sum():
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = float("nan")

    per_method = {}
    for m in sorted(set(all_methods)):
        mask = all_methods == m
        per_method[m] = (all_preds[mask] == all_labels[mask]).mean()

    metrics = {
        "overall_acc": overall_acc,
        "balanced_acc": balanced_acc,
        "cover_recall": cover_recall,
        "stego_recall": stego_recall,
        "auc": auc,
    }
    return metrics, per_method


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["image", "audio"], required=True)
    parser.add_argument("--model", choices=["image_cnn", "audio_spectrogram", "audio_waveform"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=6,
                        help="stop early if val balanced accuracy hasn't improved for this many epochs")
    parser.add_argument("--min-epochs", type=int, default=8,
                        help="never early-stop before this epoch; a collapsed model sits at exactly "
                             "0.5 balanced accuracy for a while before it starts learning, and "
                             "stopping during that window kills the run before it begins")
    parser.add_argument("--bn-momentum", type=float, default=0.01,
                        help="BatchNorm momentum; lower = more stable eval-time statistics")
    parser.add_argument("--grad-clip", type=float, default=5.0,
                        help="max gradient norm (0 disables clipping)")
    parser.add_argument("--num-workers", type=int, default=-1,
                        help="DataLoader worker processes; -1 auto-selects (2 on GPU to keep it fed, "
                             "0 on CPU where decoding is not the bottleneck)")
    parser.add_argument("--exclude-method", type=str, default=None,
                        help="embedding method to hold out of training entirely, for the blindness test")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="optional cap on training set size, for quick smoke tests")
    parser.add_argument("--tag", type=str, default=None,
                        help="run name for saved checkpoint/results, defaults to model+exclude info")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    df = load_manifest()
    exclude = [args.exclude_method] if args.exclude_method else None
    train_df = filter_manifest(df, args.modality, "train", exclude_methods=exclude)
    val_df = filter_manifest(df, args.modality, "val")   # validate on all methods
    test_df = filter_manifest(df, args.modality, "test")  # test on all methods (incl. held-out)

    if args.max_train_samples is not None and len(train_df) > args.max_train_samples:
        train_df = train_df.sample(args.max_train_samples, random_state=args.seed).reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Excluded method (held out of training): {args.exclude_method}")
    print(f"Methods seen in training: {sorted(train_df.method.unique())}")

    DatasetCls = ImageStegoDataset if args.modality == "image" else AudioStegoDataset
    train_ds = DatasetCls(train_df)
    val_ds = DatasetCls(val_df)
    test_ds = DatasetCls(test_df)

    # On GPU the model step is fast enough that PNG/WAV decoding becomes the
    # bottleneck, so workers pay off. On CPU the convolutions dominate by an
    # order of magnitude and workers only add process overhead -- and on
    # Windows, without persistent_workers, they are respawned every epoch.
    num_workers = args.num_workers
    if num_workers < 0:
        num_workers = 2 if device.type == "cuda" else 0

    loader_kwargs = {"num_workers": num_workers}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    if device.type == "cuda":
        loader_kwargs["pin_memory"] = True

    sampler_gen = torch.Generator()
    sampler_gen.manual_seed(args.seed)
    sampler = make_weighted_sampler(train_df, generator=sampler_gen)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              shuffle=(sampler is None), **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = get_model(args.model, bn_momentum=args.bn_momentum).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model {args.model}: {n_params} trainable params "
          f"(lr={args.lr}, batch={args.batch_size}, bn_momentum={args.bn_momentum}, seed={args.seed})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_bal_acc = -1.0
    best_val_auc = -1.0
    best_state = None
    best_epoch = 0
    epochs_since_improvement = 0

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
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        train_loss = running_loss / max(1, n_batches)
        val_metrics, val_per_method = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_acc={val_metrics['overall_acc']:.4f}  "
              f"val_balanced_acc={val_metrics['balanced_acc']:.4f}  "
              f"val_auc={val_metrics['auc']:.4f}  "
              f"val_cover_recall={val_metrics['cover_recall']:.4f}  "
              f"val_stego_recall={val_metrics['stego_recall']:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "lr": current_lr, **val_metrics})

        # Checkpoint on BALANCED accuracy, not raw accuracy: with a ~90/10
        # stego/cover split, raw accuracy rewards a model that just defaults
        # to predicting "stego" for almost everything.
        #
        # AUC breaks ties, and that tiebreak matters more than it sounds. A
        # model that has collapsed to one constant prediction scores EXACTLY
        # 0.5000 balanced accuracy every epoch, so on a strict > comparison
        # nothing ever counts as an improvement, the patience counter runs
        # out, and the run is killed before it has learned anything. AUC
        # keeps moving while the model is still collapsed, so it tells us
        # whether the thing is actually making progress underneath.
        bal, auc = val_metrics["balanced_acc"], val_metrics["auc"]
        auc_cmp = -1.0 if auc is None or np.isnan(auc) else auc
        eps = 1e-9
        improved = (bal > best_val_bal_acc + eps) or \
                   (bal >= best_val_bal_acc - eps and auc_cmp > best_val_auc + eps)

        if improved:
            best_val_bal_acc = max(bal, best_val_bal_acc)
            best_val_auc = max(auc_cmp, best_val_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improvement = 0
            print(f"  -> new best (val_balanced_acc={bal:.4f}, val_auc={auc_cmp:.4f}), checkpointing")
        else:
            epochs_since_improvement += 1
            if args.patience and epoch >= args.min_epochs and epochs_since_improvement >= args.patience:
                print(f"  -> no improvement in {args.patience} epochs, stopping early at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best checkpoint from epoch {best_epoch} "
              f"(val_balanced_acc={best_val_bal_acc:.4f}) for final evaluation")

    test_metrics, test_per_method = evaluate(model, test_loader, device)
    print(f"\nFinal test accuracy: {test_metrics['overall_acc']:.4f}  "
          f"(balanced: {test_metrics['balanced_acc']:.4f}, AUC: {test_metrics['auc']:.4f})")
    print(f"  cover recall: {test_metrics['cover_recall']:.4f}  "
          f"stego recall: {test_metrics['stego_recall']:.4f}")
    if test_metrics["cover_recall"] < 0.2:
        print("  WARNING: cover recall is very low -- this model is close to "
              "answering 'stego' for everything, so its per-method stego "
              "accuracies are NOT evidence of detection.")
    for m, acc in sorted(test_per_method.items()):
        flag = "  <-- HELD OUT (blind generalization test)" if m == args.exclude_method else ""
        print(f"  {m:15s} acc={acc:.4f}{flag}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = args.tag or f"{args.model}_{'excl_' + args.exclude_method if args.exclude_method else 'allmethods'}"

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{tag}.pt"))

    pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, f"train_history_{tag}.csv"), index=False)

    base = {"tag": tag, "modality": args.modality, "model": args.model,
            "excluded_method": args.exclude_method}
    test_rows = [dict(base, method=m, accuracy=acc) for m, acc in test_per_method.items()]
    test_rows.append(dict(base, method="OVERALL", accuracy=test_metrics["overall_acc"]))
    test_rows.append(dict(base, method="BALANCED", accuracy=test_metrics["balanced_acc"]))
    test_rows.append(dict(base, method="COVER_RECALL", accuracy=test_metrics["cover_recall"]))
    test_rows.append(dict(base, method="STEGO_RECALL", accuracy=test_metrics["stego_recall"]))
    test_rows.append(dict(base, method="AUC", accuracy=test_metrics["auc"]))
    results_path = os.path.join(RESULTS_DIR, f"test_results_{tag}.csv")
    pd.DataFrame(test_rows).to_csv(results_path, index=False)
    print(f"\nSaved model to models/{tag}.pt")
    print(f"Saved results to results/test_results_{tag}.csv")


if __name__ == "__main__":
    main()

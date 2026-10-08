"""
Phase 3 - consolidate every result into one comparison table.

Each training run writes its own results/test_results_<tag>.csv, which makes
it impossible to see the thing Phase 3 is actually about: whether a detector
trained on some embedding methods still detects a method it has never seen.
This script joins the CNN runs with the classical baseline into
results/phase3_summary.csv and prints tables ready to paste into the README.

The subtlety it deliberately encodes: accuracy on a single stego method is
meaningless on its own, because a detector that answers "stego" to
everything scores 100% on every stego method and 0% on covers. So every
per-method number is reported next to that run's cover recall, and the
headline metrics are balanced accuracy and AUC rather than raw accuracy.

    python3 summarize_phase3.py
"""

import os
import glob
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")

SPECIAL_ROWS = {"OVERALL", "BALANCED", "COVER_RECALL", "STEGO_RECALL", "AUC"}


def _setting_name(excluded):
    return "all_methods_train" if not excluded else f"held_out_{excluded}"


def load_cnn_runs():
    """One record per trained model, from results/test_results_*.csv."""
    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "test_results_*.csv"))):
        tag = os.path.basename(path)[len("test_results_"):-len(".csv")]
        if tag.startswith("smoke_"):
            continue  # smoke runs are plumbing checks, not results
        df = pd.read_csv(path)
        if df.empty:
            continue

        first = df.iloc[0]
        excluded = first.get("excluded_method")
        excluded = None if pd.isna(excluded) else str(excluded)

        lookup = dict(zip(df.method, df.accuracy))
        per_method = {k: v for k, v in lookup.items() if k not in SPECIAL_ROWS and k != "none"}
        cover_recall = lookup.get("COVER_RECALL", lookup.get("none", np.nan))

        runs.append({
            "detector": f"cnn:{first.get('model', tag)}",
            "modality": first.get("modality", ""),
            "setting": _setting_name(excluded),
            "held_out_method": excluded,
            "tag": tag,
            "overall_acc": lookup.get("OVERALL", np.nan),
            "balanced_acc": lookup.get("BALANCED", np.nan),
            "cover_recall": cover_recall,
            "stego_recall": lookup.get("STEGO_RECALL", np.nan),
            "auc": lookup.get("AUC", np.nan),
            "per_method": per_method,
        })
    return runs


def load_classical_runs():
    """Rebuild the same record shape from the classical baseline CSV, which
    stores per-method accuracy and sample counts but no summary rows."""
    path = os.path.join(RESULTS_DIR, "classical_baseline_results.csv")
    if not os.path.exists(path):
        return []

    df = pd.read_csv(path)
    runs = []
    for (modality, setting), grp in df.groupby(["modality", "setting"]):
        cover_rows = grp[grp.method == "none"]
        stego_rows = grp[grp.method != "none"]

        cover_recall = cover_rows.accuracy.iloc[0] if len(cover_rows) else np.nan
        # weight each stego method by its sample count to get overall stego recall
        if len(stego_rows) and stego_rows.n.sum() > 0:
            stego_recall = float((stego_rows.accuracy * stego_rows.n).sum() / stego_rows.n.sum())
        else:
            stego_recall = np.nan
        overall_n = grp.n.sum()
        overall_acc = float((grp.accuracy * grp.n).sum() / overall_n) if overall_n else np.nan

        held_out = None
        if setting.startswith("held_out_"):
            held_out = setting[len("held_out_"):]

        runs.append({
            "detector": "classical:random_forest",
            "modality": modality,
            "setting": setting,
            "held_out_method": held_out,
            "tag": f"classical_{modality}_{setting}",
            "overall_acc": overall_acc,
            "balanced_acc": float(np.nanmean([cover_recall, stego_recall])),
            "cover_recall": cover_recall,
            "stego_recall": stego_recall,
            "auc": np.nan,  # not persisted per-method by classical_baseline.py
            "per_method": dict(zip(stego_rows.method, stego_rows.accuracy)),
        })
    return runs


def build_long_table(runs):
    """One row per (run, stego method), each carrying its run's cover recall
    so a per-method number is never read in isolation."""
    rows = []
    for r in runs:
        for method, acc in sorted(r["per_method"].items()):
            rows.append({
                "detector": r["detector"],
                "modality": r["modality"],
                "setting": r["setting"],
                "method": method,
                "is_held_out": method == r["held_out_method"],
                "method_recall": acc,
                "cover_recall": r["cover_recall"],
                "balanced_acc": r["balanced_acc"],
                "auc": r["auc"],
                "tag": r["tag"],
            })
    return pd.DataFrame(rows)


def _fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def print_run_table(runs):
    print("\n### Detector performance (test set)\n")
    print("| Detector | Modality | Setting | Balanced acc | Cover recall | Stego recall | AUC |")
    print("|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda r: (r["modality"], r["detector"], r["setting"])):
        print(f"| {r['detector']} | {r['modality']} | {r['setting']} | "
              f"{_fmt(r['balanced_acc'])} | {_fmt(r['cover_recall'])} | "
              f"{_fmt(r['stego_recall'])} | {_fmt(r['auc'])} |")


def print_blindness_table(runs):
    """The Phase 3 headline: for each held-out run, how well the detector
    does on the method it never saw versus the methods it did see."""
    print("\n### Blind generalization (held-out method vs. seen methods)\n")
    print("| Detector | Modality | Held-out method | Acc on held-out | "
          "Mean acc on seen methods | Cover recall |")
    print("|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda r: (r["modality"], r["detector"])):
        if not r["held_out_method"]:
            continue
        held = r["per_method"].get(r["held_out_method"], np.nan)
        seen = [v for k, v in r["per_method"].items() if k != r["held_out_method"]]
        seen_mean = float(np.mean(seen)) if seen else np.nan
        print(f"| {r['detector']} | {r['modality']} | {r['held_out_method']} | "
              f"{_fmt(held)} | {_fmt(seen_mean)} | {_fmt(r['cover_recall'])} |")


def warn_degenerate(runs):
    bad = [r for r in runs
           if not np.isnan(r["cover_recall"]) and r["cover_recall"] < 0.2]
    if bad:
        print("\n!! Degenerate-detector warning\n")
        print("These runs have cover recall below 0.2, meaning they answer "
              "'stego' to almost everything.\nTheir per-method stego "
              "accuracies are NOT evidence of detection and should not be "
              "reported as such:\n")
        for r in bad:
            print(f"  - {r['tag']}: cover recall {r['cover_recall']:.3f}, "
                  f"balanced acc {r['balanced_acc']:.3f}")


def main():
    runs = load_cnn_runs() + load_classical_runs()
    if not runs:
        print("No results found in results/. Run run_phase3.py first.")
        return

    long_df = build_long_table(runs)
    out_path = os.path.join(RESULTS_DIR, "phase3_summary.csv")
    long_df.to_csv(out_path, index=False)

    run_df = pd.DataFrame([{k: v for k, v in r.items() if k != "per_method"} for r in runs])
    run_out = os.path.join(RESULTS_DIR, "phase3_summary_by_run.csv")
    run_df.to_csv(run_out, index=False)

    print_run_table(runs)
    print_blindness_table(runs)
    warn_degenerate(runs)

    print(f"\nSaved {out_path}")
    print(f"Saved {run_out}")


if __name__ == "__main__":
    main()

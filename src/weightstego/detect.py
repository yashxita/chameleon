"""
Phase 8 - the detector and its controls.

Reads the model zoo, extracts features per model, and runs the two tests that decide
whether there is a real signal:

  (A) held-out checkpoint: train on the train bases, test on unseen bases.
  (B) held-out method (blind, the Chameleon test): train with ONLY one embedding method as
      the positive class, test on the other method, on unseen bases.

Each test is run twice:
  baseline : classifier sees only the "quantizer-control" features (ctrl columns). If this
             already separates clean from stego, the signal is just quantization, not a
             payload.
  full     : classifier also sees the carrier-vs-control contrast features.

A payload signal exists only if `full` beats `baseline` and beats chance on held-out bases
and a held-out method. Reports balanced accuracy and AUC.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from features import model_features, split_columns

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ZOO = os.path.join(_ROOT, "results", "phase8", "zoo")
OUT = os.path.join(_ROOT, "results", "phase8")


def build_feature_table(manifest, topk):
    df = pd.read_csv(manifest)
    feats = []
    for _, r in df.iterrows():
        sd = torch.load(os.path.join(_ROOT, r.path), map_location="cpu")
        feats.append(model_features(sd, topk=topk))
    F = pd.DataFrame(feats)
    return pd.concat([df.reset_index(drop=True), F], axis=1), list(F.columns)


def _best_threshold(y, p):
    """Threshold maximising balanced accuracy on the training scores (no test leakage)."""
    best_t, best = 0.5, -1.0
    for t in np.unique(p):
        ba = balanced_accuracy_score(y, (p >= t).astype(int))
        if ba > best:
            best, best_t = ba, t
    return best_t


def _fit_eval(tr, te, cols):
    clf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")
    clf.fit(tr[cols].values, tr.label.values)
    t = _best_threshold(tr.label.values, clf.predict_proba(tr[cols].values)[:, 1])
    p = clf.predict_proba(te[cols].values)[:, 1]
    ba = balanced_accuracy_score(te.label, (p >= t).astype(int))
    auc = roc_auc_score(te.label, p) if te.label.nunique() > 1 else float("nan")
    return ba, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--manifest", default=os.path.join(ZOO, "manifest.csv"))
    args = ap.parse_args()

    tbl, cols = build_feature_table(args.manifest, args.topk)
    base_cols, full_cols = split_columns(cols)
    os.makedirs(OUT, exist_ok=True)
    tbl.to_csv(os.path.join(OUT, "features.csv"), index=False)

    tr_all = tbl[tbl.split == "train"]
    te_all = tbl[tbl.split == "test"]
    rows = []

    # (A) held-out checkpoint, all methods pooled
    for view, cset in [("baseline", base_cols), ("full", full_cols)]:
        ba, auc = _fit_eval(tr_all, te_all, cset)
        rows.append(dict(test="held_out_checkpoint", view=view, balanced=round(ba, 3),
                         auc=round(auc, 3), n_test=len(te_all)))

    # (B) blind held-out method: train on lsb + fp-qim, test on the UNSEEN quantized-survivable
    #     family qim_int8 (the case that matters).
    for view, cset in [("baseline", base_cols), ("full", full_cols)]:
        tr = tr_all[tr_all.method.isin(["none", "lsb", "qim_parity"])]
        te = te_all[te_all.method.isin(["none", "qim_int8"])]
        ba, auc = _fit_eval(tr, te, cset)
        rows.append(dict(test="blind(->qim_int8)", view=view, balanced=round(ba, 3),
                         auc=round(auc, 3), n_test=len(te)))

    # (C) per-method, held-out checkpoint: clean vs one method at a time, full features
    for m in ["lsb", "qim_parity", "qim_int8"]:
        te = te_all[te_all.method.isin(["none", m])]
        ba, auc = _fit_eval(tr_all, te, full_cols)
        rows.append(dict(test=f"per_method:{m}", view="full", balanced=round(ba, 3),
                         auc=round(auc, 3), n_test=len(te)))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "detect_summary.csv"), index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Phase 3 - Classical baseline.

Deep CNNs need a benchmark to be compared against, otherwise there's no
evidence they're actually learning something a simpler method couldn't.
This implements a classical feature-based detector in the spirit of rich
models (SPAM/SRM-style): handcrafted statistical features computed from
the noise residual, fed into a shallow classifier (Random Forest / SVM
via scikit-learn).

Image features: co-occurrence-style statistics of the pixel difference
array (a lightweight stand-in for full SPAM features, which are more
involved than fits a course-scope classical baseline).

Audio features: statistics of the sample-difference signal plus basic
spectral flatness/centroid features, the audio analogue of the same idea.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import soundfile as sf
from scipy.fft import rfft
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_image_features(img):
    """img: uint8 2D array. Returns a fixed-length feature vector built
    from the horizontal/vertical pixel-difference residual (where spatial-
    domain embedding noise shows up) plus a chi-square Pairs-of-Values
    statistic, the classical strong detector for LSB-style embedding:
    LSB embedding pulls adjacent histogram bin pairs (2k, 2k+1) toward
    equal frequency, which *lowers* this statistic relative to a natural
    image."""
    img = img.astype(np.float64)

    diff_h = np.diff(img, axis=1)
    diff_v = np.diff(img, axis=0)

    feats = []
    for d in (diff_h, diff_v):
        feats += [d.mean(), d.std(), d.var(),
                   np.mean(np.abs(d)), np.percentile(np.abs(d), 90)]
        sign_match = np.mean(np.sign(d[:, :-1]) == np.sign(d[:, 1:])) if d.shape[1] > 1 else 0.0
        feats.append(sign_match)

    lsb_plane = (img.astype(np.uint8) & 1)
    feats.append(lsb_plane.mean())

    feats += [img.mean(), img.std()]

    f = np.abs(rfft(img.flatten()))
    n = len(f)
    low_energy = f[: n // 8].sum()
    high_energy = f[n // 8:].sum()
    feats.append(high_energy / (low_energy + 1e-9))

    # chi-square pairs-of-values (log-transformed: raw values span several
    # orders of magnitude depending on image complexity)
    chi2 = _chi_square_pov(img.astype(np.uint8).flatten(), n_bins=256, vmin=0, vmax=256)
    feats.append(np.log1p(chi2))

    return np.array(feats, dtype=np.float64)


def _chi_square_pov(values_int, n_bins, vmin, vmax):
    hist, _ = np.histogram(values_int, bins=n_bins, range=(vmin, vmax))
    chi2 = 0.0
    for k in range(n_bins // 2):
        h0, h1 = hist[2 * k], hist[2 * k + 1]
        expected = (h0 + h1) / 2.0
        if expected > 0:
            chi2 += ((h0 - expected) ** 2) / expected + ((h1 - expected) ** 2) / expected
    return chi2


def extract_audio_features(sig):
    """sig: float32 1D array. Returns a fixed-length feature vector from
    the sample-difference residual, basic spectral stats, and a chi-square
    Pairs-of-Values statistic on the low byte of the 16-bit-quantized
    signal (the audio analogue of the image PoV feature above). Note:
    this feature is empirically much weaker for audio than for images in
    this dataset (see README / report), most likely because our synthetic
    covers (tones, chirps, FM synthesis) are lower-entropy than real
    speech/music recordings, this is an honest limitation worth reporting
    rather than a bug we should hide."""
    sig = sig.astype(np.float64)
    diff = np.diff(sig)

    feats = [diff.mean(), diff.std(), diff.var(),
              np.mean(np.abs(diff)), np.percentile(np.abs(diff), 90)]

    pcm = np.clip(sig * 32767, -32768, 32767).astype(np.int16)
    lsb_plane = pcm & 1
    feats.append(lsb_plane.mean())

    feats += [sig.mean(), sig.std()]

    spec = np.abs(rfft(sig))
    n = len(spec)
    low_energy = spec[: n // 8].sum()
    high_energy = spec[n // 8:].sum()
    feats.append(high_energy / (low_energy + 1e-9))

    spec_safe = spec + 1e-12
    geo_mean = np.exp(np.mean(np.log(spec_safe)))
    arith_mean = np.mean(spec_safe)
    feats.append(geo_mean / arith_mean)

    low_byte = (pcm.astype(np.int32) & 0xFF)
    chi2 = _chi_square_pov(low_byte, n_bins=256, vmin=0, vmax=256)
    feats.append(np.log1p(chi2))

    return np.array(feats, dtype=np.float64)


# ---------------------------------------------------------------------------
# Dataset assembly for classical models
# ---------------------------------------------------------------------------

def build_feature_matrix(manifest_df, modality):
    X, y, methods = [], [], []
    for _, row in manifest_df.iterrows():
        if modality == "image":
            data = np.array(Image.open(row.path))
            feat = extract_image_features(data)
        else:
            sig, _ = sf.read(row.path, dtype="float32")
            feat = extract_audio_features(sig)
        X.append(feat)
        y.append(0 if row.label == "cover" else 1)
        methods.append(row.method)
    return np.array(X), np.array(y), np.array(methods)


def run_classical_baseline(modality, exclude_methods=None):
    from dataset_loader import load_manifest, filter_manifest

    df = load_manifest()
    train_df = filter_manifest(df, modality, "train", exclude_methods=exclude_methods)
    test_df = filter_manifest(df, modality, "test")  # test on everything, incl. held-out method

    print(f"\n=== Classical baseline: {modality} "
          f"(excluded from training: {exclude_methods}) ===")

    # balance classes for training: with 9 stego variants per cover, a raw
    # 9:1 stego:cover ratio drowns out class_weight="balanced" in practice
    # and the model just learns to always predict "stego". Undersample
    # stego down to match the cover count instead.
    rng = np.random.default_rng(0)
    cover_rows = train_df[train_df.label == "cover"]
    stego_rows = train_df[train_df.label == "stego"]
    if len(stego_rows) > len(cover_rows):
        keep_idx = rng.choice(len(stego_rows), size=len(cover_rows), replace=False)
        stego_rows = stego_rows.iloc[keep_idx]
    train_df = pd.concat([cover_rows, stego_rows], ignore_index=True)

    print(f"Train (balanced): {len(train_df)} samples "
          f"({(train_df.label=='cover').sum()} cover / {(train_df.label=='stego').sum()} stego) "
          f"| Test: {len(test_df)} samples")

    X_train, y_train, _ = build_feature_matrix(train_df, modality)
    X_test, y_test, methods_test = build_feature_matrix(test_df, modality)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                   class_weight="balanced", random_state=0, n_jobs=-1)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"Overall test accuracy: {acc:.4f} | AUC: {auc:.4f}")

    # break down accuracy per method, this is the important number: if
    # accuracy on the held-out method is much lower than on seen methods,
    # that's expected and worth reporting; if it's close, that's evidence
    # of genuine blind generalization
    results_rows = []
    for m in sorted(set(methods_test)):
        mask = methods_test == m
        m_acc = accuracy_score(y_test[mask], y_pred[mask])
        results_rows.append({"modality": modality, "method": m,
                               "n": mask.sum(), "accuracy": m_acc})
        print(f"  {m:15s} n={mask.sum():4d}  accuracy={m_acc:.4f}")

    return pd.DataFrame(results_rows), clf, scaler


if __name__ == "__main__":
    results_dir = os.path.join(_PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    # baseline 1: train on all methods (standard blind setting)
    r_img, _, _ = run_classical_baseline("image", exclude_methods=None)
    r_img["setting"] = "all_methods_train"
    all_results.append(r_img)

    r_aud, _, _ = run_classical_baseline("audio", exclude_methods=None)
    r_aud["setting"] = "all_methods_train"
    all_results.append(r_aud)

    # baseline 2: held-out method test, train excluding dct_qim (image) /
    # echo_hiding (audio), the strongest / most different method in each
    # modality, to genuinely stress-test blindness
    r_img_ho, _, _ = run_classical_baseline("image", exclude_methods=["dct_qim"])
    r_img_ho["setting"] = "held_out_dct_qim"
    all_results.append(r_img_ho)

    r_aud_ho, _, _ = run_classical_baseline("audio", exclude_methods=["echo_hiding"])
    r_aud_ho["setting"] = "held_out_echo_hiding"
    all_results.append(r_aud_ho)

    final = pd.concat(all_results, ignore_index=True)
    out_path = os.path.join(results_dir, "classical_baseline_results.csv")
    final.to_csv(out_path, index=False)
    print(f"\nSaved classical baseline results to {out_path}")
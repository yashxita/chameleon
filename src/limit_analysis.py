"""
Phase 5 - epsilon sweeps: where does detection break, and what does the
attacker pay in perceptual quality to get there?

For each detector and attack, the perturbation budget eps is swept and we log

  tpr                  fraction of stego samples still flagged as stego
  cover_recall         fraction of clean covers still called cover (unchanged
                       by the attack; the attacker only touches stego files)
  balanced_acc         (cover_recall + tpr) / 2 -- the honest headline metric;
                       0.5 is a coin flip
  evasion_of_detected  among stego samples the CLEAN detector caught, the
                       fraction the attack pushed to "cover"
  tpr_<method>         per-embedding-method detection rate
  quality_vs_stego_db  PSNR (image) / SNR (audio) of the adversarial file
                       against the stego file it was made from: the attacker's
                       own added cost
  quality_vs_cover_db  the same against the ORIGINAL COVER: total distortion
                       the hidden-data file now carries
  frac_changed         fraction of pixels / samples altered by >= 1 file step

The 'random' attack adds the same-sized sign noise at random. It is the
control: if random noise of budget eps does as well as the gradient attack,
the detector is merely noise-fragile, not adversarially fooled.

    python3 limit_analysis.py --detector image_cnn --seeds 0 1 2
    python3 limit_analysis.py --detector all --smoke
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch

import attacks as A
from dataset_loader import load_manifest, filter_manifest, ImageStegoDataset, AudioStegoDataset
from models import ImageStegoCNN, AudioSpectrogramCNN, AudioWaveformCNN

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
MODELS_DIR = os.path.join(_ROOT, "models")
OUT_DIR = os.path.join(_ROOT, "results", "phase5")

IMAGE_EPS = [0, 1, 2, 3, 4, 6, 8, 12, 16]                      # grey levels
AUDIO_EPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]               # int16 steps

DETECTORS = {
    "image_cnn": dict(modality="image", cls=ImageStegoCNN, ckpt="image_cnn_allmethods.pt",
                      units=A.IMAGE_UNITS, per_sample=False, eps=IMAGE_EPS,
                      surrogate="image_cnn_excl_dct_qim"),
    # image CNN after adversarial fine-tuning (adv_train.py); not part of --detector all
    "image_cnn_advtrained": dict(modality="image", cls=ImageStegoCNN, ckpt="image_cnn_advtrained.pt",
                                 units=A.IMAGE_UNITS, per_sample=False, eps=IMAGE_EPS,
                                 surrogate="image_cnn_excl_dct_qim"),
    "audio_spectrogram": dict(modality="audio", cls=AudioSpectrogramCNN,
                              ckpt="audio_spectrogram_allmethods.pt", units=A.AUDIO_UNITS,
                              per_sample=False, eps=AUDIO_EPS, surrogate="audio_waveform"),
    "audio_waveform": dict(modality="audio", cls=AudioWaveformCNN,
                           ckpt="audio_waveform_allmethods.pt", units=A.AUDIO_UNITS,
                           per_sample=False, eps=AUDIO_EPS, surrogate="audio_spectrogram"),
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def spec_norm_constants(model, n_clips=300, seed=0):
    """Training-set mean / std of the log-mel spectrogram, cached on disk.

    These replace the batch statistics AudioSpectrogramCNN computes in its own
    forward (see attacks.FixedNormSpectrogram)."""
    path = os.path.join(OUT_DIR, "spec_norm.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return d["mu"], d["sigma"]
    df = filter_manifest(load_manifest(), "audio", "train").sample(n_clips, random_state=seed)
    ds = AudioStegoDataset(df)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    with torch.no_grad():
        spec = model.to_db(model.melspec(x))
    d = dict(mu=float(spec.mean()), sigma=float(spec.std()), n_clips=n_clips, seed=seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f)
    return d["mu"], d["sigma"]


def load_detector(name, device="cpu"):
    cfg = DETECTORS[name]
    model = cfg["cls"]()
    sd = torch.load(os.path.join(MODELS_DIR, cfg["ckpt"]), map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    if name == "audio_spectrogram":
        mu, sigma = spec_norm_constants(model)
        model = A.FixedNormSpectrogram(model, mu, sigma)
    return _finish(model, False, device)


def _finish(model, per_sample, device):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if per_sample:
        model = A.PerSample(model)
    return model.to(device)


def load_surrogate(name, device="cpu"):
    """Surrogate for the transfer attack.

    audio: the other audio model (a different architecture, same task).
    image: preferably image_cnn_excl_dct_qim.pt, the same architecture trained
           WITHOUT one of the three embedding methods, i.e. an attacker who has
           a competent detector but not the defender's weights or data. If that
           file is absent, fall back to the plain SurrogateImageCNN trained by
           surrogate.py. If neither exists, raise FileNotFoundError.
    """
    if name in ("audio_waveform", "audio_spectrogram"):
        return load_detector(name, device)
    excl = os.path.join(MODELS_DIR, "image_cnn_excl_dct_qim.pt")
    if os.path.exists(excl):
        model = ImageStegoCNN()
        model.load_state_dict(torch.load(excl, map_location="cpu"))
        return _finish(model, False, device)
    from surrogate import SurrogateImageCNN
    model = SurrogateImageCNN()
    path = os.path.join(MODELS_DIR, "surrogate_image_cnn.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return _finish(model, False, device)


@torch.no_grad()
def predict(model, x, bs=32):
    out = []
    for i in range(0, len(x), bs):
        out.append(model(x[i:i + bs]).argmax(1))
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _stack(ds):
    return torch.stack([ds[i][0] for i in range(len(ds))])


def build_attack_set(modality, n_per_method, n_cover, seed):
    """Test-split stego samples balanced across methods, clean test covers, and
    for each stego sample the original cover it was made from."""
    df = load_manifest()
    test = filter_manifest(df, modality, "test")
    DS = ImageStegoDataset if modality == "image" else AudioStegoDataset

    stego = test[test.label == "stego"]
    picks = [g.sample(min(n_per_method, len(g)), random_state=seed)
             for _, g in stego.groupby("method")]
    stego = pd.concat(picks).reset_index(drop=True)
    covers = test[test.label == "cover"].sample(min(n_cover, (test.label == "cover").sum()),
                                                 random_state=seed).reset_index(drop=True)

    ext = ".png" if modality == "image" else ".wav"
    by_name = test[test.label == "cover"].set_index("filename")
    paired = by_name.loc[[f.split("__")[0] + ext for f in stego.filename]].reset_index()

    return dict(x_stego=_stack(DS(stego)), methods=stego.method.values,
                x_cover=_stack(DS(covers)), x_paired=_stack(DS(paired)))


# ---------------------------------------------------------------------------
# Attack registry
# ---------------------------------------------------------------------------

def make_attacks(model, units, surrogate, steps, wanted):
    reg = {}

    def random_noise(x, eps):
        if eps <= 0:
            return x.clone()
        e = eps / units.scale
        noise = e * torch.sign(torch.randn_like(x))
        return units.quantize((x + noise).clamp(units.lo, units.hi))

    reg["random"] = random_noise
    reg["fgsm"] = lambda x, eps: A.fgsm(model, x, eps, units)
    reg["fgsm_bpda"] = lambda x, eps: A.fgsm(model, x, eps, units, bpda=True)
    reg["pgd"] = lambda x, eps: A.pgd(model, x, eps, units, steps=steps)
    reg["pgd_bpda"] = lambda x, eps: A.pgd(model, x, eps, units, steps=steps, bpda=True)
    if surrogate is not None:
        reg["transfer"] = lambda x, eps: A.transfer(surrogate, x, eps, units, steps=steps)
    return {k: v for k, v in reg.items() if k in wanted}


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def quality(ref, adv, modality):
    return A.psnr_db(ref, adv) if modality == "image" else A.snr_db(ref, adv)


def _finite_mean(t):
    t = t[torch.isfinite(t)]
    return float(t.mean()) if len(t) else float("nan")


def run_sweep(det_name, model, data, attack_fns, eps_list, seed, bs, units, modality, log):
    x_all, methods_all = data["x_stego"], data["methods"]
    clean_pred = predict(model, x_all, bs)
    cover_recall = float((predict(model, data["x_cover"], bs) == 0).float().mean())
    det = clean_pred == 1
    n_all, n_det = len(x_all), int(det.sum())
    log(f"  clean: tpr={n_det / n_all:.3f} cover_recall={cover_recall:.3f} "
        f"(n_stego={n_all}, n_detected={n_det}, n_cover={len(data['x_cover'])})")

    # Threat model: the attacker only needs to touch files the detector would
    # catch. A stego file the clean detector already misses is evaded for free
    # and is left untouched, so tpr(eps) is monotone non-increasing and the
    # perceptual cost is measured only where an attack was actually needed.
    x_st = x_all[det]
    x_pair = data["x_paired"][det]
    methods = methods_all[det.cpu().numpy()]
    method_total = {m: int((methods_all == m).sum()) for m in sorted(set(methods_all))}

    rows = []
    for atk, fn in attack_fns.items():
        for eps in eps_list:
            t0 = time.time()
            x_adv = torch.cat([fn(x_st[i:i + bs], eps) for i in range(0, n_det, bs)])
            pred = predict(model, x_adv, bs)
            still = pred == 1                                  # still detected
            tpr = float(still.sum()) / n_all
            row = dict(detector=det_name, attack=atk, seed=seed, eps=eps,
                       n_stego=n_all, n_attacked=n_det,
                       tpr=tpr, cover_recall=cover_recall,
                       balanced_acc=0.5 * (cover_recall + tpr),
                       evasion_of_detected=float((~still).float().mean()),
                       quality_vs_stego_db=_finite_mean(quality(x_st, x_adv, modality)),
                       quality_vs_cover_db=_finite_mean(quality(x_pair, x_adv, modality)),
                       frac_changed=float(A.frac_changed(x_st, x_adv, units).mean()))
            for m, tot in method_total.items():
                sel = torch.as_tensor(methods == m)
                row[f"tpr_{m}"] = float(still[sel].sum()) / tot
            rows.append(row)
            log(f"  {atk:10s} eps={eps:>4}  evade={row['evasion_of_detected']:.3f}  "
                f"tpr={tpr:.3f}  bal={row['balanced_acc']:.3f}  "
                f"Q={row['quality_vs_stego_db']:.1f}dB  ({time.time() - t0:.0f}s)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="all", choices=list(DETECTORS) + ["all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-per-method", type=int, default=24)
    ap.add_argument("--n-cover", type=int, default=72)
    ap.add_argument("--pgd-steps", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--attacks", nargs="+",
                    default=["random", "fgsm", "pgd", "pgd_bpda", "transfer"])
    ap.add_argument("--eps", type=float, nargs="+", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the plumbing")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.n_per_method, args.n_cover, args.pgd_steps = [0], 4, 8, 3

    os.makedirs(OUT_DIR, exist_ok=True)
    names = ([d for d in DETECTORS if d != "image_cnn_advtrained"]
             if args.detector == "all" else [args.detector])

    def log(line):
        print(line, flush=True)

    for name in names:
        cfg = DETECTORS[name]
        eps_list = args.eps if args.eps else cfg["eps"]
        eps_list = [int(e) if float(e).is_integer() else e for e in eps_list]
        log(f"\n=== {name} ({cfg['modality']}) eps={eps_list} ===")
        model = load_detector(name, args.device)

        surrogate = None
        if "transfer" in args.attacks:
            try:
                surrogate = load_surrogate(cfg["surrogate"], args.device)
            except FileNotFoundError:
                log(f"  [no surrogate checkpoint for {name}; skipping transfer]")

        all_rows = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            log(f"-- seed {seed}")
            data = build_attack_set(cfg["modality"], args.n_per_method, args.n_cover, seed)
            data = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in data.items()}
            fns = make_attacks(model, cfg["units"], surrogate, args.pgd_steps, args.attacks)
            rows = run_sweep(name, model, data, fns, eps_list, seed, args.batch_size,
                             cfg["units"], cfg["modality"], log)
            all_rows += rows

            out = os.path.join(OUT_DIR, f"sweep_{name}{args.tag}{'_smoke' if args.smoke else ''}.csv")
            pd.DataFrame(all_rows).to_csv(out, index=False)   # saved after every seed
        log(f"saved {out}")


if __name__ == "__main__":
    main()
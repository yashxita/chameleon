"""
Phase 5 - turn the epsilon sweeps into the tables and figures of the limit analysis.

Reads results/phase5/sweep_<detector>.csv (one row per detector x attack x seed x
eps), averages over seeds, and writes

  results/phase5/phase5_summary.csv   break points per detector and attack
  results/phase5/phase5_summary.md    the same as a markdown table
  results/phase5/fig_balanced_acc_vs_eps.png
  results/phase5/fig_limit_cost_vs_evasion.png
  results/phase5/fig_per_method.png

Break points are defined on EVASION OF DETECTED files (the fraction of stego
samples the clean detector caught that the attack pushed to "cover"), because it
is 0 -> 1 for every detector regardless of how good the clean detector was:

  eps_50   smallest budget evading >= 50% of detected files
  eps_90   smallest budget evading >= 90%

each reported with the perceptual cost (PSNR / SNR against the stego file) the
attacker pays at that budget. 'not reached' means even the largest budget tried
did not get there.

Colors follow a validated categorical order (fixed per attack, never cycled);
the random-noise control is neutral gray. Marker shape and line style repeat the
identity so nothing relies on hue alone.
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "results", "phase5")

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

ATTACK_STYLE = {   # fixed order = fixed identity
    "random":   dict(color="#898781", marker="x", ls=":",  label="random noise (control)"),
    "fgsm":     dict(color="#2a78d6", marker="o", ls="-",  label="FGSM"),
    "fgsm_bpda": dict(color="#4a3aa7", marker="v", ls="-", label="FGSM + BPDA"),
    "pgd":      dict(color="#eb6834", marker="s", ls="-",  label="PGD"),
    "pgd_bpda": dict(color="#1baf7a", marker="^", ls="--", label="PGD + BPDA"),
    "transfer": dict(color="#e87ba4", marker="D", ls="-.", label="transfer (black-box)"),
}
DET_TITLE = {"image_cnn": "Image CNN", "audio_spectrogram": "Audio spectrogram CNN",
             "audio_waveform": "Audio waveform CNN",
             "image_cnn_advtrained": "Image CNN, adversarially fine-tuned"}
EPS_UNIT = {"image_cnn": "grey levels (L-inf)", "audio_spectrogram": "int16 steps (L-inf)",
            "audio_waveform": "int16 steps (L-inf)", "image_cnn_advtrained": "grey levels (L-inf)"}
QUALITY_NAME = {"image_cnn": "PSNR (dB)", "audio_spectrogram": "SNR (dB)",
                "audio_waveform": "SNR (dB)", "image_cnn_advtrained": "PSNR (dB)"}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_axisbelow(True)


def load():
    frames = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(OUT, "sweep_*.csv")))
              if "smoke" not in p]
    if not frames:
        raise SystemExit("no sweep CSVs found in results/phase5/")
    return pd.concat(frames, ignore_index=True)


def aggregate(df):
    num = [c for c in df.columns if c not in ("detector", "attack", "seed", "eps")]
    g = df.groupby(["detector", "attack", "eps"])[num]
    return g.mean().reset_index(), g.std(ddof=0).reset_index()


def break_points(mean):
    rows = []
    for (det, atk), g in mean.groupby(["detector", "attack"]):
        g = g.sort_values("eps")
        clean = g[g.eps == 0].iloc[0]
        row = dict(detector=det, attack=atk, clean_tpr=clean.tpr, clean_balanced=clean.balanced_acc,
                   n_attacked=clean.n_attacked, max_evasion=g.evasion_of_detected.max())
        for q in (0.5, 0.9):
            hit = g[g.evasion_of_detected >= q]
            if len(hit):
                h = hit.iloc[0]
                row[f"eps_{int(q * 100)}"] = h.eps
                row[f"cost_db_at_eps_{int(q * 100)}"] = h.quality_vs_stego_db
                row[f"frac_changed_at_eps_{int(q * 100)}"] = h.frac_changed
            else:
                row[f"eps_{int(q * 100)}"] = np.nan
                row[f"cost_db_at_eps_{int(q * 100)}"] = np.nan
                row[f"frac_changed_at_eps_{int(q * 100)}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(v, nd=0):
    return "not reached" if pd.isna(v) else f"{v:.{nd}f}"


def write_markdown(summary, path):
    lines = ["| Detector | Attack | Clean TPR | Evade 50% at | cost | Evade 90% at | cost | Max evasion |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        unit = "grey" if r.detector.startswith("image") else "LSB"
        c50 = "" if pd.isna(r.eps_50) else f"{r.cost_db_at_eps_50:.1f} dB"
        c90 = "" if pd.isna(r.eps_90) else f"{r.cost_db_at_eps_90:.1f} dB"
        e50 = "not reached" if pd.isna(r.eps_50) else f"{r.eps_50:g} {unit}"
        e90 = "not reached" if pd.isna(r.eps_90) else f"{r.eps_90:g} {unit}"
        lines.append(f"| {DET_TITLE[r.detector]} | {ATTACK_STYLE[r.attack]['label']} | {r.clean_tpr:.3f} "
                     f"| {e50} | {c50} | {e90} | {c90} | {r.max_evasion:.2f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def fig_balanced(mean, std, dets):
    fig, axes = plt.subplots(1, len(dets), figsize=(5.2 * len(dets), 4.3), sharey=True,
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, det in zip(axes, dets):
        style_axes(ax)
        m = mean[mean.detector == det]
        s = std[std.detector == det]
        eps = sorted(m.eps.unique())
        xs = np.arange(len(eps))
        ends = []
        for atk, st in ATTACK_STYLE.items():
            a = m[m.attack == atk].sort_values("eps")
            if a.empty:
                continue
            sd = s[s.attack == atk].sort_values("eps")
            y, e = a.balanced_acc.values, sd.balanced_acc.values
            ax.fill_between(xs, y - e, y + e, color=st["color"], alpha=0.13, lw=0)
            ax.plot(xs, y, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.8,
                    ms=5.5, mec=SURFACE, mew=1.0, label=st["label"])
            ends.append((y[-1], st))
        ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (2, 3)))
        ax.text(-0.25, 0.493, "chance", color=MUTED, fontsize=8, ha="left", va="top")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{e:g}" for e in eps])
        ax.set_xlabel(f"perturbation budget eps, {EPS_UNIT[det]}", color=INK2, fontsize=9)
        ax.set_title(DET_TITLE[det], color=INK, fontsize=11, loc="left", fontweight="bold")
        ax.set_ylim(0.38, 0.95)
    axes[0].set_ylabel("balanced accuracy", color=INK2, fontsize=9)
    seen = {}
    for a in axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            seen.setdefault(l, h)
    order = [ATTACK_STYLE[k]["label"] for k in ATTACK_STYLE if ATTACK_STYLE[k]["label"] in seen]
    fig.legend([seen[l] for l in order], order, loc="lower center", ncol=len(order), frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Detection under attack: balanced accuracy vs perturbation budget "
                 "(mean over seeds, band = 1 sd)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(os.path.join(OUT, "fig_balanced_acc_vs_eps.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_limit(mean, dets, emb):
    fig, axes = plt.subplots(1, len(dets), figsize=(5.2 * len(dets), 4.3), sharey=True,
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, det in zip(axes, dets):
        style_axes(ax)
        m = mean[(mean.detector == det) & (mean.eps > 0)]
        for atk, st in ATTACK_STYLE.items():
            a = m[m.attack == atk].sort_values("eps")
            a = a[np.isfinite(a.quality_vs_stego_db)]
            if a.empty:
                continue
            ax.plot(a.quality_vs_stego_db, a.evasion_of_detected, color=st["color"],
                    marker=st["marker"], ls=st["ls"], lw=1.8, ms=5.5, mec=SURFACE, mew=1.0,
                    label=st["label"])
        xs_all = [l.get_xdata() for l in ax.lines]
        xs_all = np.concatenate([np.asarray(x, dtype=float) for x in xs_all]) if xs_all else np.array([50.0])
        refs = [q for _, q in emb.get(det, [])]
        hi = max([np.nanmax(xs_all)] + refs) + 4
        lo = min([np.nanmin(xs_all)] + refs) - 3
        for (name, q) in emb.get(det, []):
            ax.axvline(q, color=INK2, lw=1, ls=(0, (1, 3)))
            ax.text(q, 1.03, name.replace("_", " "), color=INK2, fontsize=7.5, ha="center",
                    va="bottom", bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
        ax.set_xlim(hi, lo)                       # inverted: more distortion to the right
        ax.set_ylim(-0.03, 1.12)
        ax.set_xlabel(f"attacker's added distortion, {QUALITY_NAME[det]} (lower = more visible)",
                      color=INK2, fontsize=9)
        ax.set_title(DET_TITLE[det], color=INK, fontsize=11, loc="left", fontweight="bold", pad=14)
    axes[0].set_ylabel("fraction of detected stego files evaded", color=INK2, fontsize=9)
    seen = {}
    for a in axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            seen.setdefault(l, h)
    order = [ATTACK_STYLE[k]["label"] for k in ATTACK_STYLE if ATTACK_STYLE[k]["label"] in seen]
    fig.legend([seen[l] for l in order], order, loc="lower center", ncol=len(order), frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("The limit: evasion bought per unit of added distortion "
                 "(dotted lines = the embedding's own distortion)", color=INK, fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig_limit_cost_vs_evasion.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_methods(mean, dets, attack="pgd_bpda"):
    # fixed identity per embedding method, never cycled by position
    style = {"lsb": ("#2a78d6", "o"), "lsb_matching": ("#eb6834", "s"),
             "dct_qim": ("#1baf7a", "^"), "echo_hiding": ("#4a3aa7", "v")}
    fig, axes = plt.subplots(1, len(dets), figsize=(5.2 * len(dets), 4.3), sharey=True,
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    seen = {}
    for ax, det in zip(axes, dets):
        style_axes(ax)
        m = mean[(mean.detector == det) & (mean.attack == attack)].sort_values("eps")
        if m.empty:
            continue
        xs = np.arange(len(m))
        for meth, (col, mk) in style.items():
            c = f"tpr_{meth}"
            if c not in m.columns or m[c].isna().all():
                continue                          # method does not exist for this modality
            h, = ax.plot(xs, m[c].values, color=col, marker=mk, lw=1.8, ms=5.5,
                         mec=SURFACE, mew=1.0, label=meth.replace("_", " "))
            seen.setdefault(meth, h)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{e:g}" for e in m.eps.values])
        ax.set_xlabel(f"eps, {EPS_UNIT[det]}", color=INK2, fontsize=9)
        ax.set_title(DET_TITLE[det], color=INK, fontsize=11, loc="left", fontweight="bold")
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("detection rate of that method", color=INK2, fontsize=9)
    order = [k for k in style if k in seen]
    fig.legend([seen[k] for k in order], [k.replace("_", " ") for k in order], loc="lower center",
               ncol=len(order), frameon=False, fontsize=9, labelcolor=INK2,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Which embedding breaks first under {ATTACK_STYLE[attack]['label']}",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(os.path.join(OUT, "fig_per_method.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_defense(mean, std):
    dets = ["image_cnn", "image_cnn_advtrained"]
    if not set(dets) <= set(mean.detector):
        return
    keep = [0, 1, 2, 4, 8]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True, facecolor=SURFACE)
    for ax, det in zip(axes, dets):
        style_axes(ax)
        for atk in ("pgd", "pgd_bpda"):
            st = ATTACK_STYLE[atk]
            a = mean[(mean.detector == det) & (mean.attack == atk) & mean.eps.isin(keep)].sort_values("eps")
            sd = std[(std.detector == det) & (std.attack == atk) & std.eps.isin(keep)].sort_values("eps")
            xs = np.arange(len(a))
            ax.fill_between(xs, a.balanced_acc - sd.balanced_acc, a.balanced_acc + sd.balanced_acc,
                            color=st["color"], alpha=0.13, lw=0)
            ax.plot(xs, a.balanced_acc, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.8,
                    ms=5.5, mec=SURFACE, mew=1.0, label=st["label"])
        ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (2, 3)))
        ax.set_xticks(np.arange(len(keep)))
        ax.set_xticklabels([f"{e:g}" for e in keep])
        ax.set_xlabel("perturbation budget eps, grey levels (L-inf)", color=INK2, fontsize=9)
        ax.set_title(DET_TITLE[det], color=INK, fontsize=11, loc="left", fontweight="bold")
        ax.set_ylim(0.2, 0.95)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    axes[0].set_ylabel("balanced accuracy", color=INK2, fontsize=9)
    fig.suptitle("Defense: adversarial fine-tuning (80 CPU-sized steps) vs the original detector",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig_defense.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def embedding_reference():
    """The embedding's own distortion, for reference lines on the limit figure."""
    q = pd.read_csv(os.path.join(_ROOT, "results", "embedding_quality.csv"))
    q = q.groupby(["modality", "method"]).quality_db.mean()
    out = {"image_cnn": [(f"{m}\n{q['image', m]:.0f} dB", q["image", m]) for m in ("lsb", "dct_qim")
                         if ("image", m) in q.index]}
    aud = [(f"{m}\n{q['audio', m]:.0f} dB", q["audio", m]) for m in ("lsb", "echo_hiding")
           if ("audio", m) in q.index]
    out["audio_spectrogram"] = aud
    out["audio_waveform"] = aud
    return out


def main():
    df = load()
    mean, std = aggregate(df)
    dets = [d for d in ("image_cnn", "audio_spectrogram", "audio_waveform") if d in set(df.detector)]
    summary = break_points(mean)
    summary.to_csv(os.path.join(OUT, "phase5_summary.csv"), index=False)
    write_markdown(summary, os.path.join(OUT, "phase5_summary.md"))
    fig_balanced(mean, std, dets)
    fig_limit(mean, dets, embedding_reference())
    fig_methods(mean, dets)
    fig_defense(mean, std)
    print(summary.to_string(index=False))
    print("\nwrote figures and summary to", OUT)


if __name__ == "__main__":
    main()
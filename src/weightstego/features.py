"""
Phase 8 - the detector's features.

The central problem (raised in review): ordinary quantization ALSO puts weights on a
grid, so "weights lie on a lattice" cannot by itself mean a payload is present. The
feature here is built to separate an *embedding* lattice from ordinary quantization by
contrasting the carrier weights (top-K by magnitude, where a payload is hidden) against a
random control subset of the SAME tensor (which a pure quantizer affects identically).

For a magnitude-adaptive step Delta(w) = 2^(floor(log2|w|) - (n-1)):
  residual r = w/Delta - round(w/Delta)   -> ~0 for weights sitting on the lattice
  index parity = round(w/Delta) mod 2     -> ~balanced and periodic if parity carries bits

We compute, per candidate n:
  on_lattice   = fraction of |r| < eps
  parity_bias  = |mean(parity) - 0.5|
and report the CONTRAST (top-K minus random-control). A model that is merely quantized has
contrast ~0; a QIM-embedded model has positive on_lattice contrast. The raw top-K values
and the raw control values are kept too, so a baseline classifier can be built from the
"quantizer-only" view and compared against the contrast view.
"""

import numpy as np
import torch

from embed import carrier_tensors, _topk_idx

NS = (1, 2, 3)
EPS = 1e-6


def _stats_for_subset(w, n):
    e = torch.floor(torch.log2(w.abs().clamp_min(1e-12)))
    delta = torch.pow(2.0, e - (n - 1))
    q = w / delta
    r = (q - torch.round(q)).abs()
    on_lat = (r < EPS).float().mean().item()
    parity = (torch.round(q).to(torch.int64) % 2).float()
    parity_bias = abs(parity.mean().item() - 0.5)
    return on_lat, parity_bias


def _mant_lsb_frac(w):
    ints = w.to(torch.float32).view(torch.int32)
    return (ints & 1).to(torch.float32).mean().item()


def _code_parity_bias(w):
    """|0.5 - fraction of odd int8 codes|. Embedded int8 payload forces this toward 0 on the
    carriers while clean largest codes are skewed."""
    scale = w.abs().max().clamp_min(1e-12) / 127.0
    codes = torch.round(w / scale).clamp(-127, 127).to(torch.int64)
    return abs((codes % 2).to(torch.float32).mean().item() - 0.5)


def model_features(state_dict, topk=256, min_numel=4096, seed=0):
    """Return a flat dict of named features for one model."""
    rng = np.random.default_rng(seed)
    cols = {f"onlat_n{n}": [] for n in NS}
    cols.update({f"onlat_ctrl_n{n}": [] for n in NS})
    cols.update({f"parity_n{n}": [] for n in NS})
    for name in ("mant_lsb", "mant_lsb_ctrl", "codepar", "codepar_ctrl"):
        cols[name] = []
    for k in carrier_tensors(state_dict, min_numel):
        flat = state_dict[k].reshape(-1).to(torch.float32)
        idx = _topk_idx(flat, topk)
        w_top = flat[idx]
        ctrl = torch.from_numpy(rng.choice(flat.numel(), size=topk, replace=False))
        w_ctrl = flat[ctrl]
        for n in NS:
            ol, pb = _stats_for_subset(w_top, n)
            olc, _ = _stats_for_subset(w_ctrl, n)
            cols[f"onlat_n{n}"].append(ol)
            cols[f"onlat_ctrl_n{n}"].append(olc)
            cols[f"parity_n{n}"].append(pb)
        cols["mant_lsb"].append(_mant_lsb_frac(w_top))
        cols["mant_lsb_ctrl"].append(_mant_lsb_frac(w_ctrl))
        cols["codepar"].append(_code_parity_bias(w_top))
        cols["codepar_ctrl"].append(_code_parity_bias(w_ctrl))
    feat = {}
    for name, vals in cols.items():
        v = np.array(vals) if vals else np.array([0.0])
        feat[f"{name}_mean"] = float(v.mean())
        feat[f"{name}_max"] = float(v.max())
    for n in NS:                         # the contrast: carrier minus quantizer-control
        feat[f"contrast_n{n}"] = feat[f"onlat_n{n}_mean"] - feat[f"onlat_ctrl_n{n}_mean"]
    return feat


# which feature columns a "quantizer-only" baseline is allowed to see (no carrier contrast)
def split_columns(feature_names):
    baseline = [c for c in feature_names if "ctrl" in c]
    full = list(feature_names)
    return baseline, full

"""
Phase 8 - targeted sanitizer (the second, patent-relevant leg).

Given a model flagged as carrying a quantization-surviving (qim_int8) payload, remove the
payload with the SMALLEST possible change and verify the model still works. The payload lives
in the parity of the top-K int8 codes, so we flip only those codes by one level (a half-step
in weight units). That randomises the parity (payload bit-error -> ~0.5) while changing each
touched weight by exactly one quantization step and leaving every other weight untouched.

Compared against the blunt baseline MEASER calls out: random bit flipping across all weights.
We report, on the synthetic zoo (where accuracy is measurable): payload bit-error-rate (higher
= more disrupted) against model accuracy drop (lower = better preserved).

    python3 sanitize.py --bits 256
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch

from embed import carrier_tensors, decode_qim_int8, _select_int8_carriers, _payload_bits, _int8_scale
from modelzoo import SmallCNN, _synth_batch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ZOO = os.path.join(_ROOT, "results", "phase8", "zoo")
OUT = os.path.join(_ROOT, "results", "phase8")


def accuracy(sd, n=512, seed=999):
    m = SmallCNN()
    try:
        m.load_state_dict(sd)
    except Exception:
        return float("nan")                 # not a synthetic model; skip accuracy
    m.eval()
    rng = np.random.default_rng(seed)
    x, y = _synth_batch(rng, n)
    with torch.no_grad():
        return (m(x).argmax(1) == y).float().mean().item()


def ber(sd, bits_per_tensor, seed=7, min_numel=4096):
    nt = len(carrier_tensors(sd, min_numel))
    truth = _payload_bits(bits_per_tensor * nt, seed)
    read = decode_qim_int8(sd, bits_per_tensor=bits_per_tensor, min_numel=min_numel)
    k = min(len(truth), len(read))
    return float((truth[:k] != read[:k]).mean()) if k else float("nan")


def sanitize_halfstep(sd, bits_per_tensor=256, min_numel=4096):
    """Flip the parity of the top-K int8 codes (one level each). Minimal, targeted."""
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sd.items()}
    for k in carrier_tensors(out, min_numel):
        flat = out[k].reshape(-1).to(torch.float32)
        scale = _int8_scale(flat)
        codes = torch.round(flat / scale).clamp(-127, 127)
        idx = _select_int8_carriers(codes, bits_per_tensor)
        v = codes[idx]
        step = torch.where(v >= 0, torch.ones_like(v), -torch.ones_like(v))
        codes[idx] = (v + step).clamp(-127, 127)       # move one level -> parity flips
        out[k] = (codes * scale).reshape(out[k].shape).to(out[k].dtype)
    return out


def sanitize_randflip(sd, frac=0.02, min_numel=4096, seed=0):
    """Blunt baseline: flip the lowest code bit of a random `frac` of ALL carrier weights."""
    g = torch.Generator().manual_seed(seed)
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sd.items()}
    for k in carrier_tensors(out, min_numel):
        flat = out[k].reshape(-1).to(torch.float32)
        scale = _int8_scale(flat)
        codes = torch.round(flat / scale).clamp(-127, 127)
        mask = torch.rand(codes.numel(), generator=g) < frac
        step = torch.where(codes >= 0, torch.ones_like(codes), -torch.ones_like(codes))
        codes[mask] = (codes[mask] + step[mask]).clamp(-127, 127)
        out[k] = (codes * scale).reshape(out[k].shape).to(out[k].dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=256)
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(ZOO, "manifest.csv"))
    stego = df[df.method == "qim_int8"]
    rows = []
    for _, r in stego.iterrows():
        sd = torch.load(os.path.join(_ROOT, r.path), map_location="cpu")
        base = dict(base_id=r.base_id, quant=r.quant,
                    ber0=ber(sd, args.bits), acc0=accuracy(sd))
        for name, fn in [("halfstep", sanitize_halfstep),
                         ("randflip", sanitize_randflip)]:
            s = fn(sd, bits_per_tensor=args.bits) if name == "halfstep" else fn(sd)
            rows.append(dict(method=name, **base,
                             ber1=ber(s, args.bits), acc1=accuracy(s)))
    res = pd.DataFrame(rows)
    res["ber_gain"] = res.ber1 - res.ber0
    res["acc_drop"] = res.acc0 - res.acc1
    res.to_csv(os.path.join(OUT, "sanitize_summary.csv"), index=False)
    print(res.groupby("method")[["ber0", "ber1", "ber_gain", "acc0", "acc1", "acc_drop"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()

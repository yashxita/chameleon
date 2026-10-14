"""
Phase 8 - weight steganalysis: BENIGN payload embedding for building a test corpus.

This module hides a random bitstring (NOT malware, NOT executable code) inside the
float weights of a model, so we have stego/clean pairs to train and test a detector
on. This mirrors Chameleon Phases 1-2, where we embedded benign data into images and
audio. The payload here is os.urandom bits; nothing is executed and nothing is decoded
back into anything runnable.

Two schemes, both acting on the top-K highest-magnitude weights of each chosen tensor
(the carriers an attacker would use, since large weights tolerate small changes):

  lsb        : overwrite the lowest mantissa bit of each carrier weight with a payload bit.
  qim_parity : magnitude-adaptive quantization-index-modulation. For weight w, step
               Delta = 2^(floor(log2|w|)) * 2^(1 - n), quantize v = round(w/Delta), then
               push v by +-1 so that (v mod 2) equals the payload bit, and store v*Delta.
               This is the family MEASER uses; it leaves the carrier sitting exactly on a
               magnitude-adaptive lattice, which is the fingerprint Phase 8 looks for.

Only the detector and the sanitizer are the research contribution. This file exists so
the detector has something to detect.
"""

import os
import numpy as np
import torch


def _payload_bits(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def carrier_tensors(state_dict, min_numel=4096):
    """Float tensors large enough to carry a payload, in a fixed order."""
    out = []
    for k in sorted(state_dict):
        t = state_dict[k]
        if torch.is_tensor(t) and t.dtype.is_floating_point and t.numel() >= min_numel:
            out.append(k)
    return out


def _topk_idx(flat, k):
    return torch.topk(flat.abs(), k, sorted=False).indices


def embed_lsb(state_dict, bits_per_tensor=256, seed=0, min_numel=4096):
    sd = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state_dict.items()}
    bits = _payload_bits(bits_per_tensor * len(carrier_tensors(sd, min_numel)), seed)
    p = 0
    for k in carrier_tensors(sd, min_numel):
        flat = sd[k].reshape(-1).to(torch.float32)
        idx = _topk_idx(flat, bits_per_tensor)
        b = torch.from_numpy(bits[p:p + bits_per_tensor].astype(np.int32)); p += bits_per_tensor
        ints = flat[idx].view(torch.int32)
        ints = (ints & ~1) | b
        flat[idx] = ints.view(torch.float32)
        sd[k] = flat.reshape(sd[k].shape).to(sd[k].dtype)
    return sd


def embed_qim_parity(state_dict, bits_per_tensor=256, n=1, seed=0, min_numel=4096):
    sd = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state_dict.items()}
    bits = _payload_bits(bits_per_tensor * len(carrier_tensors(sd, min_numel)), seed)
    p = 0
    for k in carrier_tensors(sd, min_numel):
        flat = sd[k].reshape(-1).to(torch.float32)
        idx = _topk_idx(flat, bits_per_tensor)
        w = flat[idx]
        b = torch.from_numpy(bits[p:p + bits_per_tensor].astype(np.float32)); p += bits_per_tensor
        e = torch.floor(torch.log2(w.abs().clamp_min(1e-12)))
        delta = torch.pow(2.0, e - (n - 1))
        v = torch.round(w / delta)
        wrong = (v.to(torch.int64) % 2).to(torch.float32) != b
        v = v + torch.where(wrong, torch.sign(w).clamp_min(1.0), torch.zeros_like(w))
        flat[idx] = v * delta
        sd[k] = flat.reshape(sd[k].shape).to(sd[k].dtype)
    return sd


def _int8_scale(t):
    return t.abs().max().clamp_min(1e-12) / 127.0


def _anchor_mag(codes):
    """Magnitude of the single largest code; this weight defines the quantization scale and is
    never modified, so embed and decode recompute the same scale."""
    return codes.abs().max()


def _select_int8_carriers(codes, k):
    """Deterministic carrier set reproducible from the stored model: the k largest-|code|
    positions STRICTLY below the anchor magnitude, ties broken by index. Carriers are kept
    below the anchor during embedding, so the top-k set is identical at decode time."""
    mag = codes.abs()
    amag = _anchor_mag(codes)
    score = torch.where(mag < amag, mag, torch.full_like(mag, -1.0))
    score = score * codes.numel() - torch.arange(codes.numel(), dtype=score.dtype)
    idx = torch.topk(score, k, sorted=False).indices
    return idx.sort().values          # fixed order so bit<->carrier alignment is reproducible


def embed_qim_int8(state_dict, bits_per_tensor=256, seed=0, min_numel=4096):
    """Payload that SURVIVES int8 quantization: it lives in the parity of the int8 codes.
    Quantize carriers to int8 codes, set the parity of the top-K codes to the payload bits,
    then store the dequantized floats. A plain-int8 clean model has natural code parity; this
    one has a forced, balanced, periodic parity on its largest codes. This is the realistic
    'ships as a quantized GGUF/int8 model' case (MEASER-style survivability)."""
    sd = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state_dict.items()}
    bits = _payload_bits(bits_per_tensor * len(carrier_tensors(sd, min_numel)), seed)
    p = 0
    for k in carrier_tensors(sd, min_numel):
        flat = sd[k].reshape(-1).to(torch.float32)
        scale = _int8_scale(flat)
        codes = torch.round(flat / scale).clamp(-127, 127)
        amag = _anchor_mag(codes).item()
        idx = _select_int8_carriers(codes, bits_per_tensor)
        b = torch.from_numpy(bits[p:p + bits_per_tensor].astype(np.int64)); p += bits_per_tensor
        v = codes[idx].to(torch.int64)
        away = v + torch.where(v >= 0, torch.ones_like(v), -torch.ones_like(v))
        toward = v - torch.where(v >= 0, torch.ones_like(v), -torch.ones_like(v))
        flip = torch.where(away.abs() < amag, away, toward)   # keep carriers below anchor
        v = torch.where((v % 2) != b, flip, v)
        codes[idx] = v.to(torch.float32)
        sd[k] = (codes * scale).reshape(sd[k].shape).to(sd[k].dtype)
    return sd


def decode_qim_int8(state_dict, payload_state_free_seed=0, bits_per_tensor=256, min_numel=4096):
    """Re-read the parity bits (for bit-error-rate). Recomputes scale and top-K from the model."""
    out = []
    for k in carrier_tensors(state_dict, min_numel):
        flat = state_dict[k].reshape(-1).to(torch.float32)
        scale = _int8_scale(flat)
        codes = torch.round(flat / scale).clamp(-127, 127)
        idx = _select_int8_carriers(codes, bits_per_tensor)
        out.append((codes[idx].to(torch.int64) % 2).to(torch.uint8).numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.uint8)


EMBEDDERS = {
    "lsb": embed_lsb,
    "qim_parity": embed_qim_parity,
    "qim_int8": embed_qim_int8,
}


def embed(state_dict, method, **kw):
    return EMBEDDERS[method](state_dict, **kw)

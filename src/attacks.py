"""
Phase 5 - adversarial attacks against the Chameleon steganalysis detectors.

Threat model
------------
The attacker holds a stego file (image or audio) that a detector would flag,
and adds a small perturbation so the detector calls it "cover". The question
the project asks is not "can the detector be fooled" (always yes, given enough
budget) but "how small can the perturbation be, and what does it cost in
perceptual quality". So every attack here works in REAL FILE UNITS and every
adversarial sample is re-quantized to what a saved file can actually hold:

  image : budget eps in grey levels (L_inf, 0-255 scale), result rounded to uint8
  audio : budget eps in int16 steps (L_inf), result rounded to the 16-bit grid

Rounding matters. A "perturbation" of 0.4 grey levels is not a perturbation at
all once the image is written to a PNG, and attacks that ignore this report
evasion that no real file could achieve.

Attacks
-------
fgsm       one signed-gradient step (Goodfellow et al.)
pgd        iterated, random start, projected onto the eps-ball (Madry et al.)
pgd_bpda   pgd with a straight-through estimator for the TLU clamp. Both the
           image and waveform models clamp their residual to +-3 (TLU); the
           clamp's gradient is exactly zero wherever the residual is larger,
           which is a form of gradient masking. BPDA (Athalye et al.) keeps the
           forward pass identical and pretends the clamp is the identity on the
           backward pass, so gradients flow again.
transfer   black box. pgd is run on a SURROGATE model the attacker owns, and the
           resulting samples are scored by a different TARGET model.

Every attack tries to make the detector output class 0 (cover).
"""

import contextlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TLU

COVER = 0


@dataclass(frozen=True)
class Units:
    """How a model's input range maps to real file units."""
    name: str
    scale: float   # float units per file step is 1/scale
    lo: float
    hi: float

    def quantize(self, x):
        return torch.round(x * self.scale) / self.scale


IMAGE_UNITS = Units("image_grey_level", 255.0, 0.0, 1.0)
AUDIO_UNITS = Units("audio_int16_step", 32768.0, -1.0, 32767.0 / 32768.0)


class PerSample(nn.Module):
    """Run a model one sample at a time.

    AudioSpectrogramCNN normalizes with spec.mean() / spec.std() taken over the
    WHOLE batch tensor, so a sample's output depends on which other samples
    share its batch. That is not a property a deployed detector should have,
    and it makes per-sample gradients meaningless (each sample's gradient picks
    up terms from every other sample). Evaluating it one sample at a time gives
    the per-file normalization the model's own design intends.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.cat([self.model(x[i:i + 1]) for i in range(x.shape[0])], dim=0)


class FixedNormSpectrogram(nn.Module):
    """AudioSpectrogramCNN with its batch statistics replaced by constants.

    The original forward normalizes each log-mel spectrogram by the mean and
    std of the WHOLE BATCH tensor. For a random batch of 32 clips those
    statistics are nearly constant, so the trained weights have effectively
    learned a fixed normalization; that is the setting in which the Phase 3
    numbers (0.635 balanced / 0.680 AUC) were measured. Applied to one file at
    a time the per-file statistics differ wildly, and the same weights fall to
    0.401 AUC. Fixing (mu, sigma) to training-set values gives a detector whose
    output depends on the file alone, which is what an attacker is attacking
    and what a deployment would run.
    """

    def __init__(self, model, mu, sigma):
        super().__init__()
        self.model = model
        self.register_buffer("mu", torch.tensor(float(mu)))
        self.register_buffer("sigma", torch.tensor(float(sigma)))

    def forward(self, x):
        m = self.model
        spec = m.to_db(m.melspec(x))
        spec = (spec - self.mu) / (self.sigma + 1e-6)
        z = m.pool(m.features(spec)).flatten(1)
        return m.classifier(z)


@contextlib.contextmanager
def straight_through_tlu(model):
    """BPDA for TLU: same forward, identity backward."""
    patched = []
    for m in model.modules():
        if isinstance(m, TLU):
            def fwd(x, _m=m):
                return x + (torch.clamp(x, -_m.threshold, _m.threshold) - x).detach()
            m.forward = fwd
            patched.append(m)
    try:
        yield
    finally:
        for m in patched:
            del m.forward


def _ste_quantize(x, units):
    """Round to the file grid in the forward pass, identity in the backward."""
    return x + (units.quantize(x) - x).detach()


def _logits_and_grad(model, x_adv, units, target):
    x_in = x_adv.clone().detach().requires_grad_(True)
    logits = model(_ste_quantize(x_in, units))
    loss = F.cross_entropy(logits, target, reduction="sum")
    grad, = torch.autograd.grad(loss, x_in)
    return logits.detach(), grad


def _project(x_adv, x0, eps_f, units):
    x_adv = torch.min(torch.max(x_adv, x0 - eps_f), x0 + eps_f)
    return x_adv.clamp(units.lo, units.hi)


def fgsm(model, x, eps, units, bpda=False):
    """One signed-gradient step of size eps toward the cover class."""
    if eps <= 0:
        return x.clone()
    eps_f = eps / units.scale
    target = torch.full((x.shape[0],), COVER, dtype=torch.long, device=x.device)
    ctx = straight_through_tlu(model) if bpda else contextlib.nullcontext()
    with ctx:
        _, grad = _logits_and_grad(model, x, units, target)
    x_adv = _project(x - eps_f * grad.sign(), x, eps_f, units)
    return units.quantize(x_adv)


def pgd(model, x, eps, units, steps=10, alpha=None, random_start=True, bpda=False):
    """Projected gradient descent toward the cover class.

    The forward pass inside the loop already sees the quantized sample (via a
    straight-through round), so the model output used to decide "has this one
    been fooled" is the output the saved file would really produce. For each
    sample the first iterate that fools the model is kept; samples that never
    fool it return the final iterate.
    """
    if eps <= 0:
        return x.clone()
    eps_f = eps / units.scale
    alpha = alpha if alpha is not None else 2.5 * eps_f / steps
    target = torch.full((x.shape[0],), COVER, dtype=torch.long, device=x.device)

    x_adv = x.clone()
    if random_start:
        x_adv = _project(x_adv + torch.empty_like(x).uniform_(-eps_f, eps_f), x, eps_f, units)

    best = units.quantize(x_adv).clone()
    fooled = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

    ctx = straight_through_tlu(model) if bpda else contextlib.nullcontext()
    with ctx:
        for _ in range(steps):
            logits, grad = _logits_and_grad(model, x_adv, units, target)
            newly = (logits.argmax(1) == COVER) & ~fooled
            if newly.any():
                best[newly] = units.quantize(x_adv)[newly]
                fooled |= newly
            if fooled.all():
                break
            x_adv = _project(x_adv - alpha * grad.sign(), x, eps_f, units)

    # the loop checks each iterate before stepping, so check the last one too
    with torch.no_grad():
        q = units.quantize(x_adv)
        last = model(q).argmax(1) == COVER
        newly = last & ~fooled
        best[newly] = q[newly]
        fooled |= newly
        best[~fooled] = q[~fooled]
    return best


def transfer(surrogate, x, eps, units, steps=10, bpda=False):
    """Black-box transfer: craft on the surrogate, caller scores on the target."""
    return pgd(surrogate, x, eps, units, steps=steps, bpda=bpda)


# ---------------------------------------------------------------------------
# Perceptual cost
# ---------------------------------------------------------------------------

def psnr_db(ref, adv, peak=1.0):
    """Per-sample PSNR in dB for tensors in [0, peak]."""
    mse = ((ref - adv) ** 2).flatten(1).mean(1)
    out = 10 * torch.log10(peak ** 2 / mse.clamp_min(1e-20))
    return torch.where(mse == 0, torch.full_like(out, float("inf")), out)


def snr_db(ref, adv):
    """Per-sample SNR in dB: signal power over perturbation power."""
    sig = (ref ** 2).flatten(1).mean(1)
    noise = ((ref - adv) ** 2).flatten(1).mean(1)
    out = 10 * torch.log10(sig.clamp_min(1e-20) / noise.clamp_min(1e-20))
    return torch.where(noise == 0, torch.full_like(out, float("inf")), out)


def frac_changed(ref, adv, units):
    """Fraction of samples (pixels / audio samples) that differ by >= 1 file step."""
    return ((ref - adv).abs() * units.scale >= 0.5).float().flatten(1).mean(1)
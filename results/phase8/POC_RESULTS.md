# Phase 8: steganalysis and sanitization of AI model weights (proof of concept)

Extends Chameleon from image/audio to AI model files. Two legs:
1. **Detect** a hidden payload in model weights, telling an embedding lattice apart from
   ordinary quantization, and generalising to an embedding method held out of training.
2. **Sanitize**: remove a quantization-surviving payload with the smallest change, and show it
   beats the blunt baseline (random bit flipping) that prior work relies on.

All numbers below are the sandbox smoke test: 12 small CNNs trained from different seeds,
split **by base checkpoint** (7 train / 5 test, so train and test never share a model).
Payloads are **benign random bitstrings**. Replace with real checkpoints on Colab
(`--source torchvision`) before drawing final conclusions.

## Payloads tested
- `lsb` - overwrite the lowest mantissa bit of top-magnitude float weights.
- `qim_parity` - magnitude-adaptive QIM parity in float weights (MEASER family).
- `qim_int8` - parity of the top int8 codes; **survives int8 quantization** (ships as a
  quantized GGUF/int8 model). Decode bit-error-rate is 0.000 before and after int8 re-quant.

## Leg 1: detection

| test | view | balanced | AUC |
|---|---|---|---|
| held-out checkpoint (all methods) | quantizer-only baseline | 0.583 | 0.717 |
| held-out checkpoint (all methods) | **full features** | **0.783** | **0.888** |
| per method: lsb | full | 0.750 | 0.785 |
| per method: qim_parity | full | 0.850 | 1.000 |
| per method: qim_int8 (survives quant) | full | 0.750 | 0.880 |
| blind -> unseen qim_int8 family | full | 0.500 | 0.620 |

Reading:
- The full feature (carrier-vs-quantizer-control contrast) beats the quantizer-only baseline
  on unseen checkpoints. So the signal is the payload, not the presence of quantization. This
  is the control the review demanded.
- The float QIM lattice is near-perfectly detected (AUC 1.00); the quantization-surviving
  `qim_int8` payload is detected at AUC 0.88 via the int8 code-parity feature. Detecting a
  payload that survives int8 is the part the prior art (Model X-Ray: fp32 only; MEASER: 0%
  detection by existing scanners) does not do.
- LSB in the mantissa is only weakly detectable (AUC 0.79), as expected: a float mantissa LSB
  is already near-random, the same lesson as audio LSB in Chameleon.
- **Honest limit:** blind generalisation to a *completely unseen* embedding family is weak
  (AUC 0.62). The feature must be told the family exists to catch it well.

## Leg 2: sanitization (on the quantization-surviving qim_int8 payload)

| method | payload BER before | payload BER after | accuracy drop |
|---|---|---|---|
| targeted half-step (ours) | 0.000 | **0.944** | 0.000 |
| random bit flip (blunt baseline) | 0.000 | 0.073 | 0.000 |

Reading: the targeted sanitizer, which only flips the parity of the detected carrier codes,
destroys the payload (bit-error 0 -> 0.94) while touching almost nothing, whereas the blunt
baseline barely dents it (0 -> 0.07) at the same budget. That contrast is the sanitization
contribution.

**Honest limit:** on this trivial synthetic task both methods show ~0 accuracy drop, so the POC
does not yet demonstrate the *utility* advantage (targeted preserving accuracy while blunt
harms it). That needs real models on Colab, where random bit flipping strong enough to remove
the payload is known to degrade utility (MEASER). Permutation (NeuPerm) should be added as a
third baseline.

## What this establishes, and what it does not
- Establishes: a controlled detection signal for QIM payloads including the quantization-
  surviving case, and a targeted sanitizer that disrupts the payload far better than the blunt
  baseline at equal change.
- Does NOT yet establish: real-model results, the utility advantage of targeted vs blunt
  sanitization, blind cross-family detection, or novelty (a prior-art search by the IP cell
  decides that). Treat as a promising first sprint, not a finished invention.

## Next
Real checkpoints (`--source torchvision`), add the permutation baseline, measure utility on a
real task, and test blind detection across QIM step families. Keep private until the IP cell
advises on filing order (file before any public disclosure).

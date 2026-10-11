# Phase 4 results: cross-modal transfer

One shared convolutional trunk, two parameter-free front ends (image: fixed KV high-pass;
audio: second difference, per-clip standardisation, reshape to 160 x 200, TLU). Metrics are
on the held-out test split: balanced accuracy and AUC, mean +- sd over 3 seeds (0, 1, 2).
Audio is restricted to echo hiding (plus covers), the only audio method the detectors can
see on these covers. Image uses all three methods. Command:

    python3 cross_modal.py --device cuda --epochs 12 --audio-methods echo_hiding \
        --lr-audio 1e-3 --joint-domain-bn --seed S --tag _v2[_sS]

| Condition | Target | Balanced acc | AUC |
|---|---|---|---|
| image only (control) | image | 0.769 +- 0.018 | 0.865 +- 0.010 |
| audio only (control) | audio | 0.674 +- 0.119 | 0.797 +- 0.126 |
| zero-shot, image trunk | audio | 0.500 +- 0.000 | 0.500 +- 0.004 |
| zero-shot, audio trunk | image | 0.436 +- 0.021 | 0.432 +- 0.009 |
| BatchNorm recalibration, image trunk | audio | 0.506 +- 0.008 | 0.522 +- 0.008 |
| BatchNorm recalibration, audio trunk | image | 0.572 +- 0.036 | 0.607 +- 0.044 |
| fine-tune 4 epochs, from image | audio | 0.621 +- 0.015 | 0.695 +- 0.019 |
| fine-tune 4 epochs, from audio | image | 0.666 +- 0.020 | 0.749 +- 0.038 |
| joint, one BatchNorm per modality | image | 0.728 +- 0.019 | 0.837 +- 0.006 |
| joint, one BatchNorm per modality | audio | 0.739 +- 0.031 | 0.860 +- 0.028 |

Per seed, audio only: 0.722, 0.761, 0.538 balanced. Joint, audio: 0.747, 0.705, 0.765.

## Findings

1. **Zero-shot transfer fails in both directions.** Image trunk on audio: exactly chance
   (AUC 0.500). Audio trunk on image: reproducibly slightly below chance (AUC 0.43), which
   is not useful transfer.
2. **One weak positive signal, in one direction only.** Re-estimating only the BatchNorm
   statistics on unlabeled image data lifts the audio-trained trunk to AUC 0.61 on images
   (above chance in all 3 seeds), while the image-trained trunk gains nothing on audio
   (0.52). Missing control: the same recalibration on an untrained trunk. The fixed
   high-pass front end may be doing the work, so this is not evidence of transferred
   knowledge yet.
3. **A shared trunk with a BatchNorm per modality serves both modalities.** Joint image
   accuracy is a consistent 0.04 below the image control (0.728 vs 0.769, about two sd).
   Joint audio is 0.739 vs a noisy control of 0.674. The audio control is unreliable
   because in seed 2 it only started learning in the last two epochs (0.538); without that
   seed it averages 0.742, level with the joint model. What can be said: the joint model
   trained reliably on audio in 3 of 3 seeds, the audio-only model in 2 of 3. That is
   suggestive, from 3 seeds, not established.
4. **Fine-tuning from the other modality did not reach from-scratch accuracy**, but this
   comparison is not matched (4 epochs at a lower rate versus 12 epochs), so it shows only
   that a short fine-tune is not enough.
5. **Why audio is restricted to echo hiding.** The first run used all three audio methods
   with one shared BatchNorm: audio only was at chance (0.500, AUC 0.505) and joint
   training collapsed the image side (0.525), so no transfer number from that run was
   interpretable. Two thirds of audio stego types are undetectable on these covers, and
   image and audio residuals have very different activation statistics.

## Limitations

Three seeds. Validation curves are noisy (image validation swings between 0.53 and 0.76
across epochs) and checkpoints are chosen on validation. Image and audio tasks differ in
difficulty. The audio front end (folding 32000 samples into a 160 x 200 grid) may be a poor
fit, so the zero-shot failure may be about this design and not a fundamental limit.

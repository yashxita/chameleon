# Phase 3, run 1 — archived results (pre-architecture-revision)

Snapshot of the first complete Phase 3 run, kept because the architecture
changes that followed make this a genuine before/after ablation rather than
superseded junk.

## What produced these

- Covers: 400 per modality, integer-exact augmentation (the fixed generator)
- `ImageStegoCNN` **without** the ABS activation and TLU truncation
- `AudioWaveformCNN` **without** any residual front end, using max pooling
- `AudioSpectrogramCNN` — identical to the current version
- 40 epochs, lr 1e-4, batch 32, seed 0

## Headline numbers

| Run | Balanced | AUC | Cover recall |
|---|---|---|---|
| image_cnn, all methods | 0.504 | 0.516 | 0.283 |
| image_cnn, held out dct_qim | 0.503 | 0.522 | 0.417 |
| audio_spectrogram, all methods | 0.653 | 0.707 | 0.800 |
| audio_spectrogram, held out echo | 0.552 | 0.652 | 1.000 |
| audio_waveform, all methods | 0.518 | 0.533 | 0.333 |
| audio_waveform, held out echo | 0.500 | 0.496 | 0.550 |

Classical baseline on the same dataset: image 0.723 balanced (0.703 held-out),
audio 0.632 (0.635 held-out) — i.e. the classical detector beat every CNN on
images in this run.

## Why they are kept

Two diagnoses came out of these curves and both belong in the write-up:

1. `image_cnn` fit the training set (loss 0.695 → 0.615) while validation AUC
   never left 0.48–0.54 — it learned nothing transferable. Adding ABS + TLU
   lifted validation AUC from 0.586 to 0.722 in a controlled A/B on identical
   data and budget.
2. `audio_waveform` never left initialization at all, loss pinned at 0.693 =
   ln 2 for all 40 epochs.

Note `train_history_audio_spectrogram_excl_echo_hiding.csv` and
`train_history_audio_waveform_excl_echo_hiding.csv` are not in this snapshot;
every test-results file is.

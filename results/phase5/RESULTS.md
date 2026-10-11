# Phase 5 and 6 results: adversarial limit analysis, explainability, defense

All numbers are means over 3 seeds on the held-out test split. A detector is attacked
only on stego files it catches when clean (a stego file it already misses is evaded for
free and left untouched). Every adversarial file is rounded to what a real PNG / WAV can
hold, and the headline results were re-checked by saving real files and re-scoring them.

Budgets are L-infinity: grey levels for images, int16 steps for audio. "Cost" is the
PSNR / SNR of the adversarial file against the stego file it was made from. The
embeddings' own distortion, for comparison: image LSB 58 dB, image DCT-QIM 48 dB,
audio echo hiding 22 dB, audio LSB 83 dB.

## Headline table

| Detector | Attack | Evades 50% at | Evades 90% at | Max evasion |
|---|---|---|---|---|
| Image CNN | random noise (control) | not reached | not reached | 0.02 (at 16 grey levels) |
| Image CNN | FGSM | 2 grey levels, 42 dB | not reached | 0.73 |
| Image CNN | PGD | **1 grey level, 51 dB** | **1 grey level, 51 dB** | 1.00 |
| Image CNN | PGD + BPDA | 1 grey level, 51 dB | 1 grey level, 51 dB | 1.00 |
| Audio waveform CNN | random noise (control) | not reached | not reached | 0.34 |
| Audio waveform CNN | PGD | 4096 steps, 12.6 dB | not reached | 0.63 |
| Audio waveform CNN | PGD + BPDA | **1024 steps, 23.5 dB** | 4096 steps, 12.2 dB | 0.96 |
| Audio waveform CNN | transfer (spectrogram -> waveform) | 4096 steps, 14.2 dB | not reached | 0.69 |
| Audio spectrogram CNN | PGD | not reached | not reached | 0.40 |
| Audio spectrogram CNN | PGD + BPDA | not reached | not reached | 0.35 |
| Audio spectrogram CNN | transfer (waveform -> spectrogram) | not reached | not reached | 0.00 |

Full table: `phase5_summary.md` / `phase5_summary.csv`. Figures: `fig_balanced_acc_vs_eps.png`,
`fig_limit_cost_vs_evasion.png`, `fig_per_method.png`, `fig_gradcam.png`, `fig_defense.png`.

## Findings

1. **The image detector is adversarially fragile, not noise fragile.** A 1-grey-level PGD
   perturbation (51 dB PSNR, the same size as the embedding itself) evades 100% of
   detected files. Random sign noise 16x larger evades 2%. A 1-grey-level perturbation
   changes about 51-54% of pixels, so it would also destroy an LSB payload: evading the
   detector this way is not the same as delivering a message. Hiding a payload that
   survives needs the perturbation applied to the cover before embedding.
2. **Audio is different: evasion costs more distortion than the payload.** Echo hiding
   sits at 22 dB SNR. Evading half of the waveform detector's catches needs 23.5 dB with
   the strongest attack, i.e. roughly as much added distortion as the embedding; 90% needs
   12 dB, about 10 dB worse. The spectrogram detector's echo detection holds at 100% up to
   512 steps and never drops below about 45% even at 12 dB.
3. **Gradient masking is real on the audio waveform model, absent on the image model.**
   On audio, PGD+BPDA beats plain PGD by a wide margin (50% evasion at 23.5 dB vs 12.6 dB,
   and 0.96 vs 0.63 at the top budget). On images PGD and PGD+BPDA are identical. I
   expected the TLU clamp to mask image gradients and tested it: FGSM+BPDA plateaus at the
   same ~0.7 as FGSM, so the hypothesis was wrong. FGSM is weak because one signed step
   overshoots a highly non-linear surface; iterating fixes it.
4. **The spectrogram model's reported accuracy depends on batch composition.** Its forward
   pass normalizes by the mean and std of the whole batch. In random batches of 32 that is
   nearly constant (0.635 balanced / 0.680 AUC, as reported). Scored one file at a time it
   is 0.428 / 0.401, below chance. Replacing the batch statistics with fixed training-set
   constants gives a batch-independent detector (0.618 / 0.811), which is what was attacked.
   Any deployment of this model needs the fixed constants (`attacks.FixedNormSpectrogram`).
5. **Grad-CAM** (`fig_gradcam.png`): P(stego) goes from 1.00 to 0.12-0.22 under the
   1-grey-level attack while the attention maps collapse from broad regions to one or two
   isolated blobs.
6. **Defense, negative result.** 80 steps of PGD adversarial fine-tuning (CPU-sized budget)
   did not make the image detector robust: PGD at 1 grey level still evades 100%, clean
   balanced accuracy fell from 0.84 to 0.71 (cover recall 0.82 -> 0.61). At eps 8 plain PGD
   shows an apparent recovery (balanced ~0.68) that PGD+BPDA removes (0.36): that is
   gradient masking, an artifact of a weaker attack, not robustness. This says the budget
   was far too small, not that the defense cannot work; run it longer on a GPU (below).

## Limitations (state these in the report)

- The audio detectors mostly detect echo hiding only: LSB audio is undetectable on these
  synthetic covers even before any attack (Phase 3 finding), so audio "clean TPR" is
  0.36 (waveform) and 0.57 (spectrogram).
- Small attack sets (48 stego and 72 cover files per seed, 3 seeds) and 8-10 PGD steps,
  chosen to fit a 2-core CPU. Error bands in the figures are 1 sd over only 3 seeds.
- The image transfer (black-box) attack is NOT run: it needs a competent surrogate, and a
  plain CNN trained here stayed at chance on CPU. `limit_analysis.py` uses
  `models/image_cnn_excl_dct_qim.pt` if present (same architecture trained without
  DCT-QIM); run `--attacks transfer --tag _transfer` once it is available.
- Covers come from 15 source photographs and synthetic audio (see the Phase 3 limitations).

## Reproduce

```bash
cd src
python3 limit_analysis.py --detector image_cnn --seeds 0 1 2 --n-per-method 16 \
    --pgd-steps 8 --eps 0 1 2 4 8 16 --attacks random fgsm pgd pgd_bpda
python3 limit_analysis.py --detector audio_waveform --seeds 0 1 2
python3 limit_analysis.py --detector audio_waveform --seeds 0 1 2 --eps 512 1024 2048 4096 --tag _hi
python3 limit_analysis.py --detector audio_spectrogram --seeds 0 1 2
python3 limit_analysis.py --detector audio_spectrogram --seeds 0 1 2 --eps 512 1024 2048 4096 --tag _hi
python3 limit_analysis.py --detector image_cnn --seeds 0 1 2 --n-per-method 16 --pgd-steps 8 \
    --eps 0 1 2 4 8 16 --attacks fgsm_bpda --tag _fgsmbpda
python3 explain.py --eps 1
python3 adv_train.py --steps 80 --eps 2
python3 limit_analysis.py --detector image_cnn_advtrained --seeds 0 1 2 --n-per-method 16 \
    --pgd-steps 8 --eps 0 1 2 4 8 --attacks pgd pgd_bpda
python3 plot_phase5.py
```

Needs `models/image_cnn_allmethods.pt`, `audio_spectrogram_allmethods.pt`,
`audio_waveform_allmethods.pt` (checkpoints are gitignored; they come from the Phase 3 run).

## Full-strength defense on a GPU (Colab)

```bash
python3 adv_train.py --steps 1500 --pgd-steps 5 --eps 2 --batch-size 32
```

## Phase 4 (cross-modal): code ready, needs a GPU run

`cross_modal.py` trains one shared trunk with parameter-free image and audio front ends
and evaluates single-modality controls, zero-shot transfer both ways, BatchNorm-only
recalibration, fine-tuning, and joint training. It runs end to end on a tiny CPU subset
(`--smoke`); the real run needs a GPU (about 40 minutes on a T4):

```bash
python3 cross_modal.py --device cuda --epochs 12
```

Results go to `results/phase4/cross_modal_results.csv`. The audio front end adds a
per-clip standardization before the TLU clamp (explained in the file's docstring).
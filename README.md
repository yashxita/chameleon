# Chameleon
### Blind Cross-Modal Steganalysis with Adversarial Limit Analysis

Watermarking course mini project.

**Team:** Yashita Puri (23BCI0166), Kuriak Tom Jacob (23BCI0095), Arya Kiran Parge (23BCI0076)

---

## What this project does

Steganalysis is the task of detecting whether a piece of media (image or
audio) has hidden data embedded in it. This project builds:

1. A **blind** detector: trained across multiple different embedding
   algorithms so it generalizes to methods it wasn't specifically trained on,
   rather than only recognizing one fingerprint.
2. A **cross-modal** study: detectors for both images and audio, with an
   evaluation of whether what one modality's detector learns transfers to
   the other, and whether a jointly-trained model helps or hurts.
3. An **adversarial limit analysis**: attacking our own trained detectors
   with FGSM/PGD-style perturbations and sweeping the perturbation strength
   to find the exact point where detection collapses, evaluated alongside
   the perceptual cost (PSNR/SNR) of the attack.

## Project status

- [x] **Phase 1** -- setup, scope, environment
- [x] **Phase 2** -- cover dataset generation + multi-method embedding pipeline
- [x] **Phase 3** -- blind steganalysis models (image + audio CNNs, classical baseline)
- [ ] **Phase 4** -- cross-modal transfer analysis
- [ ] **Phase 5** -- adversarial attacks + limit analysis
- [ ] **Phase 6** -- explainability + basic defense
- [ ] **Phase 7** -- consolidation, report, demo

## Repo structure

```
src/
  data_generation.py     # cover images (crops of real photographs, integer-
                         # exact augmentation) and synthesized cover audio
  embedding_image.py     # LSB, LSB-matching, DCT-QIM embedding for images
  embedding_audio.py     # LSB, LSB-matching, echo hiding for audio
  build_dataset.py       # all embedding methods x payload rates over every
                         # cover; writes manifest.csv with a strict
                         # cover-level train/val/test split
  dataset_loader.py      # PyTorch datasets; re-roots manifest paths onto
                         # whichever checkout it runs from
  models.py              # ImageStegoCNN, AudioSpectrogramCNN, AudioWaveformCNN
  classical_baseline.py  # handcrafted residual features + random forest
  train.py               # CNN training, balanced-accuracy checkpointing
  run_phase3.py          # runs all six Phase 3 configurations
  summarize_phase3.py    # joins every result into one comparison table
chameleon_phase3_colab.ipynb   # self-contained GPU run of all of Phase 3
data/
  covers/  stego/        # generated, gitignored
sample_data/             # small committed sample of the generated data
results/
  manifest.csv           # full dataset manifest (label, method, rate, split)
  embedding_quality.csv  # PSNR (image) / SNR (audio) per sample
  sample_covers_grid.png # 8 real covers: visual check of cover diversity
  phase3_summary*.csv    # consolidated Phase 3 results
  test_results_*.csv     # per-run test metrics
  train_history_*.csv    # per-run training curves
  archive_v1/            # first full run, kept as an ablation (see below)
```

## Dataset

Development environment didn't have reliable access to download standard
steganalysis corpora (BOSSbase, ALASKA2), so covers are generated
programmatically. Everything is seeded and exactly reproducible. **If you have
access to BOSSbase or a similar corpus, point `IMG_COVER_DIR`/`AUD_COVER_DIR`
in `build_dataset.py` at your own covers** -- the embedding and dataset logic
doesn't depend on how covers were sourced.

- **Images:** 1500 covers, 256x256 grayscale, cropped from 15 real
  photographs bundled with scikit-image. 3 methods (LSB, LSB matching,
  DCT-QIM) x 3 payload rates (0.1 / 0.2 / 0.4 bpp) = 13,500 stego.
- **Audio:** 1500 covers, 2 s at 16 kHz mono, synthesized (FM "voice-like"
  tones, chirps, filtered noise, tone chords). 3 methods (LSB, LSB matching,
  echo hiding) x 3 payload rates (0.05 / 0.1 / 0.2 bps) = 13,500 stego.
- **Split:** 70/15/15 train/val/test (1050/225/225 covers per modality),
  enforced at the cover level -- a cover and all nine of its stego variants
  always share a split. Verified: zero covers span more than one split.

**Cover augmentation is integer-exact on purpose.** Images are augmented
only by cropping, flips and 90-degree rotations, and colour sources
contribute a single channel rather than a floating-point luma. An earlier
version resampled, jittered brightness and added sub-quantization noise,
which randomized the least significant bit plane of every cover *before*
embedding and erased exactly the statistic spatial steganalysis relies on.
Fixing it lifted the classical baseline from 0.621 to 0.723 balanced
accuracy on images, and from 0.479 to 0.632 on audio, with imperceptibility
unchanged. Audio background noise was removed for the same reason.

Mean imperceptibility:

| Modality | Method | Mean quality |
|---|---|---|
| Image | LSB | 58.1 dB PSNR |
| Image | LSB matching | 58.1 dB PSNR |
| Image | DCT-QIM | 48.5 dB PSNR |
| Audio | LSB | 83.1 dB SNR |
| Audio | LSB matching | 83.1 dB SNR |
| Audio | Echo hiding | 22.3 dB SNR |

## Phase 3 results

All numbers are on the held-out test split. Covers are only ~10% of test
samples, so raw accuracy is misleading -- a detector that answers "stego"
to everything scores ~90%. **Balanced accuracy** (mean of cover recall and
stego recall) and **AUC** are the honest metrics, and per-method accuracy
is only meaningful next to cover recall.

### Detector performance

| Detector | Modality | Training | Balanced | AUC | Cover recall |
|---|---|---|---|---|---|
| **CNN** (`ImageStegoCNN`) | image | all methods | **0.818** | **0.918** | 0.827 |
| **CNN** (`ImageStegoCNN`) | image | held out DCT-QIM | **0.783** | **0.879** | 0.804 |
| Random forest | image | all methods | 0.752 | -- | 0.876 |
| Random forest | image | held out DCT-QIM | 0.771 | -- | 0.862 |
| CNN (`AudioSpectrogramCNN`) | audio | all methods | 0.635 | 0.680 | 0.573 |
| CNN (`AudioSpectrogramCNN`) | audio | held out echo hiding | 0.551 | 0.623 | 0.684 |
| CNN (`AudioWaveformCNN`) | audio | all methods | 0.610 | 0.640 | 0.822 |
| CNN (`AudioWaveformCNN`) | audio | held out echo hiding | 0.503 | 0.503 | 0.840 |
| Random forest | audio | all methods | 0.626 | -- | 0.702 |
| Random forest | audio | held out echo hiding | 0.563 | -- | 0.644 |

### Blind generalization

The test of the project's central claim: train without one method, then
measure detection of that method alongside the methods that were seen.

| Detector | Held-out method | Acc. on held-out | Acc. on seen methods | Cover recall |
|---|---|---|---|---|
| Image CNN | DCT-QIM | **0.767** | 0.756 / 0.761 | 0.804 |
| Image random forest | DCT-QIM | 0.643 | 0.763 / 0.633 | 0.862 |
| Audio spectrogram CNN | echo hiding | 0.379 | 0.449 / 0.424 | 0.684 |
| Audio waveform CNN | echo hiding | 0.176 | 0.160 / 0.160 | 0.840 |

### Findings

1. **Blind detection holds for images.** The image CNN, never shown a single
   DCT-QIM sample, detects DCT-QIM at 0.767 -- as well as the two spatial
   methods it *was* trained on -- while keeping cover recall at 0.804. A
   transform-domain method is caught by a detector trained only on
   spatial-domain methods. This is the result the project set out to test.
2. **The CNN beats the classical baseline on images**, 0.818 vs 0.752
   balanced, and by a wider margin on the unseen method (0.767 vs 0.643).
3. **Audio detection is limited to echo hiding.** Both audio CNNs detect it
   well (spectrogram 0.970, waveform 0.836) and both fail on sample-domain
   LSB. This is a property of the covers, not the models: the cover audio's
   LSB plane has lag-1 autocorrelation +0.0007 *before* embedding -- already
   indistinguishable from coin flips -- because consecutive samples of a
   full-scale waveform at 16 kHz differ by thousands of quantization steps.
   Three different waveform front ends (none, high-pass, direct LSB-plane
   extraction) all sat at chance on LSB in controlled tests.
4. **The audio held-out tests are therefore uninformative**, not negative.
   With echo hiding removed, the remaining training methods are the
   undetectable ones, so there is nothing for the detector to learn.
5. **The spectrogram/waveform comparison behaves as designed for echo
   hiding:** the spectrogram model, which sees spectral structure, is the
   stronger echo detector (0.970 vs 0.836).

### Architecture ablation (`results/archive_v1/`)

The first full run used 400 covers and earlier versions of two models:

| Model | Run 1 balanced / AUC | Run 2 balanced / AUC | Change |
|---|---|---|---|
| `ImageStegoCNN` | 0.504 / 0.516 | **0.818 / 0.918** | + ABS activation, + TLU, 400 -> 1500 covers |
| `AudioWaveformCNN` | 0.518 / 0.533 | 0.610 / 0.640 | + high-pass front end, avg instead of max pooling |
| `AudioSpectrogramCNN` | 0.653 / 0.707 | 0.635 / 0.680 | unchanged (control) |

`ImageStegoCNN` was "XuNet-inspired" but lacked XuNet's ABS activation after
the first convolution and its residual truncation (TLU); without ABS, ReLU
discards every negative residual. A controlled A/B on identical data took
validation AUC from 0.586 to 0.722 from the architecture change alone,
before the larger dataset. `AudioWaveformCNN` previously never left
initialization (training loss pinned at ln 2 = 0.693); it now learns echo
hiding. The unchanged spectrogram model holding steady is the evidence that
the improvements come from the changes, not from noise.

### Limitations

- **Cover diversity.** 1500 image covers sounds like a lot, but all are
  crops of 15 source photographs. Real corpora use 10,000+ independent
  images; results here likely overstate how well these detectors would
  generalize to arbitrary photos.
- **Synthetic audio.** Real recordings contain quiet passages where the LSB
  plane carries structure; these covers do not, which is why audio LSB is
  undetectable. A real speech/music corpus would be needed to test audio
  LSB steganalysis properly.
- **Image training hadn't converged.** The all-methods image model's best
  validation epoch was its last (25 of 25), so more epochs would likely help.
- Single seed. Differences of a couple of points between runs (e.g. random
  forest all-methods 0.752 vs held-out 0.771) are within run-to-run noise.

## How to reproduce

**Recommended -- Colab (about 45 minutes, free T4 GPU):** upload
`chameleon_phase3_colab.ipynb`, set Runtime -> Change runtime type -> T4 GPU,
then Runtime -> Run all. It rebuilds the entire dataset from seed, so nothing
needs uploading, and downloads the results as a zip at the end. The dataset
build takes a few minutes and the six training runs about 30.

**Locally:**

```bash
pip install -r requirements.txt
cd src
python data_generation.py        # 1500 cover images + 1500 cover clips
python build_dataset.py          # all methods x rates, writes manifest.csv
python dataset_loader.py         # integrity check
python classical_baseline.py     # random forest baseline
python run_phase3.py --smoke     # minutes: confirms the pipeline works
python run_phase3.py             # all six CNN runs
python summarize_phase3.py       # consolidated tables
```

On CPU only, expect the full CNN batch to take several hours.

## Next: Phase 4

Cross-modal transfer. Planned design: modality-specific *parameter-free*
residual front ends (KV filter for images; high-pass plus a reshape of the
32,000-sample waveform to a 160x200 map for audio) feeding one fully shared
convolutional trunk, so zero-shot transfer involves no audio-fitted weights
at all. Conditions: single-modality controls, zero-shot transfer in both
directions, BatchNorm-recalibration-only transfer, full fine-tuning, and
joint training. Given finding 3, the audio side of every condition is
effectively about echo hiding.

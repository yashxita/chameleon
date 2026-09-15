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
- [ ] **Phase 3** -- blind steganalysis models (image + audio CNNs, classical baseline)
- [ ] **Phase 4** -- cross-modal transfer analysis
- [ ] **Phase 5** -- adversarial attacks + limit analysis
- [ ] **Phase 6** -- explainability + basic defense
- [ ] **Phase 7** -- consolidation, report, demo

## Repo structure

```
src/
  data_generation.py    # generates cover images (from real bundled photos,
                         # augmented) and cover audio (synthesized)
  embedding_image.py     # LSB, LSB-matching, DCT-QIM embedding for images
  embedding_audio.py     # LSB, LSB-matching, echo hiding for audio
  build_dataset.py       # runs all embedding methods x payload rates over
                          # every cover, writes manifest.csv with a strict
                          # cover-level train/val/test split
data/
  covers/                # cover images & audio (generated, gitignored)
  stego/                 # stego images & audio (generated, gitignored)
sample_data/              # small committed sample of what the generated
                           # data looks like (~60 files), for reference
results/
  manifest.csv            # full dataset manifest (label, method, rate, split)
  embedding_quality.csv   # PSNR (image) / SNR (audio) per sample
  sample_covers_grid.png  # quick visual sanity check of cover diversity
```

## Why no big external dataset (BOSSbase/ALASKA2/etc.)

Development environment didn't have reliable access to download standard
steganalysis corpora, so covers are generated programmatically instead:
images from augmenting real bundled sample photographs (crops, flips,
rotation, brightness jitter, with a variance filter to reject blank/low
texture patches), audio via synthesis (FM "voice-like" tones, chirps,
filtered noise, tone chords). Everything is seeded, so it's exactly
reproducible. **If you have access to BOSSbase or a similar corpus, you can
substitute it by pointing `IMG_COVER_DIR`/`AUD_COVER_DIR` in
`build_dataset.py` at your own cover files** -- the embedding and dataset
logic doesn't depend on how the covers were sourced.

## How to reproduce the dataset

```bash
pip install -r requirements.txt

cd src
python3 data_generation.py   # writes 400 cover images + 400 cover audio clips
python3 build_dataset.py     # embeds all methods x rates, writes manifest.csv
```

Expect ~3-4 minutes total and ~400MB of generated data (not committed to
the repo, see `.gitignore`).

## Dataset summary (current)

- **Images:** 400 covers (256x256 grayscale), 3 embedding methods (LSB,
  LSB matching, DCT-QIM) x 3 payload rates (0.1 / 0.2 / 0.4 bpp) = 3600
  stego samples
- **Audio:** 400 covers (2s, 16kHz mono), 3 embedding methods (LSB, LSB
  matching, echo hiding) x 3 payload rates (0.05 / 0.1 / 0.2 bps) = 3600
  stego samples
- **Split:** 70/15/15 train/val/test, enforced at the cover level (a
  cover and all its stego variants stay in the same split)

Mean imperceptibility (see `results/embedding_quality.csv` for full
breakdown):

| Modality | Method | Mean quality |
|---|---|---|
| Image | LSB | ~58 dB PSNR |
| Image | LSB matching | ~58 dB PSNR |
| Image | DCT-QIM | ~48.6 dB PSNR |
| Audio | LSB | ~82.5 dB SNR |
| Audio | LSB matching | ~82.5 dB SNR |
| Audio | Echo hiding | ~22.3 dB SNR |

## Next: Phase 3

Blind steganalysis CNNs (image + audio), a classical baseline (handcrafted
features + shallow classifier), and held-out-method evaluation to verify
the "blind" claim actually holds.

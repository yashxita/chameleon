"""
Phase 2 - Cover media generation.

We don't have reliable internet access to big steganalysis datasets like
BOSSbase/ALASKA2 in this environment, so we build a reasonably sized and
diverse cover dataset ourselves:

  - Images: random crops of skimage's bundled sample photographs (real
    photographs, not synthetic), augmented only by operations that are
    exact on 8-bit integers.

  - Audio: synthesized programmatically (tones, FM "voice-like" synthesis,
    filtered noise, chirps) since we can't reliably fetch external audio
    corpora here. Each clip is randomized so no two are identical.

Everything is deterministic given a seed, so the dataset is reproducible.

Why the augmentation is integer-exact (important)
--------------------------------------------------
The first version of this file built covers through a floating-point
pipeline: rgb2gray, anti-aliased resize/rotate at arbitrary angles,
brightness/contrast jitter, and finally `+ N(0, 0.003)` noise before
quantizing to uint8. That noise is about 0.003 * 255 = 0.77 grey levels,
roughly one quantization step, and the interpolation contributed more of
the same.

The effect was to randomize the least significant bit plane of every
cover BEFORE anything was embedded. Measured on those covers, the LSB
plane had mean 0.5005 and horizontal autocorrelation -0.007 -- i.e. it was
already indistinguishable from coin flips -- and embedding LSB at 0.1 bpp
moved the chi-square pairs-of-values statistic only from 467 to 402.

That matters because the LSB plane's natural structure is exactly what
spatial-domain steganalysis detects. Randomizing it first makes LSB and
LSB-matching close to undetectable by construction, which is why both the
classical baseline and the CNNs sat at chance on them. It is a property of
the cover generator, not a finding about steganalysis.

So augmentation here is restricted to operations that map 8-bit values to
themselves exactly: cropping, horizontal/vertical flips, and 90-degree
rotations. No resampling, no gain/bias, no additive noise. Colour sources
contribute one channel at a time rather than a floating-point luma, for
the same reason. Diversity instead comes from the number of distinct crop
positions available (tens of thousands per base image), and exact
duplicates are rejected explicitly.

The audio generator had the same flaw in a louder form: it added
`N(0, 0.02)` background noise to signals that are then written as 16-bit
PCM. 0.02 of full scale is ~655 int16 quantization steps, which randomizes
roughly the low nine bits and makes sample-domain audio LSB steganalysis
impossible by construction. That noise is now off by default (see
BACKGROUND_NOISE below).
"""

import os
import numpy as np
from PIL import Image
import skimage.data as skdata
import soundfile as sf

RNG_SEED = 42
IMG_SIZE = 256
AUDIO_SR = 16000
AUDIO_DURATION = 2.0  # seconds

# Reject near-blank, low-texture crops: they carry almost no steganalysis
# signal either way. Expressed in grey levels (the old float threshold of
# 0.03 on a [0,1] image is 0.03 * 255).
MIN_PATCH_STD = 7.0

# Additive background noise for synthesized audio, in units of full scale.
# Anything at or above 1/32767 (~3e-5) starts corrupting the 16-bit LSB
# plane; the original 0.02 destroyed the low nine bits. Keep at 0.0 unless
# you specifically want to study detection under a noise floor.
BACKGROUND_NOISE = 0.0

# Base images that are real photographs, delivered by skimage as uint8.
# Excluded on purpose: `horse` (boolean mask), `checkerboard` (synthetic,
# degenerate bit planes), and anything smaller than IMG_SIZE in either
# dimension (`page`, `text`) since we no longer upscale.
_CANDIDATE_BASES = [
    "astronaut", "camera", "cat", "chelsea", "coffee", "coins",
    "moon", "rocket", "brick", "grass", "gravel", "clock",
    "hubble_deep_field", "immunohistochemistry", "retina",
]


def _load_base_images(size=IMG_SIZE, verbose=True):
    """Load bundled skimage photographs, keeping them as raw uint8.

    Returns a list of (name, array) where array is uint8 and either 2D
    grayscale or 3D RGB, and is at least `size` in both spatial dimensions.
    Nothing is converted to float here -- that is the entire point.
    """
    imgs = []
    for name in _CANDIDATE_BASES:
        try:
            arr = getattr(skdata, name)()
        except Exception as e:
            if verbose:
                print(f"  skip {name}: {e}")
            continue

        if arr.dtype != np.uint8:
            if verbose:
                print(f"  skip {name}: dtype {arr.dtype}, not uint8")
            continue
        if arr.ndim == 3 and arr.shape[2] != 3:
            if verbose:
                print(f"  skip {name}: {arr.shape[2]} channels")
            continue
        if arr.ndim not in (2, 3):
            if verbose:
                print(f"  skip {name}: {arr.ndim} dimensions")
            continue
        if min(arr.shape[:2]) < size:
            if verbose:
                print(f"  skip {name}: {arr.shape[:2]} smaller than {size}")
            continue

        imgs.append((name, arr))
    return imgs


def generate_cover_images(n_images, out_dir, size=IMG_SIZE, seed=RNG_SEED, verbose=True):
    """Generate n_images distinct grayscale cover images as 8-bit PNGs.

    Augmentation is crop + flip + 90-degree rotation only, all exact on
    uint8, so the natural least-significant-bit structure of the source
    photographs survives into the covers.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    base_imgs = _load_base_images(size=size, verbose=verbose)

    # Every candidate must load. Crops are drawn by indexing into this list,
    # so losing even one source photograph (say, a future scikit-image
    # release dropping a bundled image) would not crash -- it would quietly
    # produce a completely different dataset from the one the published
    # results were computed on. Fail loudly instead.
    loaded = [name for name, _ in base_imgs]
    missing = [name for name in _CANDIDATE_BASES if name not in loaded]
    if missing:
        raise RuntimeError(
            f"Base image(s) failed to load: {missing}. The dataset is defined by "
            f"exactly these {len(_CANDIDATE_BASES)} source photographs; continuing "
            f"without them would silently generate a different dataset. Check the "
            f"scikit-image installation (verified with 0.26, where all are bundled)."
        )
    if verbose:
        print(f"Loaded {len(base_imgs)} base images for augmentation: "
              f"{[n for n, _ in base_imgs]}")

    manifest = []
    seen = set()
    rejected_flat = 0
    rejected_dup = 0

    for i in range(n_images):
        patch = None
        for _attempt in range(200):
            _name, base = base_imgs[rng.integers(0, len(base_imgs))]

            # Colour sources contribute a single channel. A weighted luma
            # would be a float and would re-randomise the low bit.
            plane = base[:, :, rng.integers(0, 3)] if base.ndim == 3 else base

            h, w = plane.shape
            top = rng.integers(0, h - size + 1)
            left = rng.integers(0, w - size + 1)
            candidate = plane[top:top + size, left:left + size]

            # exact-on-uint8 augmentation only
            if rng.random() < 0.5:
                candidate = np.fliplr(candidate)
            if rng.random() < 0.5:
                candidate = np.flipud(candidate)
            k = int(rng.integers(0, 4))
            if k:
                candidate = np.rot90(candidate, k)
            candidate = np.ascontiguousarray(candidate)

            if candidate.std() < MIN_PATCH_STD:
                rejected_flat += 1
                continue

            digest = hash(candidate.tobytes())
            if digest in seen:
                rejected_dup += 1
                continue

            seen.add(digest)
            patch = candidate
            break

        if patch is None:
            raise RuntimeError(
                f"Could not find a distinct, textured crop for image {i} after 200 "
                f"attempts. Lower MIN_PATCH_STD or add more base images."
            )

        fname = f"cover_{i:05d}.png"
        Image.fromarray(patch, mode="L").save(os.path.join(out_dir, fname))
        manifest.append(fname)

    if verbose:
        print(f"Wrote {len(manifest)} cover images to {out_dir}")
        print(f"  rejected {rejected_flat} low-texture crops, {rejected_dup} duplicates")
    return manifest


def _fm_voice_like(t, rng):
    """Crude FM-synthesis signal loosely mimicking voiced-speech formants."""
    f0 = rng.uniform(90, 220)  # pitch
    sig = np.zeros_like(t)
    formants = [rng.uniform(500, 900), rng.uniform(1200, 1800), rng.uniform(2200, 3000)]
    for k, fF in enumerate(formants, start=1):
        mod = np.sin(2 * np.pi * rng.uniform(2, 6) * t)
        inst_freq = fF + 40 * mod
        sig += (1.0 / k) * np.sin(2 * np.pi * inst_freq * t / max(1, k))
    carrier = np.sin(2 * np.pi * f0 * t)
    out = 0.6 * carrier + 0.4 * sig
    envelope = np.clip(np.sin(np.pi * t / t[-1]), 0.15, 1.0)
    return out * envelope


def _chirp(t, rng):
    f_start = rng.uniform(80, 400)
    f_end = rng.uniform(400, 3000)
    k = (f_end - f_start) / t[-1]
    return np.sin(2 * np.pi * (f_start * t + 0.5 * k * t ** 2))


def _filtered_noise(t, rng):
    noise = rng.normal(0, 1, t.shape)
    # simple moving-average low-pass to make it less harsh/white
    win = rng.integers(3, 25)
    kernel = np.ones(win) / win
    return np.convolve(noise, kernel, mode="same")


def _tone_chord(t, rng):
    n_tones = rng.integers(2, 5)
    sig = np.zeros_like(t)
    for _ in range(n_tones):
        f = rng.uniform(100, 2000)
        amp = rng.uniform(0.2, 0.6)
        sig += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return sig


def generate_cover_audio(n_clips, out_dir, sr=AUDIO_SR, duration=AUDIO_DURATION,
                         seed=RNG_SEED + 1, background_noise=BACKGROUND_NOISE,
                         verbose=True):
    """Generate n_clips synthetic cover audio clips (mono, 16-bit PCM WAV).

    background_noise is in units of full scale and defaults to 0. See the
    note at the top of this file: at the original 0.02 it was roughly 655
    int16 quantization steps, which randomized the low nine bits of every
    sample and made sample-domain LSB steganalysis undetectable by
    construction.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    generators = [_fm_voice_like, _chirp, _filtered_noise, _tone_chord]
    manifest = []
    for i in range(n_clips):
        gen = generators[rng.integers(0, len(generators))]
        sig = gen(t, rng)

        if background_noise:
            sig = sig + rng.normal(0, background_noise, sig.shape)

        # normalize to avoid clipping, leave headroom for embedding later
        peak = np.max(np.abs(sig)) + 1e-9
        sig = 0.85 * sig / peak

        fname = f"cover_{i:05d}.wav"
        sf.write(os.path.join(out_dir, fname), sig.astype(np.float32), sr, subtype="PCM_16")
        manifest.append(fname)

    if verbose:
        print(f"Wrote {len(manifest)} cover audio clips to {out_dir}")
        if not background_noise:
            print("  (background noise disabled to preserve the 16-bit LSB plane)")
    return manifest


if __name__ == "__main__":
    import argparse
    import os as _os

    # 1500 rather than the original 400: the first full training run was
    # limited by cover diversity as much as by architecture -- only 280
    # covers reached the training split, and all 2,520 stego samples were
    # derived from those same 280. Crops are free, so this is the cheapest
    # available increase in diversity. It remains capped by the 15 source
    # photographs, which is the honest limitation to report.
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=1500)
    parser.add_argument("--n-clips", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    _PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    IMG_OUT = _os.path.join(_PROJECT_ROOT, "data/covers/images")
    AUD_OUT = _os.path.join(_PROJECT_ROOT, "data/covers/audio")
    generate_cover_images(args.n_images, IMG_OUT, seed=args.seed)
    generate_cover_audio(args.n_clips, AUD_OUT, seed=args.seed + 1)

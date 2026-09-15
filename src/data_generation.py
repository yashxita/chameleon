"""
Phase 2 - Cover media generation.

We don't have reliable internet access to big steganalysis datasets like
BOSSbase/ALASKA2 in this environment, so we build a reasonably sized and
diverse cover dataset ourselves:

  - Images: start from skimage's bundled sample images (real photographs,
    not synthetic), convert to grayscale (standard in steganalysis
    literature), then expand via random crops/flips/rotations/brightness
    jitter to get hundreds of distinct 256x256 covers.

  - Audio: synthesized programmatically (tones, FM "voice-like" synthesis,
    filtered noise, chirps) since we can't reliably fetch external audio
    corpora here. Each clip is randomized so no two are identical.

Everything is deterministic given a seed, so the dataset is reproducible.
"""

import os
import numpy as np
from PIL import Image
import skimage.data as skdata
from skimage.color import rgb2gray
from skimage.transform import resize, rotate
import soundfile as sf

RNG_SEED = 42
IMG_SIZE = 256
AUDIO_SR = 16000
AUDIO_DURATION = 2.0  # seconds


def _load_base_images():
    """Load every bundled skimage sample image that is a plain 2D/3D photo."""
    names = [
        "astronaut", "camera", "cat", "chelsea", "coffee", "coins",
        "horse", "moon", "page", "rocket", "text", "brick", "grass",
        "gravel", "checkerboard", "clock", "hubble_deep_field",
        "immunohistochemistry", "retina",
    ]
    imgs = []
    for name in names:
        try:
            fn = getattr(skdata, name)
            arr = fn()
            if arr.ndim == 3:
                arr = rgb2gray(arr)  # rgb2gray already returns floats in [0, 1]
            elif arr.ndim == 2:
                arr = arr.astype(np.float64)
                if arr.max() > 1.0:  # raw uint8-range grayscale, normalize
                    arr = arr / 255.0
            else:
                continue
            imgs.append(arr)
        except Exception as e:
            print(f"  skip {name}: {e}")
    return imgs


def generate_cover_images(n_images, out_dir, size=IMG_SIZE, seed=RNG_SEED):
    """Generate n_images distinct grayscale cover images via augmentation
    of real bundled photographs, saved as 8-bit PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    base_imgs = _load_base_images()
    print(f"Loaded {len(base_imgs)} base images for augmentation")

    manifest = []
    min_std = 0.03  # reject near-blank low-texture patches (bad for steganalysis)
    for i in range(n_images):
        patch = None
        for _attempt in range(30):
            base = base_imgs[rng.integers(0, len(base_imgs))]
            h, w = base.shape

            # random crop (if base is big enough), else resize up first
            if h < size or w < size:
                base_r = resize(base, (size + 20, size + 20), anti_aliasing=True)
                h, w = base_r.shape
            else:
                base_r = base

            top = rng.integers(0, h - size + 1)
            left = rng.integers(0, w - size + 1)
            candidate = base_r[top:top + size, left:left + size]
            if candidate.std() >= min_std:
                patch = candidate
                break
        if patch is None:
            patch = candidate  # fall back to last attempt rather than crash

        # random flip
        if rng.random() < 0.5:
            patch = np.fliplr(patch)
        if rng.random() < 0.5:
            patch = np.flipud(patch)

        # small random rotation then re-crop to size to avoid black borders
        angle = rng.uniform(-8, 8)
        if abs(angle) > 0.5:
            patch = rotate(patch, angle, mode="reflect")

        # brightness / contrast jitter
        gain = rng.uniform(0.85, 1.15)
        bias = rng.uniform(-0.05, 0.05)
        patch = np.clip(patch * gain + bias, 0, 1)

        # tiny noise so augmented duplicates aren't bit-identical
        patch = np.clip(patch + rng.normal(0, 0.003, patch.shape), 0, 1)

        img8 = (patch * 255).astype(np.uint8)
        fname = f"cover_{i:05d}.png"
        Image.fromarray(img8, mode="L").save(os.path.join(out_dir, fname))
        manifest.append(fname)

    print(f"Wrote {len(manifest)} cover images to {out_dir}")
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


def generate_cover_audio(n_clips, out_dir, sr=AUDIO_SR, duration=AUDIO_DURATION, seed=RNG_SEED + 1):
    """Generate n_clips synthetic cover audio clips (mono, 16-bit PCM WAV)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    generators = [_fm_voice_like, _chirp, _filtered_noise, _tone_chord]
    manifest = []
    for i in range(n_clips):
        gen = generators[rng.integers(0, len(generators))]
        sig = gen(t, rng)

        # mix in a touch of background noise for realism
        sig = sig + rng.normal(0, 0.02, sig.shape)

        # normalize to avoid clipping, leave headroom for embedding later
        peak = np.max(np.abs(sig)) + 1e-9
        sig = 0.85 * sig / peak

        fname = f"cover_{i:05d}.wav"
        sf.write(os.path.join(out_dir, fname), sig.astype(np.float32), sr, subtype="PCM_16")
        manifest.append(fname)

    print(f"Wrote {len(manifest)} cover audio clips to {out_dir}")
    return manifest


if __name__ == "__main__":
    import os as _os
    _PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    IMG_OUT = _os.path.join(_PROJECT_ROOT, "data/covers/images")
    AUD_OUT = _os.path.join(_PROJECT_ROOT, "data/covers/audio")
    generate_cover_images(400, IMG_OUT)
    generate_cover_audio(400, AUD_OUT)

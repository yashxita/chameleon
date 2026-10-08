"""
Audio steganography embedding methods.

Audio covers are float32 PCM in [-1, 1] at 16kHz mono. We convert to 16-bit
integer PCM for the sample-domain methods (LSB) since that's the standard
way LSB audio steganography is defined, and work in float domain for
echo hiding.

payload_rate is expressed in bits-per-sample (bps), analogous to
bits-per-pixel for images.

Performance note: LSB matching was a per-sample Python loop and is now
vectorised. It draws from the random generator in exactly the same order
and quantity as before, so a given seed produces the same stego audio;
`_reference_lsb_matching_audio` keeps the loop version for checking.
"""

import numpy as np


def _to_int16(sig_float):
    return np.clip(sig_float * 32767.0, -32768, 32767).astype(np.int16)


def _to_float(sig_int16):
    return sig_int16.astype(np.float64) / 32767.0


def embed_lsb_audio(cover, payload_rate, rng):
    """Classic LSB replacement on 16-bit PCM samples. Same 'easy to detect'
    baseline idea as image LSB."""
    pcm = _to_int16(cover).copy()
    n = len(pcm)
    n_embed = int(n * payload_rate)

    positions = rng.choice(n, size=n_embed, replace=False)
    bits = rng.integers(0, 2, size=n_embed, dtype=np.int16)

    pcm[positions] = (pcm[positions] & ~1) | bits
    return _to_float(pcm).astype(np.float32)


def embed_lsb_matching_audio(cover, payload_rate, rng):
    """+-1 LSB matching on 16-bit PCM samples, analogous to the image version:
    avoids the detectable bias of forced LSB replacement. Vectorised; one
    direction is drawn per mismatching sample, in position order, exactly
    as the original loop drew them."""
    pcm = _to_int16(cover).astype(np.int32)
    n = len(pcm)
    n_embed = int(n * payload_rate)

    positions = rng.choice(n, size=n_embed, replace=False)
    bits = rng.integers(0, 2, size=n_embed)

    vals = pcm[positions]
    mismatch = (vals & 1) != bits
    k = int(mismatch.sum())
    if k:
        directions = rng.choice([-1, 1], size=k)
        v = vals[mismatch]
        new = v + directions
        out_of_range = (new < -32768) | (new > 32767)
        new[out_of_range] = v[out_of_range] - directions[out_of_range]
        vals[mismatch] = new
        pcm[positions] = vals

    pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
    return _to_float(pcm).astype(np.float32)


def embed_echo_hiding(cover, payload_rate, rng, sr=16000):
    """Echo hiding: split the signal into segments, and for each segment
    add a faint, short-delay echo. A '1' bit uses one delay (e.g. 1ms) and
    a '0' bit uses another delay (e.g. 0.5ms), inaudible under normal
    listening but recoverable via cepstral analysis. This is a genuinely
    different statistical domain from sample-LSB methods (it perturbs the
    autocorrelation structure of the signal rather than individual sample
    values), which is exactly the kind of variety a 'blind' detector needs
    to generalize across."""
    sig = cover.astype(np.float64).copy()
    n = len(sig)

    delay0 = int(0.0005 * sr)  # 0.5 ms -> bit 0
    delay1 = int(0.0010 * sr)  # 1.0 ms -> bit 1
    echo_amp = 0.08  # tuned for perceptual transparency (~22 dB SNR)

    # segment length derived from payload rate: fewer, longer segments for
    # lower payload rate (one bit per segment)
    bits_capacity = max(1, int(n * payload_rate / max(1, delay1)))
    seg_len = max(delay1 * 4, n // max(1, bits_capacity))
    n_segments = n // seg_len
    if n_segments == 0:
        return cover  # too short to embed anything meaningfully

    bits = rng.integers(0, 2, size=n_segments)
    out = sig.copy()
    for s in range(n_segments):
        start = s * seg_len
        end = start + seg_len
        segment = sig[start:end]
        delay = delay1 if bits[s] == 1 else delay0
        echo = np.zeros_like(segment)
        if delay < len(segment):
            echo[delay:] = segment[:len(segment) - delay] * echo_amp
        out[start:end] = segment + echo

    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0:
        out = out / peak * 0.98
    return out.astype(np.float32)


EMBED_METHODS_AUDIO = {
    "lsb": embed_lsb_audio,
    "lsb_matching": embed_lsb_matching_audio,
    "echo_hiding": embed_echo_hiding,
}


def snr_db(cover, stego):
    cover = cover.astype(np.float64)
    stego = stego.astype(np.float64)
    noise = cover - stego
    signal_power = np.mean(cover ** 2)
    noise_power = np.mean(noise ** 2) + 1e-12
    return 10 * np.log10(signal_power / noise_power)


def _reference_lsb_matching_audio(cover, payload_rate, rng):
    """Original per-sample loop, kept only to verify equivalence."""
    pcm = _to_int16(cover).astype(np.int32).copy()
    n = len(pcm)
    n_embed = int(n * payload_rate)
    positions = rng.choice(n, size=n_embed, replace=False)
    bits = rng.integers(0, 2, size=n_embed)
    for idx, bit in zip(positions, bits):
        cur_lsb = pcm[idx] & 1
        if cur_lsb != bit:
            direction = rng.choice([-1, 1])
            new_val = np.clip(pcm[idx] + direction, -32768, 32767)
            if (new_val & 1) != bit:
                new_val = np.clip(pcm[idx] - direction, -32768, 32767)
            pcm[idx] = new_val
    pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
    return _to_float(pcm).astype(np.float32)


if __name__ == "__main__":
    import time
    rng0 = np.random.default_rng(0)
    covers = [rng0.uniform(-0.85, 0.85, 32000).astype(np.float32) for _ in range(4)]
    covers += [np.full(32000, -1.0, np.float32), np.full(32000, 1.0, np.float32)]
    same = all(
        np.array_equal(embed_lsb_matching_audio(c, r, np.random.default_rng(42)),
                       _reference_lsb_matching_audio(c, r, np.random.default_rng(42)))
        for c in covers for r in (0.05, 0.1, 0.2))
    t = time.time(); [embed_lsb_matching_audio(covers[0], 0.2, np.random.default_rng(1)) for _ in range(5)]
    tf = (time.time() - t) / 5
    t = time.time(); [_reference_lsb_matching_audio(covers[0], 0.2, np.random.default_rng(1)) for _ in range(5)]
    tr = (time.time() - t) / 5
    print(f"lsb_matching_audio identical to original: {same}   "
          f"{tr * 1000:7.1f} ms -> {tf * 1000:6.1f} ms per clip ({tr / tf:.0f}x)")

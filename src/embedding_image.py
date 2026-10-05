"""
Image steganography embedding methods.

Each function takes a cover image (uint8, HxW grayscale) and a payload
(random bits, since for steganalysis training we don't care about message
content, only about the statistical footprint the embedding leaves), and
returns a stego image of the same shape.

payload_rate is in bits-per-pixel (bpp), matching how the steganalysis
literature (BOSSbase-style) reports capacity.

Performance note
----------------
LSB matching and DCT-QIM were originally written as per-pixel and
per-block Python loops, which made the dataset build the slowest step of
the whole project (~25 minutes for 1500 covers). Both are now vectorised
with numpy. They draw from the random generator in exactly the same order
and quantity as the loops did, so a given seed still produces the same
stego images; `_reference_*` keeps the original loop versions so that
equivalence can be checked (see the __main__ block).
"""

import numpy as np


def _make_payload_bits(n_bits, rng):
    return rng.integers(0, 2, size=n_bits, dtype=np.uint8)


def embed_lsb(cover, payload_rate, rng):
    """Classic LSB replacement: flip the least significant bit of randomly
    selected pixels to match payload bits. This is the 'easy to detect'
    baseline method, deliberately naive."""
    img = cover.copy()
    h, w = img.shape
    n_pixels = h * w
    n_embed = int(n_pixels * payload_rate)
    flat = img.flatten()

    positions = rng.choice(n_pixels, size=n_embed, replace=False)
    bits = _make_payload_bits(n_embed, rng)

    flat[positions] = (flat[positions] & 0xFE) | bits
    return flat.reshape(h, w)


def embed_lsb_matching(cover, payload_rate, rng):
    """LSB matching (+-1 embedding): instead of forcing the LSB to a value
    (which creates a detectable 'pairs of values' artifact), randomly add
    or subtract 1 when the LSB doesn't already match the payload bit.
    Harder to detect than plain LSB replacement.

    Vectorised. Positions are distinct (sampled without replacement), so
    there is no ordering dependency between them, and one direction is
    drawn per mismatching pixel in position order -- exactly the draws the
    original loop made.
    """
    h, w = cover.shape
    n_pixels = h * w
    n_embed = int(n_pixels * payload_rate)
    flat = cover.astype(np.int16).flatten()

    positions = rng.choice(n_pixels, size=n_embed, replace=False)
    bits = _make_payload_bits(n_embed, rng)

    vals = flat[positions]
    mismatch = (vals & 1) != bits
    k = int(mismatch.sum())
    if k:  # the loop made no draw at all when nothing mismatched
        directions = rng.choice([-1, 1], size=k)
        v = vals[mismatch]
        new = v + directions
        # at 0 or 255 the step would clip and leave parity wrong, so step
        # the other way instead (same rule as the original)
        out_of_range = (new < 0) | (new > 255)
        new[out_of_range] = v[out_of_range] - directions[out_of_range]
        vals[mismatch] = new
        flat[positions] = vals

    return flat.reshape(h, w).astype(np.uint8)


def embed_dct(cover, payload_rate, rng, block_size=8):
    """Simplified JPEG-style DCT embedding: split the image into 8x8 blocks,
    take the 2D DCT, and perturb a fixed set of mid-frequency coefficients
    by rounding to even/odd (quantization-index-modulation) to encode bits.
    This mimics the family of transform-domain methods (like the general
    idea behind JMiPOD/UERD) without needing a full JPEG codec, and leaves
    a different statistical footprint than spatial-domain LSB methods,
    which is exactly the variety we need for a 'blind' detector.

    Vectorised: all selected blocks are transformed in one batched DCT.
    Bit i goes to block block_order[i // 3], coefficient i % 3 -- the same
    assignment the original nested loop made.
    """
    from scipy.fft import dctn, idctn

    bs = block_size
    img = cover.astype(np.float64)
    h, w = img.shape
    bh, bw = h // bs, w // bs
    n_blocks = bh * bw

    # use 3 mid-frequency coefficient positions per block (skip DC, skip
    # very high freq which is fragile / very low freq which is perceptually
    # sensitive)
    coeff_positions = [(2, 1), (1, 2), (2, 2)]
    n_coef = len(coeff_positions)
    max_bits = n_blocks * n_coef
    n_embed = min(int(h * w * payload_rate), max_bits)

    bits = _make_payload_bits(n_embed, rng)
    block_order = rng.permutation(n_blocks)
    quant_step = 6.0

    out = img.copy()
    if n_embed == 0:
        return np.clip(out, 0, 255).astype(np.uint8)

    # blocks that receive at least one bit, in embedding order
    n_used = -(-n_embed // n_coef)
    used = block_order[:n_used]

    grid = img[:bh * bs, :bw * bs]
    blocks = grid.reshape(bh, bs, bw, bs).transpose(0, 2, 1, 3).reshape(n_blocks, bs, bs)
    coeffs = dctn(blocks[used], axes=(1, 2), norm="ortho")

    for j, (ci, cj) in enumerate(coeff_positions):
        bit_idx = np.arange(j, n_embed, n_coef)
        if bit_idx.size == 0:
            continue
        rows = bit_idx // n_coef
        b = bits[bit_idx]
        c = coeffs[rows, ci, cj]
        q = np.round(c / quant_step)
        wrong = np.mod(q, 2) != b
        q = np.where(wrong, q + np.where(c >= q * quant_step, 1.0, -1.0), q)
        coeffs[rows, ci, cj] = q * quant_step

    # only blocks that were embedded into are written back, as before
    new_blocks = blocks.copy()
    new_blocks[used] = idctn(coeffs, axes=(1, 2), norm="ortho")
    out[:bh * bs, :bw * bs] = (
        new_blocks.reshape(bh, bw, bs, bs).transpose(0, 2, 1, 3).reshape(bh * bs, bw * bs)
    )
    return np.clip(out, 0, 255).astype(np.uint8)


EMBED_METHODS = {
    "lsb": embed_lsb,
    "lsb_matching": embed_lsb_matching,
    "dct_qim": embed_dct,
}


def psnr(cover, stego):
    cover = cover.astype(np.float64)
    stego = stego.astype(np.float64)
    mse = np.mean((cover - stego) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0) - 10 * np.log10(mse)


# ---------------------------------------------------------------------------
# Reference (original loop) implementations, kept only to verify that the
# vectorised versions above produce identical output for the same seed.
# ---------------------------------------------------------------------------

def _reference_lsb_matching(cover, payload_rate, rng):
    img = cover.copy().astype(np.int16)
    h, w = img.shape
    n_pixels = h * w
    n_embed = int(n_pixels * payload_rate)
    flat = img.flatten()
    positions = rng.choice(n_pixels, size=n_embed, replace=False)
    bits = _make_payload_bits(n_embed, rng)
    for idx, bit in zip(positions, bits):
        cur_lsb = flat[idx] & 1
        if cur_lsb != bit:
            direction = rng.choice([-1, 1])
            new_val = np.clip(flat[idx] + direction, 0, 255)
            if (new_val & 1) != bit:
                new_val = np.clip(flat[idx] - direction, 0, 255)
            flat[idx] = new_val
    return flat.reshape(h, w).astype(np.uint8)


def _reference_dct(cover, payload_rate, rng, block_size=8):
    from scipy.fftpack import dct, idct
    img = cover.copy().astype(np.float64)
    h, w = img.shape
    bh, bw = h // block_size, w // block_size
    n_blocks = bh * bw
    coeff_positions = [(2, 1), (1, 2), (2, 2)]
    max_bits = n_blocks * len(coeff_positions)
    n_embed = min(int(h * w * payload_rate), max_bits)
    bits = _make_payload_bits(n_embed, rng)
    block_order = rng.permutation(n_blocks)
    quant_step = 6.0
    bit_i = 0
    out = img.copy()
    for b_idx in block_order:
        if bit_i >= n_embed:
            break
        bi, bj = divmod(b_idx, bw)
        y0, x0 = bi * block_size, bj * block_size
        block = img[y0:y0 + block_size, x0:x0 + block_size]
        block_dct = dct(dct(block.T, norm="ortho").T, norm="ortho")
        for (ci, cj) in coeff_positions:
            if bit_i >= n_embed:
                break
            bit = bits[bit_i]
            coeff = block_dct[ci, cj]
            q = round(coeff / quant_step)
            if (q % 2) != bit:
                q += 1 if coeff >= q * quant_step else -1
            block_dct[ci, cj] = q * quant_step
            bit_i += 1
        block_rec = idct(idct(block_dct.T, norm="ortho").T, norm="ortho")
        out[y0:y0 + block_size, x0:x0 + block_size] = block_rec
    return np.clip(out, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import time
    rng0 = np.random.default_rng(0)
    covers = [rng0.integers(0, 256, (256, 256), dtype=np.uint8) for _ in range(4)]
    covers += [np.zeros((256, 256), np.uint8), np.full((256, 256), 255, np.uint8)]
    for name, fast, ref in [("lsb_matching", embed_lsb_matching, _reference_lsb_matching),
                            ("dct_qim", embed_dct, _reference_dct)]:
        same = True
        for c in covers:
            for rate in (0.1, 0.2, 0.4):
                a = fast(c, rate, np.random.default_rng(42))
                b = ref(c, rate, np.random.default_rng(42))
                same &= np.array_equal(a, b)
        t = time.time(); [fast(covers[0], 0.4, np.random.default_rng(1)) for _ in range(5)]
        tf = (time.time() - t) / 5
        t = time.time(); [ref(covers[0], 0.4, np.random.default_rng(1)) for _ in range(5)]
        tr = (time.time() - t) / 5
        print(f"{name:13s} identical to original: {same}   "
              f"{tr * 1000:7.1f} ms -> {tf * 1000:6.1f} ms per image ({tr / tf:.0f}x)")

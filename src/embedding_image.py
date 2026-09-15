"""
Image steganography embedding methods.

Each function takes a cover image (uint8, HxW grayscale) and a payload
(random bits, since for steganalysis training we don't care about message
content, only about the statistical footprint the embedding leaves), and
returns a stego image of the same shape.

payload_rate is in bits-per-pixel (bpp), matching how the steganalysis
literature (BOSSbase-style) reports capacity.
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
    Harder to detect than plain LSB replacement."""
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
            new_val = flat[idx] + direction
            new_val = np.clip(new_val, 0, 255)
            # if clipping flipped us back to wrong parity, push other way
            if (new_val & 1) != bit:
                new_val = np.clip(flat[idx] - direction, 0, 255)
            flat[idx] = new_val

    return flat.reshape(h, w).astype(np.uint8)


def embed_dct(cover, payload_rate, rng, block_size=8):
    """Simplified JPEG-style DCT embedding: split the image into 8x8 blocks,
    take the 2D DCT, and perturb a fixed set of mid-frequency coefficients
    by rounding to even/odd (quantization-index-modulation) to encode bits.
    This mimics the family of transform-domain methods (like the general
    idea behind JMiPOD/UERD) without needing a full JPEG codec, and leaves
    a different statistical footprint than spatial-domain LSB methods,
    which is exactly the variety we need for a 'blind' detector."""
    from scipy.fftpack import dct, idct

    img = cover.copy().astype(np.float64)
    h, w = img.shape
    bh, bw = h // block_size, w // block_size
    n_blocks = bh * bw

    # use 3 mid-frequency coefficient positions per block (skip DC, skip
    # very high freq which is fragile / very low freq which is perceptually
    # sensitive)
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

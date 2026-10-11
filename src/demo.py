"""
Phase 7 live demo: detect a stego file, attack the detector, show the verdict flip.

    python3 demo.py                              # picks a stego image from the test split
    python3 demo.py --file my_image.png          # your own 256x256 grayscale PNG
    python3 demo.py --modality audio             # picks an echo-hiding clip from the test split
    python3 demo.py --modality audio --file clip.wav
    python3 demo.py --method lsb_matching --eps 2

What it does
  1. scores the file with the trained detector (probability that it hides data)
  2. runs a white-box PGD attack (audio also uses BPDA) inside a size budget eps
  3. SAVES the attacked file as a real PNG / WAV, loads it back, scores it again
     (so the result survives the file format, not just the in-memory tensor)
  4. prints the quality cost (PSNR / SNR) and how many samples changed
  5. writes a picture (image) or numbers (audio) to results/demo/
"""

import os
import argparse

import numpy as np
import torch
import soundfile as sf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import attacks as A
from dataset_loader import load_manifest, filter_manifest, ImageStegoDataset, AudioStegoDataset
from limit_analysis import load_detector

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "results", "demo")


def p_stego(model, x):
    with torch.no_grad():
        return torch.softmax(model(x), 1)[0, 1].item()


def verdict(p):
    return "STEGO (hides data)" if p >= 0.5 else "COVER (looks clean)"


def pick_from_test(modality, method, seed, model):
    """First test stego file of the chosen method that the detector catches."""
    df = filter_manifest(load_manifest(), modality, "test")
    st = df[(df.label == "stego") & (df.method == method)].sample(frac=1, random_state=seed)
    DS = ImageStegoDataset if modality == "image" else AudioStegoDataset
    ds = DS(st.head(60))
    for i in range(len(ds)):
        x = ds[i][0].unsqueeze(0)
        if model(x).argmax(1).item() == 1:
            return x, st.iloc[i].path
    raise SystemExit(f"detector catches none of the first 60 {method} files; try --seed")


def load_file(modality, path):
    if modality == "image":
        img = Image.open(path).convert("L")
        if img.size != (256, 256):
            print(f"note: resizing {img.size} to 256x256")
            img = img.resize((256, 256))
        return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)[None, None]
    sig, _sr = sf.read(path, dtype="float32")
    if sig.ndim > 1:
        sig = sig[:, 0]
    sig = np.pad(sig, (0, max(0, 32000 - len(sig))))[:32000]
    return torch.from_numpy(sig)[None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["image", "audio"], default="image")
    ap.add_argument("--file", default=None, help="your own PNG (image) or WAV (audio)")
    ap.add_argument("--method", default=None, help="test-split method to pick when --file is not given")
    ap.add_argument("--eps", type=float, default=None,
                    help="budget: grey levels (image, default 1) or int16 steps (audio, default 2048)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(OUT, exist_ok=True)

    img = args.modality == "image"
    det_name = "image_cnn" if img else "audio_waveform"
    units = A.IMAGE_UNITS if img else A.AUDIO_UNITS
    eps = args.eps if args.eps is not None else (1 if img else 2048)
    method = args.method or ("lsb_matching" if img else "echo_hiding")
    model = load_detector(det_name)

    if args.file:
        x, src = load_file(args.modality, args.file), args.file
    else:
        x, src = pick_from_test(args.modality, method, args.seed, model)
    p0 = p_stego(model, x)
    print(f"\nfile      : {src}")
    print(f"detector  : {det_name}")
    print(f"BEFORE    : P(stego) = {p0:.3f}  ->  {verdict(p0)}")
    if p0 < 0.5:
        print("The detector already calls this file clean, so there is nothing to attack.")
        print("Try a different file, or a stego file made with a method it detects.")
        return

    x_adv = A.pgd(model, x, eps, units, steps=args.steps, bpda=not img)

    # save as a real file, reload, and score the file we reloaded
    if img:
        path = os.path.join(OUT, "adversarial.png")
        Image.fromarray(np.round(x_adv[0, 0].numpy() * 255).astype(np.uint8), mode="L").save(path)
        back = load_file("image", path)
    else:
        path = os.path.join(OUT, "adversarial.wav")
        pcm = np.clip(np.round(x_adv[0, 0].numpy() * 32768), -32768, 32767).astype(np.int16)
        sf.write(path, pcm, 16000, subtype="PCM_16")
        back = load_file("audio", path)
    p1 = p_stego(model, back)

    cost = A.psnr_db(x, back) if img else A.snr_db(x, back)
    unit_name = "grey levels" if img else "int16 steps"
    changed = A.frac_changed(x, back, units).item()
    biggest = ((back - x).abs() * units.scale).max().item()
    print(f"ATTACK    : PGD{'' if img else ' + BPDA'}, budget {eps:g} {unit_name}, {args.steps} steps")
    print(f"saved     : {path}   (real {'PNG' if img else 'WAV'}, reloaded from disk)")
    print(f"AFTER     : P(stego) = {p1:.3f}  ->  {verdict(p1)}")
    print(f"cost      : {cost.item():.1f} dB {'PSNR' if img else 'SNR'} vs the stego file, "
          f"largest change {biggest:.0f} {unit_name}, {100 * changed:.0f}% of samples changed")
    print("RESULT    : " + ("detector FOOLED" if p1 < 0.5 else "detector still catches it at this budget"))
    if img:
        print("NOTE      : changing about half the pixels would also destroy an LSB payload; the demo")
        print("            shows the detector is fragile, not that a message was delivered.")

    if img:
        diff = (back - x).abs()[0, 0].numpy() * 255
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.7))
        for a, im, t, kw in [(ax[0], x[0, 0].numpy(), f"stego\n{verdict(p0)}  P={p0:.2f}", dict(cmap="gray", vmin=0, vmax=1)),
                             (ax[1], back[0, 0].numpy(), f"after attack\n{verdict(p1)}  P={p1:.2f}", dict(cmap="gray", vmin=0, vmax=1)),
                             (ax[2], diff, f"|difference| ({cost.item():.0f} dB PSNR)", dict(cmap="Blues", vmin=0, vmax=max(1, eps)))]:
            a.imshow(im, **kw)
            a.set_title(t, fontsize=9)
            a.axis("off")
        fig.subplots_adjust(top=0.80, left=0.02, right=0.98, bottom=0.03, wspace=0.05)
        fig.savefig(os.path.join(OUT, "demo_image.png"), dpi=150)
        print("figure    : " + os.path.join(OUT, "demo_image.png"))


if __name__ == "__main__":
    main()

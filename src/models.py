"""
Phase 3 - Model architectures.

Image model: XuNet-inspired. The key idea that separates real steganalysis
CNNs from a generic image classifier is the FIRST layer: a fixed high-pass
filter (KV kernel, standard in the steganalysis literature) that converts
the image into its noise residual before any learned convolutions happen.
Embedding artifacts live in this high-frequency residual, not in the raw
pixel content, so this single design choice is what makes the network
actually about steganalysis rather than "is this a picture of a cat".

Audio model: a small CNN over the log-mel spectrogram, which plays the
same role for audio that pixel-domain CNNs play for images. We also
provide a raw-waveform 1D CNN as an architectural comparison point (both
are trained in the training script; results are compared).

What changed after the first full training run, and why
--------------------------------------------------------
The first complete Phase 3 run exposed two architectural faults. Both
models below were revised in response; AudioSpectrogramCNN was left alone
because it was the one that worked (validation AUC 0.703, detecting
echo_hiding at 0.856 while missing sample-domain LSB -- exactly the
behaviour its design predicts).

1. ImageStegoCNN sat at 0.504 balanced accuracy / 0.516 AUC while its
   training loss fell from 0.695 to 0.615, i.e. it fit the training set
   and generalized nothing. It called itself XuNet-inspired but omitted
   XuNet's two defining components:

     - the ABS activation after the first convolution. Embedding
       residuals are symmetric about zero, so the informative statistic is
       |r|. Feeding a zero-mean residual straight into BatchNorm + ReLU
       discards every negative residual -- half the evidence.
     - truncation (TLU) of the residual. Unbounded outliers from strong
       image edges dominate the activation statistics and drown out the
       embedding-scale perturbations.

   The KV filter now also rescales its input to 0-255 first, so the TLU
   threshold is expressed in the grey-level units the literature uses
   rather than in [0,1] units where a threshold of 3 would never clip.

   Measured on a held-out A/B with identical data and budget, this took
   validation AUC from 0.586 to 0.722 and accuracy off the 0.500 floor to
   0.650.

2. AudioWaveformCNN never left its initialization at all -- training loss
   held at 0.693 (= ln 2, the constant-output loss) for all 40 epochs. It
   now gets a fixed high-pass residual front end and average rather than
   max pooling, which is the better-motivated architecture.

   That did NOT fix it, and the reason is worth recording, because it is a
   property of the data rather than of the model. Three front ends were
   tried on identical data: none (the original), a high-pass second
   difference, and direct LSB-plane extraction. All three sit at
   validation AUC ~0.50.

   The cause is measurable. The cover audio's LSB plane has lag-1
   autocorrelation +0.0007 and mean 0.5004 -- it is already
   indistinguishable from coin flips before anything is embedded, so LSB
   embedding has no structure to disturb. This is intrinsic to the signal:
   consecutive samples of a full-scale waveform at 16 kHz differ by
   thousands of quantization steps, which leaves the low bit
   equidistributed. Spatial image steganalysis works precisely because
   neighbouring pixels differ by only a few grey levels, making the low
   bit partly predictable. Real recordings recover some of this in quiet
   passages; these synthesized covers have none, as the amplitude envelope
   never drops below 0.15 of full scale.

   The high-pass front end is kept because it is the correct design and
   rules out "no residual extraction" as the explanation. The remaining
   failure is a dataset limitation and should be reported as one. It is
   corroborated independently by AudioSpectrogramCNN, which detects
   echo_hiding at 0.856 while scoring 0.339 on sample-domain LSB.
"""

import torch
import torch.nn as nn
import torchaudio


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class TLU(nn.Module):
    """Truncated linear unit: clamp(x, -T, +T).

    Bounds the residual so that a handful of strong edges cannot dominate
    the statistics the later layers see. Standard preprocessing in
    steganalysis CNNs.
    """

    def __init__(self, threshold=3.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return torch.clamp(x, -self.threshold, self.threshold)

    def extra_repr(self):
        return f"threshold={self.threshold}"


class AbsLayer(nn.Module):
    """|x|. Embedding noise is symmetric about zero, so its magnitude is
    the statistic that carries information; sign does not."""

    def forward(self, x):
        return torch.abs(x)


# ---------------------------------------------------------------------------
# Image model
# ---------------------------------------------------------------------------

# The classic KV (Ker-Vetterly) high-pass residual kernel used in SRM/XuNet-
# style steganalysis preprocessing. Highlights the kind of high-frequency
# noise pattern that spatial-domain embedding disturbs.
_KV_KERNEL = torch.tensor([
    [-1,  2, -2,  2, -1],
    [ 2, -6,  8, -6,  2],
    [-2,  8, -12, 8, -2],
    [ 2, -6,  8, -6,  2],
    [-1,  2, -2,  2, -1],
], dtype=torch.float32) / 12.0


class HighPassResidual(nn.Module):
    """Fixed (non-trainable) high-pass filter applied before the learned
    conv stack. This is what lets the network see embedding noise instead
    of image content.

    Input arrives as [0, 1] floats; we rescale to 0-255 so the residual is
    in grey-level units and the TLU threshold downstream means what it
    means in the literature.
    """

    def __init__(self, scale=255.0):
        super().__init__()
        self.register_buffer("kernel", _KV_KERNEL.view(1, 1, 5, 5))
        self.scale = scale

    def forward(self, x):
        return nn.functional.conv2d(x * self.scale, self.kernel, padding=2)


class ImageStegoCNN(nn.Module):
    def __init__(self, tlu_threshold=3.0):
        super().__init__()
        self.hpf = HighPassResidual()
        self.tlu = TLU(tlu_threshold)

        # Group 1 carries the ABS. See the module docstring: without it the
        # network throws away every negative residual.
        self.g1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            AbsLayer(),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(2),            # 256 -> 128
        )
        self.g2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, padding=2),
            nn.BatchNorm2d(32), nn.Tanh(), nn.AvgPool2d(2),      # 128 -> 64
        )
        self.g3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AvgPool2d(2),   # 64 -> 32
        )
        self.g4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.AvgPool2d(2),  # 32 -> 16
        )
        self.g5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.AvgPool2d(2),  # 16 -> 8
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.tlu(self.hpf(x))
        x = self.g5(self.g4(self.g3(self.g2(self.g1(x)))))
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Audio models
# ---------------------------------------------------------------------------

class AudioSpectrogramCNN(nn.Module):
    """Log-mel-spectrogram CNN, the audio analogue of the image model above:
    a fixed, well-understood transform (mel spectrogram) first, then
    learned conv layers on top.

    Deliberately unchanged from the first full run: this is the model that
    worked, and keeping it fixed keeps that result comparable.
    """

    def __init__(self, sr=16000, n_mels=64):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=512, hop_length=160, n_mels=n_mels
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

        def block(c_in, c_out, pool=True):
            layers = [
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.AvgPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(1, 16),
            block(16, 32),
            block(32, 64),
            block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        # x: [B, 1, T] raw waveform in, compute spectrogram here so the
        # dataset loader can stay simple and shared across model types
        spec = self.melspec(x)       # [B, 1, n_mels, frames]
        spec = self.to_db(spec)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        x = self.features(spec)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# Second difference. The audio counterpart of the KV kernel: it annihilates
# locally-linear signal and leaves sample-scale perturbation behind.
_AUDIO_HP_KERNEL = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32)


class AudioHighPassResidual(nn.Module):
    """Fixed high-pass front end for raw audio.

    Samples arrive as floats in [-1, 1], where a one-LSB change to 16-bit
    PCM is 1/32768, about 3e-5 relative -- utterly swamped by the carrier.
    We rescale to int16 units so a single-LSB perturbation is O(1), then
    take a second difference to suppress the smooth underlying signal and
    leave the perturbation standing.
    """

    def __init__(self, scale=32768.0):
        super().__init__()
        self.register_buffer("kernel", _AUDIO_HP_KERNEL.view(1, 1, 3))
        self.scale = scale

    def forward(self, x):
        return nn.functional.conv1d(x * self.scale, self.kernel, padding=1)


class AudioWaveformCNN(nn.Module):
    """1D CNN on the raw waveform, as an architectural comparison against
    the spectrogram model. Sample-domain embedding methods (LSB) should be
    more visible here, since computing a mel spectrogram smooths over
    single-sample perturbations.

    Now preceded by a fixed high-pass residual and TLU, mirroring the image
    model. Pooling is average rather than max: max pooling over a residual
    keeps only the largest deviation in each window and throws away the
    rest of the distribution, which is where the statistic lives.
    """

    def __init__(self, tlu_threshold=3.0):
        super().__init__()
        self.hpf = AudioHighPassResidual()
        self.tlu = TLU(tlu_threshold)

        def block(c_in, c_out, k=9, pool=4, act="relu", use_abs=False):
            layers = [nn.Conv1d(c_in, c_out, k, padding=k // 2)]
            if use_abs:
                layers.append(AbsLayer())
            layers.append(nn.BatchNorm1d(c_out))
            layers.append(nn.Tanh() if act == "tanh" else nn.ReLU(inplace=True))
            layers.append(nn.AvgPool1d(pool))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(1, 16, act="tanh", use_abs=True),
            block(16, 32, act="tanh"),
            block(32, 64),
            block(64, 128),
            block(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.tlu(self.hpf(x))
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    img_model = ImageStegoCNN()
    dummy_img = torch.randn(4, 1, 256, 256).clamp(0, 1)
    out = img_model(dummy_img)
    print("ImageStegoCNN output shape:", out.shape)
    print("  params:", sum(p.numel() for p in img_model.parameters() if p.requires_grad))

    spec_model = AudioSpectrogramCNN()
    dummy_aud = torch.randn(4, 1, 32000).clamp(-1, 1)
    out = spec_model(dummy_aud)
    print("AudioSpectrogramCNN output shape:", out.shape)
    print("  params:", sum(p.numel() for p in spec_model.parameters() if p.requires_grad))

    wave_model = AudioWaveformCNN()
    out = wave_model(dummy_aud)
    print("AudioWaveformCNN output shape:", out.shape)
    print("  params:", sum(p.numel() for p in wave_model.parameters() if p.requires_grad))

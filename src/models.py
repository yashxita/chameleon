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
"""

import torch
import torch.nn as nn
import torchaudio


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
    of image content."""

    def __init__(self):
        super().__init__()
        kernel = _KV_KERNEL.view(1, 1, 5, 5)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        return nn.functional.conv2d(x, self.kernel, padding=2)


class ImageStegoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hpf = HighPassResidual()

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
            block(1, 16),     # 256 -> 128
            block(16, 32),    # 128 -> 64
            block(32, 64),    # 64 -> 32
            block(64, 128),   # 32 -> 16
            block(128, 128),  # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.hpf(x)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Audio models
# ---------------------------------------------------------------------------

class AudioSpectrogramCNN(nn.Module):
    """Log-mel-spectrogram CNN, the audio analogue of the image model above:
    a fixed, well-understood transform (mel spectrogram) first, then
    learned conv layers on top."""

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


class AudioWaveformCNN(nn.Module):
    """1D CNN directly on the raw waveform, as an architectural comparison
    against the spectrogram model. Sample-domain embedding methods (LSB)
    may be more visible here since spectrogram computation can smooth over
    single-sample perturbations."""

    def __init__(self):
        super().__init__()

        def block(c_in, c_out, k=9, pool=4):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, k, padding=k // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(pool),
            )

        self.features = nn.Sequential(
            block(1, 16),
            block(16, 32),
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
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


if __name__ == "__main__":
    img_model = ImageStegoCNN()
    dummy_img = torch.randn(4, 1, 256, 256)
    out = img_model(dummy_img)
    print("ImageStegoCNN output shape:", out.shape)
    print("  params:", sum(p.numel() for p in img_model.parameters() if p.requires_grad))

    spec_model = AudioSpectrogramCNN()
    dummy_aud = torch.randn(4, 1, 32000)
    out = spec_model(dummy_aud)
    print("AudioSpectrogramCNN output shape:", out.shape)
    print("  params:", sum(p.numel() for p in spec_model.parameters() if p.requires_grad))

    wave_model = AudioWaveformCNN()
    out = wave_model(dummy_aud)
    print("AudioWaveformCNN output shape:", out.shape)
    print("  params:", sum(p.numel() for p in wave_model.parameters() if p.requires_grad))
# Phase 8: weight steganalysis (PRIVATE - do not merge to public main before IP review)

Extends Chameleon from image/audio to AI model files. Detects a benign hidden payload in
model weights and separates an embedding lattice from ordinary quantization.

**Keep this on a private branch.** Public disclosure before a patent filing weakens rights.
Also add `results/phase8/zoo/` to `.gitignore` (the .pt model files are large and regenerable).

## Files
- `embed.py` - hide a benign random bitstring in weights (LSB, QIM-parity). Builds test data only.
- `features.py` - carrier-vs-quantizer-control lattice features.
- `modelzoo.py` - build many independent checkpoints + clean/stego variants + quantization; manifest split by base model.
- `detect.py` - train/test with two controls: held-out checkpoint, held-out method; baseline vs full features.

## Run (sandbox / CPU smoke test, offline)
```bash
cd src/weightstego
python3 modelzoo.py --n-models 12 --steps 120 --bits 256 --quant
python3 detect.py --topk 256
```
Outputs: `results/phase8/zoo/manifest.csv`, `results/phase8/features.csv`, `results/phase8/detect_summary.csv`.

## Run (Colab / real checkpoints)
```bash
cd src/weightstego
python3 modelzoo.py --source torchvision --n-models 8 --bits 256 --quant
python3 detect.py --topk 256
```

See `results/phase8/POC_RESULTS.md` for the first findings and the next sprint.

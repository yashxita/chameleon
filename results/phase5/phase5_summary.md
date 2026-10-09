| Detector | Attack | Clean TPR | Evade 50% at | cost | Evade 90% at | cost | Max evasion |
|---|---|---|---|---|---|---|---|
| Audio spectrogram CNN | FGSM | 0.565 | not reached |  | not reached |  | 0.03 |
| Audio spectrogram CNN | PGD | 0.565 | not reached |  | not reached |  | 0.40 |
| Audio spectrogram CNN | PGD + BPDA | 0.565 | not reached |  | not reached |  | 0.35 |
| Audio spectrogram CNN | random noise (control) | 0.565 | not reached |  | not reached |  | 0.01 |
| Audio spectrogram CNN | transfer (black-box) | 0.565 | not reached |  | not reached |  | 0.00 |
| Audio waveform CNN | FGSM | 0.361 | not reached |  | not reached |  | 0.01 |
| Audio waveform CNN | PGD | 0.361 | 4096 LSB | 12.6 dB | not reached |  | 0.63 |
| Audio waveform CNN | PGD + BPDA | 0.361 | 1024 LSB | 23.5 dB | 4096 LSB | 12.2 dB | 0.96 |
| Audio waveform CNN | random noise (control) | 0.361 | not reached |  | not reached |  | 0.34 |
| Audio waveform CNN | transfer (black-box) | 0.361 | 4096 LSB | 14.2 dB | not reached |  | 0.69 |
| Image CNN | FGSM | 0.840 | 2 grey | 42.2 dB | not reached |  | 0.73 |
| Image CNN | FGSM + BPDA | 0.840 | 2 grey | 42.1 dB | not reached |  | 0.68 |
| Image CNN | PGD | 0.840 | 1 grey | 51.0 dB | 1 grey | 51.0 dB | 1.00 |
| Image CNN | PGD + BPDA | 0.840 | 1 grey | 51.1 dB | 1 grey | 51.1 dB | 1.00 |
| Image CNN | random noise (control) | 0.840 | not reached |  | not reached |  | 0.02 |
| Image CNN, adversarially fine-tuned | PGD | 0.826 | 1 grey | 51.0 dB | 1 grey | 51.0 dB | 1.00 |
| Image CNN, adversarially fine-tuned | PGD + BPDA | 0.826 | 1 grey | 51.0 dB | 1 grey | 51.0 dB | 1.00 |
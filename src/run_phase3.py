"""
Phase 3 - run every training configuration needed to close the phase out.

Six runs, which is what Phase 3 actually requires:

  image  / image_cnn          -- all methods          (standard blind setting)
  image  / image_cnn          -- held out dct_qim     (blindness stress test)
  audio  / audio_spectrogram  -- all methods
  audio  / audio_spectrogram  -- held out echo_hiding (blindness stress test)
  audio  / audio_waveform     -- all methods          (architecture comparison)
  audio  / audio_waveform     -- held out echo_hiding

Each run is a subprocess, so one crash doesn't take the whole batch down,
and a run whose results CSV already exists is skipped unless --force, so an
interrupted batch can simply be restarted. Everything is streamed to
results/phase3_runs.log as well as to the console, so progress can be
followed without watching the terminal.

    python3 run_phase3.py --smoke     # tiny capped runs, minutes, plumbing check
    python3 run_phase3.py             # the real thing
    python3 run_phase3.py --only image_cnn_allmethods
"""

import os
import sys
import time
import argparse
import datetime
import subprocess

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
LOG_PATH = os.path.join(RESULTS_DIR, "phase3_runs.log")

RUNS = [
    {"tag": "image_cnn_allmethods",             "modality": "image", "model": "image_cnn",          "exclude": None},
    {"tag": "image_cnn_excl_dct_qim",           "modality": "image", "model": "image_cnn",          "exclude": "dct_qim"},
    {"tag": "audio_spectrogram_allmethods",     "modality": "audio", "model": "audio_spectrogram",  "exclude": None},
    {"tag": "audio_spectrogram_excl_echo_hiding", "modality": "audio", "model": "audio_spectrogram", "exclude": "echo_hiding"},
    {"tag": "audio_waveform_allmethods",        "modality": "audio", "model": "audio_waveform",     "exclude": None},
    {"tag": "audio_waveform_excl_echo_hiding",  "modality": "audio", "model": "audio_waveform",     "exclude": "echo_hiding"},
]


def emit(line, log_f):
    sys.stdout.write(line)
    sys.stdout.flush()
    log_f.write(line)
    log_f.flush()


def run_one(cfg, args, log_f):
    tag = ("smoke_" + cfg["tag"]) if args.smoke else cfg["tag"]
    results_path = os.path.join(RESULTS_DIR, f"test_results_{tag}.csv")

    if os.path.exists(results_path) and not args.force:
        emit(f"\n[skip] {tag} -- results already exist (use --force to re-run)\n", log_f)
        return "skipped", 0.0

    cmd = [
        sys.executable, "-u", os.path.join(_SRC_DIR, "train.py"),
        "--modality", cfg["modality"],
        "--model", cfg["model"],
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--patience", str(args.patience),
        "--min-epochs", str(args.min_epochs),
        "--num-workers", str(args.num_workers),
        "--tag", tag,
    ]
    if cfg["exclude"]:
        cmd += ["--exclude-method", cfg["exclude"]]
    if args.max_train_samples:
        cmd += ["--max-train-samples", str(args.max_train_samples)]

    emit(f"\n{'=' * 78}\n", log_f)
    emit(f"[run] {tag}  ({datetime.datetime.now():%Y-%m-%d %H:%M:%S})\n", log_f)
    emit(f"      {' '.join(cmd[1:])}\n", log_f)
    emit(f"{'=' * 78}\n", log_f)

    start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=_SRC_DIR)
    for line in proc.stdout:
        emit(line, log_f)
    code = proc.wait()
    elapsed = time.time() - start

    status = "ok" if code == 0 else f"FAILED (exit {code})"
    emit(f"[done] {tag}: {status} in {elapsed / 60:.1f} min\n", log_f)
    return status, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="tiny capped runs written under smoke_* tags, to check the pipeline works")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=-1,
                        help="-1 lets train.py auto-select by device")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--only", type=str, default=None,
                        help="run just this tag (see RUNS at the top of this file)")
    parser.add_argument("--force", action="store_true",
                        help="re-run even if a results CSV already exists")
    args = parser.parse_args()

    # smoke defaults: small enough to finish in minutes, big enough to prove
    # the loop trains, evaluates, checkpoints and writes its outputs
    if args.smoke:
        if args.epochs is None:
            args.epochs = 2
        if args.max_train_samples is None:
            args.max_train_samples = 300
        args.patience = 0  # don't early-stop a 2-epoch run
    elif args.epochs is None:
        args.epochs = 20

    runs = RUNS if args.only is None else [r for r in RUNS if r["tag"] == args.only]
    if not runs:
        print(f"No run matches --only {args.only!r}. Valid tags: {[r['tag'] for r in RUNS]}")
        return 1

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = []
    with open(LOG_PATH, "a", encoding="utf-8") as log_f:
        emit(f"\n\n########## phase 3 batch "
             f"({'SMOKE' if args.smoke else 'FULL'}, {datetime.datetime.now():%Y-%m-%d %H:%M:%S}) "
             f"##########\n", log_f)
        batch_start = time.time()
        for cfg in runs:
            status, elapsed = run_one(cfg, args, log_f)
            summary.append((cfg["tag"], status, elapsed))

        emit(f"\n{'=' * 78}\nBATCH SUMMARY "
             f"(total {(time.time() - batch_start) / 60:.1f} min)\n{'=' * 78}\n", log_f)
        for tag, status, elapsed in summary:
            emit(f"  {tag:40s} {status:20s} {elapsed / 60:6.1f} min\n", log_f)
        emit(f"\nLog written to {LOG_PATH}\n", log_f)

    failed = [t for t, s, _ in summary if s.startswith("FAILED")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Train the five-class vocal emotion classifier on TESS and RAVDESS (Appendix B.3).

Classes: angry, happy, neutral, pleasant_surprise, sad. Other emotions in the
corpora (calm, fearful, disgust) are skipped.

Usage:
    python scripts/train_voice_classifier.py --data data/tess data/ravdess --out models/voice
    python scripts/train_voice_classifier.py --data data/tess data/ravdess --out models/voice \
        --epochs 2000 --activation linear
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pitch_emotion import voice  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="Folders containing TESS / RAVDESS .wav files")
    ap.add_argument("--out", default="models/voice")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--activation", default="linear", choices=["linear", "relu"],
                    help="Hidden-layer activation; the appendix specifies linear")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cache = Path(args.out) / "features.npz"
    if cache.exists():
        d = np.load(cache)
        X, y = d["X"], d["y"]
    else:
        X, y = voice.load_training_data(*args.data)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=X, y=y)
    print(f"{len(y)} clips; class counts:",
          dict(zip(voice.VOICE_EMOTIONS, np.bincount(y, minlength=len(voice.VOICE_EMOTIONS)).tolist())))

    report = voice.train_classifier(X, y, args.out, epochs=args.epochs, batch_size=args.batch_size,
                                    seed=args.seed, activation=args.activation)
    print(f"Test accuracy: {report['accuracy']:.3f}")
    print(json.dumps({e: round(report[e]["recall"], 3) for e in voice.VOICE_EMOTIONS}, indent=1))


if __name__ == "__main__":
    main()

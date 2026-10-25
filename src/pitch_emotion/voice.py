"""Vocal emotion measures.

* ``praat_pitch`` (Appendix B.3): mean F0 and F0 standard deviation (pitch
  variation) via Praat, with sex-specific pitch ranges.
* ``extract_features`` / ``build_classifier`` / ``train_classifier``
  (Appendix B.3, following Gorodnichenko et al., 2023): 180 Librosa features
  (128 Mel, 40 MFCC, 12 chroma) and a dense neural network trained on TESS and
  RAVDESS to classify five vocal emotions.
* ``aggregate_segment_export`` (Appendix B.2): duration-weighted mean of
  segment-level neutrality scores, e.g. from iMotions / audEERING exports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PITCH_RANGES = {"woman": (100.0, 600.0), "man": (75.0, 500.0)}
VOICE_EMOTIONS = ("angry", "happy", "neutral", "pleasant_surprise", "sad")


# ------------------------------------------------------------------ Praat pitch
def praat_pitch(audio: str | Path, sex: str) -> dict:
    """Mean F0 and pitch variation (SD of F0) with Praat via ``parselmouth``.

    Pitch range: 100-600 Hz for women, 75-500 Hz for men; other settings are
    Praat defaults. Lower pitch variation indicates a more neutral voice.
    """
    import parselmouth

    sex = sex.lower()
    if sex not in PITCH_RANGES:
        raise ValueError(f"sex must be one of {list(PITCH_RANGES)}, got {sex!r}")
    floor, ceiling = PITCH_RANGES[sex]
    pitch = parselmouth.Sound(str(audio)).to_pitch(pitch_floor=floor, pitch_ceiling=ceiling)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if f0.size == 0:
        return {"mean_f0": np.nan, "sd_f0": np.nan, "n_voiced_frames": 0}
    return {"mean_f0": float(f0.mean()), "sd_f0": float(f0.std(ddof=1)) if f0.size > 1 else np.nan,
            "n_voiced_frames": int(f0.size)}


# ------------------------------------------------------- Librosa + classifier
def extract_features(audio: str | Path, sr: int | None = None) -> np.ndarray:
    """180 features: 40 MFCCs, 12 chroma and 128 Mel bands, each time-averaged."""
    import librosa

    y, sr = librosa.load(str(audio), sr=sr, mono=True)
    stft = np.abs(librosa.stft(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).mean(axis=1)
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr).mean(axis=1)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).mean(axis=1)
    return np.concatenate([mfcc, chroma, mel]).astype(np.float32)


def label_from_filename(path: str | Path) -> str | None:
    """Map TESS / RAVDESS file names to the five classes; other emotions -> None.

    RAVDESS: ``03-01-05-01-02-01-12.wav`` (3rd field = emotion code).
    TESS:    ``OAF_back_angry.wav`` / ``YAF_dog_ps.wav`` (last field = emotion).
    """
    stem = Path(path).stem.lower()
    parts = stem.split("-")
    if len(parts) == 7 and all(p.isdigit() for p in parts):
        return {"01": "neutral", "03": "happy", "04": "sad", "05": "angry",
                "08": "pleasant_surprise"}.get(parts[2])
    last = stem.split("_")[-1]
    return {"angry": "angry", "happy": "happy", "neutral": "neutral", "sad": "sad",
            "ps": "pleasant_surprise", "surprised": "pleasant_surprise"}.get(last)


def load_training_data(*dirs: str | Path) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for d in dirs:
        for p in sorted(Path(d).rglob("*.wav")):
            label = label_from_filename(p)
            if label is not None:
                X.append(extract_features(p))
                y.append(VOICE_EMOTIONS.index(label))
    if not X:
        raise FileNotFoundError("No labelled TESS/RAVDESS .wav files found.")
    return np.stack(X), np.array(y)


def build_classifier(n_features: int = 180, hidden: int = 200, n_layers: int = 3,
                     dropout: float = 0.3, activation: str = "linear"):
    """Fully connected network described in Appendix B.3.

    Three dense layers of 200 units, 30 % dropout, a 5-unit softmax output and
    the Adam optimizer. The appendix specifies linear activations; note that
    stacked linear layers are equivalent to a single linear map, so
    ``activation="relu"`` is offered as an alternative.
    """
    from tensorflow import keras

    model = keras.Sequential([keras.Input(shape=(n_features,))])
    for _ in range(n_layers):
        model.add(keras.layers.Dense(hidden, activation=activation))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(len(VOICE_EMOTIONS), activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_classifier(X: np.ndarray, y: np.ndarray, out_dir: str | Path, epochs: int = 2000,
                     batch_size: int = 64, test_size: float = 0.2, seed: int = 42,
                     activation: str = "linear") -> dict:
    """Train on 80 % of the data, evaluate on 20 %, and save model + scaler + report."""
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X_tr)
    model = build_classifier(n_features=X.shape[1], activation=activation)
    model.fit(scaler.transform(X_tr), y_tr, epochs=epochs, batch_size=batch_size, verbose=0)
    pred = model.predict(scaler.transform(X_te), verbose=0).argmax(axis=1)
    report = classification_report(y_te, pred, labels=range(len(VOICE_EMOTIONS)),
                                   target_names=VOICE_EMOTIONS, output_dict=True, zero_division=0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "voice_classifier.keras")
    (out_dir / "scaler.json").write_text(json.dumps({"mean": scaler.mean_.tolist(),
                                                      "scale": scaler.scale_.tolist()}))
    (out_dir / "test_report.json").write_text(json.dumps(report, indent=2))
    return report


class VoiceClassifier:
    """Load a trained classifier and predict vocal emotion probabilities."""

    def __init__(self, model_dir: str | Path):
        from tensorflow import keras

        model_dir = Path(model_dir)
        self.model = keras.models.load_model(model_dir / "voice_classifier.keras")
        s = json.loads((model_dir / "scaler.json").read_text())
        self.mean, self.scale = np.array(s["mean"]), np.array(s["scale"])

    def predict(self, audio: str | Path) -> dict:
        x = (extract_features(audio) - self.mean) / self.scale
        probs = self.model.predict(x[None, :], verbose=0)[0]
        out = {f"p_{e}": float(p) for e, p in zip(VOICE_EMOTIONS, probs)}
        out["voice_emotion"] = VOICE_EMOTIONS[int(probs.argmax())]
        out["voice_neutral_binary"] = int(out["voice_emotion"] == "neutral")
        return out


# ------------------------------------------------ segment-level export (B.2)
def duration_weighted_mean(segments: pd.DataFrame, start_col: str, end_col: str,
                           value_col: str) -> dict:
    """Duration-weighted average of a segment-level score (pauses already excluded)."""
    df = segments[[start_col, end_col, value_col]].apply(pd.to_numeric, errors="coerce").dropna()
    dur = (df[end_col] - df[start_col]).clip(lower=0)
    if dur.sum() == 0:
        return {"n_segments": len(df), "voiced_seconds": 0.0, "neutral_voice": np.nan}
    return {"n_segments": len(df), "voiced_seconds": float(dur.sum()),
            "neutral_voice": float(np.average(df[value_col], weights=dur))}


def aggregate_segment_export(path: str | Path, start_col: str, end_col: str, neutral_col: str,
                             skiprows: int = 0, sep: str = ",") -> dict:
    """Video-level ``Neutral voice`` from a segment-level export (e.g. iMotions audEERING)."""
    seg = pd.read_csv(path, skiprows=skiprows, sep=sep)
    missing = [c for c in (start_col, end_col, neutral_col) if c not in seg.columns]
    if missing:
        raise KeyError(f"{path}: columns not found {missing}. Available: {list(seg.columns)[:30]}")
    return duration_weighted_mean(seg, start_col, end_col, neutral_col)

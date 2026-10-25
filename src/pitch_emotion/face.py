"""Facial emotion measures.

* Open-source track (Appendix B.3): sample a frame every 0.1 s with OpenCV,
  detect faces with a pre-trained Haar cascade, score emotions with the ``fer``
  library, and compute the share of face frames in which "neutral" is the
  dominant emotion (``Neutral face``).
* Commercial track (Appendix B.2): ``aggregate_frame_export`` applies the same
  video-level rule to frame-level exports from iMotions / Affectiva AFFDEX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

FER_EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


def neutral_dominance(frames: pd.DataFrame, emotion_cols: Sequence[str], neutral_col: str) -> dict:
    """Share of face frames in which ``neutral_col`` has the highest score.

    Frames where every emotion column is missing are treated as frames
    without a detected face and excluded from the denominator.
    """
    scores = frames[list(emotion_cols)].apply(pd.to_numeric, errors="coerce")
    has_face = scores.notna().any(axis=1)
    face_scores = scores[has_face]
    n_face = int(has_face.sum())
    if n_face == 0:
        return {"n_frames": len(frames), "n_face_frames": 0, "face_presence": 0.0, "neutral_face": np.nan}
    dominant = face_scores.fillna(-np.inf).idxmax(axis=1)
    return {
        "n_frames": len(frames),
        "n_face_frames": n_face,
        "face_presence": n_face / len(frames),
        "neutral_face": float((dominant == neutral_col).mean()),
    }


def sample_frames(video: str | Path, interval_s: float = 0.1, save_dir: str | Path | None = None):
    """Yield ``(timestamp_s, frame)`` every ``interval_s`` seconds.

    If ``save_dir`` is given, frames are also written there as JPG files,
    one folder per video, as described in the appendix.
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * interval_s)))
    if save_dir is not None:
        save_dir = Path(save_dir) / Path(video).stem
        save_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t = idx / fps
            if save_dir is not None:
                cv2.imwrite(str(save_dir / f"frame_{idx:06d}.jpg"), frame)
            yield t, frame
        idx += 1
    cap.release()


def analyze_video_fer(video: str | Path, interval_s: float = 0.1,
                      save_dir: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Frame-level FER scores and the video-level ``Neutral face`` measure.

    Face presence is detected with OpenCV's frontal-face Haar cascade; when
    several faces appear, the largest one (the presenter) is scored.
    """
    import cv2
    from fer import FER

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    detector = FER(mtcnn=False)
    rows = []
    for t, frame in sample_frames(video, interval_s, save_dir):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        row = {"time_s": t, "haar_face": len(faces) > 0}
        if len(faces):
            results = detector.detect_emotions(frame)
            if results:
                best = max(results, key=lambda r: r["box"][2] * r["box"][3])
                row.update(best["emotions"])
        rows.append(row)
    frames = pd.DataFrame(rows)
    for e in FER_EMOTIONS:
        if e not in frames:
            frames[e] = np.nan
    summary = neutral_dominance(frames, FER_EMOTIONS, "neutral")
    summary["haar_face_presence"] = float(frames["haar_face"].mean()) if len(frames) else 0.0
    return frames, summary


def aggregate_frame_export(path: str | Path, emotion_cols: Sequence[str], neutral_col: str,
                           skiprows: int = 0, sep: str = ",") -> dict:
    """Video-level ``Neutral face`` from a frame-level export (e.g. iMotions AFFDEX).

    Column names differ between iMotions versions; set them in ``config.yaml``.
    """
    frames = pd.read_csv(path, skiprows=skiprows, sep=sep)
    missing = [c for c in [*emotion_cols, neutral_col] if c not in frames.columns]
    if missing:
        raise KeyError(f"{path}: columns not found {missing}. Available: {list(frames.columns)[:30]}")
    cols = list(dict.fromkeys([*emotion_cols, neutral_col]))
    return neutral_dominance(frames, cols, neutral_col)

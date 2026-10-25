"""Tests for the measure definitions that need no heavy models."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pitch_emotion import face, text_emotion, validation, voice  # noqa: E402


def test_neutral_dominance_excludes_frames_without_face():
    frames = pd.DataFrame({
        "happy":   [0.8, 0.1, np.nan, 0.2],
        "neutral": [0.1, 0.7, np.nan, 0.6],
        "sad":     [0.1, 0.2, np.nan, 0.2],
    })
    s = face.neutral_dominance(frames, ["happy", "neutral", "sad"], "neutral")
    assert s["n_face_frames"] == 3
    assert s["face_presence"] == pytest.approx(0.75)
    assert s["neutral_face"] == pytest.approx(2 / 3)


def test_neutral_dominance_no_face():
    frames = pd.DataFrame({"happy": [np.nan], "neutral": [np.nan]})
    assert np.isnan(face.neutral_dominance(frames, ["happy", "neutral"], "neutral")["neutral_face"])


def test_duration_weighted_mean():
    seg = pd.DataFrame({"start": [0.0, 2.0], "end": [1.0, 5.0], "neutral": [0.2, 0.6]})
    s = voice.duration_weighted_mean(seg, "start", "end", "neutral")
    assert s["voiced_seconds"] == pytest.approx(4.0)
    assert s["neutral_voice"] == pytest.approx((0.2 * 1 + 0.6 * 3) / 4)


@pytest.mark.parametrize("name,label", [
    ("03-01-01-01-01-01-01.wav", "neutral"),
    ("03-01-08-02-02-01-12.wav", "pleasant_surprise"),
    ("03-01-02-01-01-01-01.wav", None),          # calm: excluded
    ("OAF_back_angry.wav", "angry"),
    ("YAF_dog_ps.wav", "pleasant_surprise"),
    ("YAF_dog_fear.wav", None),
])
def test_label_from_filename(name, label):
    assert voice.label_from_filename(name) == label


def test_wildcard_expansion():
    vocab = ["happy", "happiness", "hat", "joy"]
    assert text_emotion.expand_wildcards(["happ*", "joy", "absent"], vocab) == ["happiness", "happy", "joy"]


def test_nrc_scores():
    lex = {e: set() for e in text_emotion.NRC_EMOTIONS}
    lex["joy"] = {"love", "great"}
    lex["fear"] = {"risk"}
    text = "I love this great idea. There is some risk here."
    s = text_emotion.nrc_scores(text, lex)
    assert s["joy"] == pytest.approx((2 / 5 + 0) / 2)
    assert s["fear"] == pytest.approx((0 + 1 / 5) / 2)
    assert s["positive_verbal"] == s["joy"]
    assert s["negative_verbal"] == pytest.approx((s["sadness"] + s["anger"] + s["fear"]) / 3)


def test_embedding_scorer_with_stub_vectors():
    class KV(dict):
        @property
        def index_to_key(self):
            return list(self.keys())

    kv = KV(good=np.array([1.0, 0.0]), bad=np.array([0.0, 1.0]), product=np.array([1.0, 1.0]))
    scorer = text_emotion.EmbeddingEmotionScorer(kv, {"pos": ["good"], "neg": ["bad"]})
    s = scorer.score("good good product")
    assert s["pos"] > s["neg"]


def test_human_agreement():
    out = validation.human_agreement(pd.Series([0.9, 0.2, 0.6, 0.4]), pd.Series([1, 0, 0, 0]))
    assert out == {"n": 4, "matches": 3, "agreement": 0.75}

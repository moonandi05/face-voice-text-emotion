"""Textual emotion measures.

Two approaches from Appendix B:

* ``EmbeddingEmotionScorer`` (B.2): a "geometric" sentiment measure. A Word2Vec
  model is trained on the full transcript corpus; each emotion is represented by
  the average vector of its seed words (e.g. LIWC positive / negative emotion
  lists), each transcript by the average vector of its words, and intensity is
  the cosine similarity between the two (Gennaro & Ash, 2022).
* ``nrc_scores`` (B.3): sentence-level word-share scores from the NRC
  Emotion Lexicon, averaged to the creator level.

The LIWC dictionary is licensed and is NOT distributed with this repository;
supply your own seed-word files (one word per line, LIWC-style trailing ``*``
wildcards are supported).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_RE.split(text.strip()) if s]


def load_wordlist(path: str | Path) -> list[str]:
    """Read a seed-word list: one entry per line, ``#`` comments ignored."""
    words = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.append(line)
    return words


def expand_wildcards(entries: Iterable[str], vocabulary: Iterable[str]) -> list[str]:
    """Expand LIWC-style entries such as ``happ*`` against a vocabulary."""
    vocab = list(vocabulary)
    vocab_set = set(vocab)
    out: set[str] = set()
    for e in entries:
        if e.endswith("*"):
            stem = e[:-1]
            out.update(w for w in vocab if w.startswith(stem))
        elif e in vocab_set:
            out.add(e)
    return sorted(out)


def train_word2vec(transcripts: Sequence[str], vector_size: int = 300, window: int = 5,
                   min_count: int = 2, epochs: int = 20, seed: int = 42, workers: int = 1):
    """Train Word2Vec (Mikolov et al., 2013) on the transcript corpus.

    ``workers=1`` together with ``seed`` makes training reproducible.
    """
    from gensim.models import Word2Vec

    sentences = [tokenize(s) for t in transcripts for s in split_sentences(t)]
    return Word2Vec(sentences=sentences, vector_size=vector_size, window=window,
                    min_count=min_count, epochs=epochs, seed=seed, workers=workers)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else float("nan")


class EmbeddingEmotionScorer:
    """Cosine similarity between a transcript's mean vector and emotion centroids."""

    def __init__(self, keyed_vectors, seed_words: dict[str, Iterable[str]]):
        self.kv = keyed_vectors
        vocab = list(keyed_vectors.index_to_key)
        self.seeds: dict[str, list[str]] = {}
        self.centroids: dict[str, np.ndarray] = {}
        for emotion, entries in seed_words.items():
            words = expand_wildcards(entries, vocab)
            if not words:
                raise ValueError(f"No seed words for '{emotion}' found in the corpus vocabulary.")
            self.seeds[emotion] = words
            self.centroids[emotion] = np.mean([keyed_vectors[w] for w in words], axis=0)

    def transcript_vector(self, text: str) -> np.ndarray | None:
        vecs = [self.kv[w] for w in tokenize(text) if w in self.kv]
        return np.mean(vecs, axis=0) if vecs else None

    def score(self, text: str) -> dict[str, float]:
        v = self.transcript_vector(text)
        if v is None:
            return {e: float("nan") for e in self.centroids}
        return {e: _cosine(v, c) for e, c in self.centroids.items()}


# --------------------------------------------------------------------------- NRC
NRC_EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy",
                "sadness", "surprise", "trust", "positive", "negative")


def load_nrc_lexicon(path: str | Path) -> dict[str, set[str]]:
    """Load ``NRC-Emotion-Lexicon-Wordlevel-v0.92.txt`` (word<TAB>emotion<TAB>0/1)."""
    lex: dict[str, set[str]] = {e: set() for e in NRC_EMOTIONS}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 3 and parts[2] == "1" and parts[1] in lex:
            lex[parts[1]].add(parts[0].lower())
    return lex


def nrc_scores(text: str, lexicon: dict[str, set[str]],
               emotions: Sequence[str] = ("joy", "sadness", "surprise", "anger", "fear")) -> dict[str, float]:
    """Creator-level NRC emotion scores as in Appendix B.3.

    For each sentence, the score of an emotion is the share of its words linked
    to that emotion (0-1); sentence scores are averaged. Also returns
    ``positive_verbal`` (joy / happiness) and ``negative_verbal`` (mean of
    sadness, anger and fear).
    """
    per_sentence = []
    for sent in split_sentences(text):
        toks = tokenize(sent)
        if toks:
            per_sentence.append({e: sum(t in lexicon[e] for t in toks) / len(toks) for e in emotions})
    if not per_sentence:
        out = {e: float("nan") for e in emotions}
    else:
        out = {e: float(np.mean([s[e] for s in per_sentence])) for e in emotions}
    out["positive_verbal"] = out.get("joy", float("nan"))
    neg = [out[e] for e in ("sadness", "anger", "fear") if e in out]
    out["negative_verbal"] = float(np.mean(neg)) if neg else float("nan")
    return out

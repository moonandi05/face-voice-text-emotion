"""Measure validation (Appendix B.4).

* ``tool_correlations``: convergence between two tools on the same videos
  (e.g. fer vs. Affectiva, Praat vs. audEERING).
* ``human_agreement``: dichotomize continuous scores (>= 0.5 = neutral) and
  count matches with human coding.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def tool_correlations(a: pd.DataFrame, b: pd.DataFrame, columns: Sequence[str] | dict[str, str],
                      on: str = "video_id") -> pd.DataFrame:
    """Pearson correlations between matching measures from two tools.

    ``columns`` is either a list of shared column names or a mapping
    ``{column_in_a: column_in_b}``.
    """
    mapping = dict(columns) if isinstance(columns, dict) else {c: c for c in columns}
    merged = a.merge(b, on=on, suffixes=("_a", "_b"))
    rows = []
    for ca, cb in mapping.items():
        xa = merged[f"{ca}_a" if f"{ca}_a" in merged else ca]
        xb = merged[f"{cb}_b" if f"{cb}_b" in merged else cb]
        ok = xa.notna() & xb.notna()
        r, p = stats.pearsonr(xa[ok], xb[ok]) if ok.sum() > 2 else (np.nan, np.nan)
        rows.append({"measure_a": ca, "measure_b": cb, "n": int(ok.sum()), "r": r, "p": p})
    return pd.DataFrame(rows)


def human_agreement(scores: pd.Series, human: pd.Series, threshold: float = 0.5) -> dict:
    """Agreement between dichotomized algorithm scores and human labels (1 = neutral)."""
    df = pd.DataFrame({"score": scores, "human": human}).dropna()
    algo = (df["score"] >= threshold).astype(int)
    matches = int((algo == df["human"].astype(int)).sum())
    return {"n": len(df), "matches": matches, "agreement": matches / len(df) if len(df) else np.nan}

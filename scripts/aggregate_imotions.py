"""Aggregate iMotions exports (Affectiva AFFDEX frames, audEERING voice segments)
into video-level Neutral face and Neutral voice measures (Appendix B.2).

Expects one CSV per video, named <video_id>.csv, in the folders set in config.yaml.

Usage:
    python scripts/aggregate_imotions.py --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pitch_emotion import face, voice  # noqa: E402

log = logging.getLogger("imotions")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    c = yaml.safe_load(Path(args.config).read_text())["imotions"]

    rows: dict[str, dict] = {}
    affdex_dir = Path(c["affdex_dir"])
    for p in sorted(affdex_dir.glob("*.csv")) if affdex_dir.exists() else []:
        try:
            s = face.aggregate_frame_export(p, c["affdex_emotion_cols"], c["affdex_neutral_col"],
                                            skiprows=c.get("affdex_skiprows", 0))
            rows.setdefault(p.stem, {"video_id": p.stem}).update(
                {k if k.startswith(("neutral_", "face_")) else f"face_{k}": v for k, v in s.items()})
        except Exception as exc:
            log.error("%s: %s", p.name, exc)

    aud_dir = Path(c["audeering_dir"])
    for p in sorted(aud_dir.glob("*.csv")) if aud_dir.exists() else []:
        try:
            s = voice.aggregate_segment_export(p, c["audeering_start_col"], c["audeering_end_col"],
                                               c["audeering_neutral_col"], skiprows=c.get("audeering_skiprows", 0))
            rows.setdefault(p.stem, {"video_id": p.stem}).update(
                {k if k.startswith(("neutral_", "voice_")) else f"voice_{k}": v for k, v in s.items()})
        except Exception as exc:
            log.error("%s: %s", p.name, exc)

    if not rows:
        sys.exit("No exports aggregated. Check the folders and column names in config.yaml.")
    out = pd.DataFrame(rows.values())
    Path(c["output"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(c["output"], index=False)
    log.info("Wrote %d videos to %s", len(out), c["output"])


if __name__ == "__main__":
    main()

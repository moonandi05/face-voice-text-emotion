"""Run the open-source face / voice / text pipeline on a folder of pitch videos.

Usage:
    python scripts/run_pipeline.py --config config.yaml
    python scripts/run_pipeline.py --config config.yaml --steps audio text   # subset of steps

Steps (each is resumable; finished outputs in work/ are reused):
    audio  video -> audio, optional Demucs vocal isolation
    asr    transcription (Whisper or Google SpeechRecognition)
    face   frame sampling, Haar face detection, FER emotions -> Neutral face
    voice  Praat pitch (mean / SD of F0) and optional vocal-emotion classifier
    text   Word2Vec emotion similarity and NRC lexicon scores (corpus-level step)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pitch_emotion import audio, face, text_emotion, transcribe, voice  # noqa: E402

ALL_STEPS = ("audio", "asr", "face", "voice", "text")
log = logging.getLogger("pipeline")


def load_sex(path: str | None) -> dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path, dtype={"video_id": str})
    return dict(zip(df["video_id"], df["sex"].str.lower()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=list(ALL_STEPS))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    P, A, F, V, T = cfg["paths"], cfg["asr"], cfg["face"], cfg["voice"], cfg["text"]
    work = Path(P["work"])
    dirs = {k: work / k for k in ("audio", "vocals", "chunks", "transcripts", "face", "frames", "voice")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    videos = sorted(Path(P["videos"]).glob("*.mp4"))
    if not videos:
        sys.exit(f"No .mp4 files found in {P['videos']}")
    sex_map = load_sex(P.get("sex_file"))
    classifier = voice.VoiceClassifier(V["classifier_dir"]) if V.get("classifier_dir") and "voice" in args.steps else None

    for i, video in enumerate(videos, 1):
        vid = video.stem
        log.info("[%d/%d] %s", i, len(videos), vid)
        ext = ".mp3" if A["engine"] == "whisper" else ".wav"
        raw_audio = dirs["audio"] / f"{vid}{ext}"
        speech = raw_audio
        try:
            # ---------------------------------------------------------- audio
            # asr and voice depend on these files, so they are created whenever missing.
            needs_audio = bool({"audio", "asr", "voice"} & set(args.steps))
            if needs_audio and not raw_audio.exists():
                audio.video_to_audio(video, raw_audio)
            if needs_audio and A.get("isolate_vocals"):
                vocals = dirs["vocals"] / "htdemucs" / raw_audio.stem / "vocals.wav"
                if not vocals.exists():
                    vocals = audio.separate_vocals(raw_audio, dirs["vocals"], device=A.get("demucs_device"))
                speech = vocals

            # ------------------------------------------------------------ asr
            tx_path = dirs["transcripts"] / f"{vid}.txt"
            if "asr" in args.steps and not tx_path.exists():
                if A["engine"] == "whisper":
                    text = transcribe.transcribe_whisper(speech, A["whisper_model"], A["language"])
                else:
                    chunks = audio.split_on_silence_chunks(speech, dirs["chunks"] / vid, min_silence_ms=700)
                    text = transcribe.transcribe_google_chunks(chunks, language=A.get("google_language", "en-US"))
                tx_path.write_text(text, encoding="utf-8")

            # ----------------------------------------------------------- face
            face_summary = dirs["face"] / f"{vid}.json"
            if "face" in args.steps and F.get("enabled", True) and not face_summary.exists():
                frames, summary = face.analyze_video_fer(
                    video, F["interval_s"], dirs["frames"] if F.get("save_frames") else None)
                frames.to_csv(dirs["face"] / f"{vid}_frames.csv", index=False)
                face_summary.write_text(json.dumps(summary))

            # ---------------------------------------------------------- voice
            voice_summary = dirs["voice"] / f"{vid}.json"
            if "voice" in args.steps and not voice_summary.exists():
                mono = audio.to_mono(speech, dirs["voice"] / f"{vid}_mono.wav")
                result: dict = {}
                if V.get("praat"):
                    sex = sex_map.get(vid)
                    if sex in voice.PITCH_RANGES:
                        result.update(voice.praat_pitch(mono, sex))
                    else:
                        log.warning("%s: no sex in %s; skipping Praat (pitch range is sex-specific)",
                                    vid, P.get("sex_file"))
                if classifier is not None:
                    result.update(classifier.predict(mono))
                voice_summary.write_text(json.dumps(result))
        except Exception as exc:  # keep going; one bad video should not stop the batch
            log.error("%s failed: %s", vid, exc)

    # -------------------------------------------------------- text (corpus)
    ids = [v.stem for v in videos]
    rows = {vid: {"video_id": vid} for vid in ids}
    transcripts = {vid: (dirs["transcripts"] / f"{vid}.txt").read_text(encoding="utf-8")
                   for vid in ids if (dirs["transcripts"] / f"{vid}.txt").exists()}
    if "text" in args.steps and transcripts:
        if T.get("positive_words") and T.get("negative_words"):
            model = text_emotion.train_word2vec(list(transcripts.values()), **T["word2vec"])
            scorer = text_emotion.EmbeddingEmotionScorer(model.wv, {
                "positive_words": text_emotion.load_wordlist(T["positive_words"]),
                "negative_words": text_emotion.load_wordlist(T["negative_words"]),
            })
            (work / "seed_words_in_vocab.json").write_text(json.dumps(scorer.seeds, indent=1))
            for vid, tx in transcripts.items():
                rows[vid].update(scorer.score(tx))
        if T.get("nrc_lexicon"):
            lex = text_emotion.load_nrc_lexicon(T["nrc_lexicon"])
            for vid, tx in transcripts.items():
                rows[vid].update({f"nrc_{k}": v for k, v in text_emotion.nrc_scores(tx, lex).items()})

    # ---------------------------------------------------------------- merge
    for vid in ids:
        for kind, prefix in (("face", "face_"), ("voice", "voice_")):
            p = dirs[kind] / f"{vid}.json"
            if p.exists():
                for k, v in json.loads(p.read_text()).items():
                    key = k if k.startswith(("neutral_", "face_", "voice_", "p_")) else prefix + k
                    rows[vid][key] = v
        rows[vid]["word_count"] = len(text_emotion.tokenize(transcripts.get(vid, "")))
    out = pd.DataFrame(rows.values())
    if "voice_sd_f0" in out:
        out["neutral_voice_pitch_sd"] = out["voice_sd_f0"]  # lower = more neutral (Appendix B.3)
    Path(P["output"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(P["output"], index=False)
    log.info("Wrote %d rows to %s", len(out), P["output"])


if __name__ == "__main__":
    main()

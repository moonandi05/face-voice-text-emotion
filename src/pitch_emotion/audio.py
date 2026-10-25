"""Audio preparation: video-to-audio conversion, vocal isolation, silence splitting."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def video_to_audio(video: str | Path, out_path: str | Path, sample_rate: int = 44100) -> Path:
    """Extract the audio track of a video with FFmpeg.

    The output format follows the extension of ``out_path``: .mp3 for the
    commercial track (Appendix B.2), .wav for the open-source track (B.3).
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg not found on PATH. Install it first (see README).")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ar", str(sample_rate)]
    if out_path.suffix.lower() == ".mp3":
        cmd += ["-codec:a", "libmp3lame", "-q:a", "2"]
    else:
        cmd += ["-acodec", "pcm_s16le"]
    cmd.append(str(out_path))
    subprocess.run(cmd, check=True)
    return out_path


def separate_vocals(audio: str | Path, out_dir: str | Path, model: str = "htdemucs",
                    device: str | None = None) -> Path:
    """Remove background music and ambient noise with Demucs (two-stem mode).

    Returns the path of the isolated vocal track.
    """
    out_dir = Path(out_dir)
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", model, "-o", str(out_dir)]
    if device:
        cmd += ["-d", device]
    cmd.append(str(audio))
    subprocess.run(cmd, check=True)
    return out_dir / model / Path(audio).stem / "vocals.wav"


def to_mono(audio: str | Path, out_path: str | Path) -> Path:
    """Convert audio to a single-channel WAV with pydub."""
    from pydub import AudioSegment

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    AudioSegment.from_file(audio).set_channels(1).export(out_path, format="wav")
    return out_path


def split_on_silence_chunks(audio: str | Path, out_dir: str | Path, min_silence_ms: int = 700,
                            silence_offset_db: float = -16.0, keep_silence_ms: int = 300) -> list[Path]:
    """Split audio wherever silence lasts ``min_silence_ms`` or longer (700 ms in B.3).

    The silence threshold is set relative to the clip's average loudness.
    """
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    sound = AudioSegment.from_file(audio).set_channels(1)
    chunks = split_on_silence(
        sound,
        min_silence_len=min_silence_ms,
        silence_thresh=sound.dBFS + silence_offset_db,
        keep_silence=keep_silence_ms,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, chunk in enumerate(chunks):
        p = out_dir / f"{Path(audio).stem}_chunk{i:04d}.wav"
        chunk.export(p, format="wav")
        paths.append(p)
    return paths

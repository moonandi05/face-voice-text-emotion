"""Speech-to-text: Whisper (Appendix B.2) or Google SpeechRecognition on chunks (B.3)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

_WHISPER_CACHE: dict = {}


def transcribe_whisper(audio: str | Path, model_name: str = "large-v3", language: str = "en") -> str:
    """Transcribe a (vocal-isolated) audio file with OpenAI Whisper."""
    import whisper

    if model_name not in _WHISPER_CACHE:
        _WHISPER_CACHE[model_name] = whisper.load_model(model_name)
    result = _WHISPER_CACHE[model_name].transcribe(str(audio), language=language)
    return result["text"].strip()


def transcribe_google_chunks(chunk_paths: Iterable[str | Path], language: str = "en-US") -> str:
    """Transcribe silence-split WAV chunks with the free Google Speech Recognition
    service and join them into one creator-level transcript.

    Chunks the service cannot understand are skipped.
    """
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    texts = []
    for path in chunk_paths:
        with sr.AudioFile(str(path)) as source:
            audio = recognizer.record(source)
        try:
            texts.append(recognizer.recognize_google(audio, language=language))
        except sr.UnknownValueError:
            continue
    return " ".join(texts).strip()

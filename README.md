# multimodal-pitch-emotion

Facial, vocal and textual emotion measures for entrepreneurial pitch videos.

This repository implements the video analysis procedure in Appendix B of:

> Mao, J. J., Williams, C., Stroe, S., & Sirén, C. (2026). Poker face and steady voice: Gender and reactions to emotional neutrality in crowdfunding. *Journal of Business Venturing Insights*. https://doi.org/10.1016/j.jbvi.2026.e00600

Each pitch video is decomposed into three channels (face, voice and text), following the approach of Hu and Ma (2025), and summarized as one row of video-level measures. The two main variables in the paper are **Neutral face** (share of face frames in which neutrality is the dominant emotion) and **Neutral voice** (degree of emotional neutrality in the voice).

## Two tracks

| | Open-source track (Appendix B.3) | Commercial track (Appendix B.2) |
|---|---|---|
| Text | FFmpeg → (Demucs) → Whisper or Google SpeechRecognition; NRC lexicon and Word2Vec similarity | FFmpeg → Demucs → Whisper; Word2Vec similarity to LIWC seed words |
| Face | OpenCV frames every 0.1 s, Haar cascade, `fer` | iMotions Affectiva AFFDEX 2.0 |
| Voice | Praat pitch mean / SD; five-class neural network trained on TESS + RAVDESS | iMotions audEERING |
| Script | `scripts/run_pipeline.py` | `scripts/aggregate_imotions.py` (+ text step of `run_pipeline.py`) |

iMotions is commercial software, so the commercial track starts from its CSV exports and applies the paper's aggregation rules. The paper's main analysis uses the commercial track; the open-source track is the earlier pipeline, used here for validation (Appendix B.4) and for anyone without an iMotions licence.

## Repository structure

```
├── config.yaml                     # paths, model choices, iMotions column names
├── src/pitch_emotion/
│   ├── audio.py                    # video→audio, Demucs vocal isolation, silence splitting, mono
│   ├── transcribe.py               # Whisper; Google SpeechRecognition on chunks
│   ├── text_emotion.py             # Word2Vec emotion similarity; NRC lexicon scores
│   ├── face.py                     # frame sampling, Haar + fer, Neutral face; AFFDEX aggregation
│   ├── voice.py                    # Praat pitch; Librosa features + classifier; audEERING aggregation
│   └── validation.py               # tool-to-tool correlations; agreement with human coding
├── scripts/
│   ├── run_pipeline.py             # open-source pipeline (resumable, step by step)
│   ├── aggregate_imotions.py       # iMotions exports → Neutral face / Neutral voice
│   └── train_voice_classifier.py   # train the TESS + RAVDESS vocal emotion classifier
├── tests/test_measures.py          # unit tests for the measure definitions
└── data/README.md                  # what to put where (videos, lexicons, exports)
```

## Installation

Python 3.10 or 3.11 is recommended, because `fer`, TensorFlow and PyTorch (used by Whisper and Demucs) do not all support newer versions yet.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

FFmpeg must be on your PATH (`brew install ffmpeg` on macOS; on Windows, download a build from gyan.dev and add its `bin` folder to PATH). A GPU is strongly recommended for Demucs and Whisper `large-v3`; on Google Colab, select a T4 or better.

If you only need the iMotions aggregation, the core packages (`numpy`, `pandas`, `scipy`, `pyyaml`) are enough.

## Data

See [`data/README.md`](data/README.md). In short, you need the videos (`data/videos/<video_id>.mp4`), a sex file for the Praat pitch ranges (`data/sex.csv`), seed-word lists for the embedding measure, the NRC Emotion Lexicon, and, for the commercial track, iMotions exports. Licensed resources (LIWC, NRC, TESS, RAVDESS, iMotions) and the Kickstarter videos are not redistributed here.

## Usage

**Open-source pipeline**

```bash
python scripts/run_pipeline.py --config config.yaml
```

Run a subset of steps with `--steps audio asr face voice text`. Each step writes to `work/` and skips videos that are already done, so an interrupted run can be restarted. The text step is corpus-level (Word2Vec is trained on all transcripts), so run it after transcription is complete.

**Commercial track (iMotions exports)**

```bash
python scripts/aggregate_imotions.py --config config.yaml
```

Export one CSV per video from iMotions, named `<video_id>.csv`, and set the column names and header rows in the `imotions` section of `config.yaml`. Column names differ between iMotions versions, so check one export before running. To get the textual measures used in the paper, also run `python scripts/run_pipeline.py --steps audio asr text` with `engine: whisper`.

**Vocal emotion classifier (optional)**

```bash
python scripts/train_voice_classifier.py --data data/tess data/ravdess --out models/voice
```

Then set `voice.classifier_dir: models/voice` in `config.yaml`.

## Measures

| Column | Source | Definition |
|---|---|---|
| `neutral_face` | `fer` or AFFDEX | Share of frames with a detected face in which neutrality has the highest score among the seven emotions |
| `face_presence` | `fer` or AFFDEX | Share of sampled frames with a scored face |
| `face_haar_face_presence` | OpenCV | Share of sampled frames in which the Haar cascade detects a face |
| `neutral_voice` | audEERING | Duration-weighted mean of segment-level neutral confidence (pauses excluded by the software) |
| `voice_mean_f0`, `voice_sd_f0` | Praat | Mean F0 and its SD (pitch variation), range 100–600 Hz for women and 75–500 Hz for men |
| `neutral_voice_pitch_sd` | Praat | Open-track proxy for vocal neutrality: pitch variation, where **lower values mean a more neutral voice** |
| `p_<emotion>`, `voice_emotion`, `voice_neutral_binary` | Classifier | Predicted probabilities for angry, happy, neutral, pleasant surprise and sad; argmax label; 1 if neutral |
| `positive_words`, `negative_words` | Word2Vec | Cosine similarity between the transcript's mean word vector and the mean vector of the seed words |
| `nrc_<emotion>` | NRC lexicon | Mean over sentences of the share of words linked to joy, sadness, surprise, anger or fear |
| `nrc_positive_verbal`, `nrc_negative_verbal` | NRC lexicon | Joy; mean of sadness, anger and fear |
| `word_count` | Transcript | Number of word tokens |

## Validation

`pitch_emotion.validation` reproduces the checks in Appendix B.4:

```python
import pandas as pd
from pitch_emotion.validation import tool_correlations, human_agreement

open_track = pd.read_csv("outputs/video_measures.csv", dtype={"video_id": str})
imotions = pd.read_csv("outputs/imotions_measures.csv", dtype={"video_id": str})

# Convergence between fer and Affectiva (paper: r = 0.78 for neutrality, N = 183)
tool_correlations(open_track, imotions, ["neutral_face"])

# Agreement with human coding on a random subset (paper: 23/30 for Affectiva, 25/30 for audEERING)
human = pd.read_csv("data/human_coding.csv", dtype={"video_id": str})   # video_id, face_neutral, voice_neutral (0/1)
m = imotions.merge(human, on="video_id")
human_agreement(m["neutral_face"], m["face_neutral"])     # scores >= 0.5 count as neutral
```

## Notes on implementation choices

A few steps in the appendix are manual or underspecified, and the code makes the following choices explicit:

- **Video to audio.** The appendix uses MoviePy for the open track; this repository uses FFmpeg for both tracks (MoviePy calls FFmpeg internally).
- **Manual steps.** The Microsoft Word dictation double-check of transcripts and the manual trimming of audio to the entrepreneur's voice are not automated. Demucs removes music and noise but not other speakers, so videos should be screened as described in Section 3 of the paper.
- **Dominant facial emotion.** The appendix lists seven emotion categories but refers to "the five categories" when choosing the dominant one; the code takes the maximum over all seven.
- **Vocal classifier.** The appendix calls this model a CNN, but its described architecture is fully connected (three dense layers of 200 units, 30% dropout, five-way softmax, Adam, 2,000 epochs, batch size 64). It is implemented as described, with linear hidden activations by default. Because stacked linear layers reduce to a single linear map, `--activation relu` is available as an alternative. Features are standardized on the training split, which the appendix does not mention. Test accuracy will differ from the reported 85% depending on the exact TESS and RAVDESS files and random split.
- **Word2Vec.** Hyperparameters are not reported in the appendix; the defaults in `config.yaml` (300 dimensions, window 5, 20 epochs, fixed seed, single worker for reproducibility) are reasonable starting points. Seed words missing from the corpus vocabulary are dropped; the words actually used are written to `work/seed_words_in_vocab.json`.
- **Google SpeechRecognition.** The free web API is rate-limited and may change; Whisper is the default.

## Testing

```bash
pip install pytest
pytest tests
```

## Citation

If you use this code, please cite the paper above (see `CITATION.cff`). The three-channel decomposition follows Hu, A., & Ma, S. (2025), *Persuading investors: A video-based study*, Journal of Finance. The vocal classifier follows Gorodnichenko, Y., Pham, T., & Talavera, O. (2023), *The voice of monetary policy*, American Economic Review.

## Acknowledgements

Supported by the Swiss National Science Foundation (grant 10.000.989) and a Bocconi University PhD fellowship. Thanks to the Behavioral Lab at the University of St. Gallen and to Dr. Markus Schönberger at iMotions for assistance with the video analysis.

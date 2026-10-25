# Data

Nothing in this folder except this file is committed (see `.gitignore`).

| Path | Content | Needed for |
|---|---|---|
| `videos/<video_id>.mp4` | Pitch videos; the file name is the video ID used everywhere | `run_pipeline.py` |
| `sex.csv` | Columns `video_id`, `sex` (`woman` / `man`) | Praat pitch ranges |
| `lexicons/positive_seeds.txt`, `lexicons/negative_seeds.txt` | One seed word per line, e.g. the LIWC positive and negative emotion lists; trailing `*` wildcards allowed | Word2Vec emotion measure |
| `lexicons/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` | NRC Emotion Lexicon, from https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm | NRC scores |
| `tess/`, `ravdess/` | Toronto Emotional Speech Set and RAVDESS speech `.wav` files | Training the vocal classifier |
| `imotions/affdex/<video_id>.csv` | Frame-level AFFDEX export from iMotions | `aggregate_imotions.py` |
| `imotions/audeering/<video_id>.csv` | Segment-level audEERING export from iMotions, with segment start, end and neutral confidence | `aggregate_imotions.py` |
| `human_coding.csv` | Columns `video_id`, `face_neutral`, `voice_neutral` (0/1) for the validation subset | Validation |

**Licences.** LIWC is commercial and cannot be shared. The NRC lexicon is free for research use under its own terms. TESS and RAVDESS have their own licences; download them from their official sources. Kickstarter videos are not redistributed.

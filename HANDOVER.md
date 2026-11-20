# Hand-over: continuing OpenDragon on another PC

Written 21 November 2026, when the work moved from the desktop to the laptop.

## Where things stand
- **The plan is agreed:** `PLAN.md`. The user approved it and said "start Stage 0".
- **Stage 0 so far, on the desktop:** created the folder layout and an empty git repo (`main`, no commits yet), and a venv with numpy, vosk, sounddevice and pytest. The venv isn't copied, so recreate it.
- **Stage 0 still to do:**
  1. `opendragon/score.py`: WER, name accuracy and latency stats, with tests.
  2. Sentence lists in `data/prompts/`: ~120 enrolment and ~60 test sentences, in original wording. Use no digits. About 15 test sentences should contain names or jargon (Siobhan "shiv-awn", Ngaio "nye-oh", Kubernetes, ...). No test sentence may also appear in the enrolment list.
  3. A recording tool (tkinter + sounddevice, 16 kHz mono wav). It shows one sentence at a time, and the user presses space to start and stop.
  4. A sanity check: score the 9 usable clips in `voice-test-70` with `vosk-model-small-en-us-0.15` (40 MB, Apache 2.0). Leave out me_01.wav, which is silent. Measure WER, real-time factor and finalisation latency.
  5. Log the results in `STAGES.md`, make the first commit, and check in with the user.

## Earlier results on the same 9 clips, for comparison
Whisper small.en: 15.2% WER. Parakeet 0.6B: 16.3%. Moonshine base: 28.3%.

## Folders in this bundle
Put them under the laptop's `Documents`, at the same paths:
- `OpenDragon\` (this project)
- `OpenDictate\voice-test-70\` (the recordings plus `prompts.json`)
- `OpenDictate\experiments\personal-lm\README.md` (why fusing a language model into Parakeet failed for names)
- `Dragon16-analysis\` (`ARCHITECTURE.md` plus `dragon-schema\`)

## Check the laptop before going further
The plan's figures assumed the desktop:

| Desktop spec | Matters for |
|---|---|
| 16 GB RAM | Stage 1b's graph build is sized for this |
| ~30 GB free disk | the plan needs ~13–18 GB |
| WSL with Ubuntu 20.04 | Kaldi |
| Python 3.13 | the venv |

Re-check all four on the laptop. If it has less RAM or disk, prune the language model harder and update PLAN.md.

Stage 1b compiles Kaldi, which loads every core for 1–2 hours. On a laptop, ask the user when to run it, and cap the build's parallel jobs.

## Ground rules (unchanged)
- Ask before any download over ~100 MB, and before installing WSL or other system-level software, including apt packages.
- Never decrypt, extract, convert or redistribute any of Dragon's files. Only open-source models and published methods.
- Explain each stage in plain English, and check in at the end of each one.

## Prompt to paste into the new session on the laptop
> Read Documents\OpenDragon\HANDOVER.md and PLAN.md, then continue Stage 0 of OpenDragon from where the hand-over says. Recreate the venv first.

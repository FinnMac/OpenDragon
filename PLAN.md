# OpenDragon: build plan

*Draft for agreement, 21 November 2026. Nothing has been downloaded or installed yet.*

OpenDragon will be an open-source speech recogniser built the way Dragon NaturallySpeaking 16 is built. That means a "hybrid" recogniser, a pipeline of separate parts you can inspect and change:

**sound → features → acoustic network → search over a word graph → lattice → language-model rescoring → text**

Its main new contribution will be Dragon-style **continuous learning of your voice**.

It is built only from published research and open-source models (Kaldi and Vosk, both Apache 2.0). It uses none of Dragon's files. The map from Dragon's parts to open-source parts is in `Dragon16-analysis\ARCHITECTURE.md`.

---

## 1. Why a hybrid recogniser, in one paragraph

Parakeet and Whisper are "end-to-end": one big network turns sound straight into spelling. That is accurate, but there is nowhere to tell it that "Siobhan" is pronounced *shiv-awn*. The fusion experiment showed this: no bonus fixes it without breaking other words.

A hybrid recogniser splits the job in two:

- An **acoustic network** only judges *sounds*. Each 10 ms slice of audio gets a score for each of a few thousand sub-sound units (Dragon calls them "pels", Kaldi calls them "senones").
- A separate **search** combines those scores with two things you can edit:
  - a **pronunciation dictionary**, e.g. `Siobhan  SH IH V AO N`
  - a **language model**, which knows which word sequences are likely

Because each part is separate, each can be adapted to you separately. That is how Dragon works.

## 2. What this PC can do, and where each piece runs

Facts I checked today:

- **WSL is already installed.** It has Ubuntu 20.04 (the default), Debian and docker-desktop, all on WSL 2. No new system software is needed to get a Linux environment.
- **Disk:** about **29.8 GB free** on C:. That is enough, but only just: see the disk budget below.
- **Machine:** 16 GB RAM, i7-12700F (12 cores, 20 threads), no GPU. Everything below is CPU-only.
- **Existing software:** OpenDictate's Windows venv has Python 3.13 with vosk, numpy, onnxruntime and sherpa-onnx. It does not have torch or Kaldi.

**Decision: a split setup.**

| Where | What runs there | Why |
|---|---|---|
| **Windows** (new venv in `OpenDragon\.venv`) | Recording test audio; the quick Vosk baseline; scoring and reports | Vosk's Windows wheel works already. Your audio and OpenDictate live here. |
| **WSL Ubuntu 20.04** (Kaldi built from source in the Linux home folder, `~/kaldi`) | Building decoding graphs, real lattices, i-vectors, rescoring, and the acoustic-network adaptation | Kaldi's build and recipe scripts are Linux-first, and building it natively on Windows is painful. CPU-only Kaldi builds cleanly on Ubuntu with OpenBLAS. |

The project folder itself stays on Windows at `C:\Users\finnh\Documents\OpenDragon`, so you can see everything. Linux reaches it at `/mnt/c/Users/finnh/Documents/OpenDragon`. Only Kaldi's compiled code lives inside Linux, because file access across the two systems is slow for compiling.

**Rejected options:**

- **Python-only (no Kaldi).** You can't build a word graph or get lattices without OpenFst and Kaldi's decoders.
- **Native Windows Kaldi.** It's unsupported and fragile.
- **kaldi-active-grammar.** It has Windows wheels and runtime pronunciations, but it is AGPL-licensed and hides the lattices.

## 3. Open-source models to use (sizes checked today)

| Download | Size | What it is | Used in |
|---|---|---|---|
| `vosk-model-en-us-0.22-lgraph` | **131 MB** | A complete Kaldi hybrid with a compact graph. Its acoustic net is a TDNN-F "chain" model with online i-vectors. At this size it is the closest open analogue to Dragon (44 MB acoustic + 210 MB language model). | Stage 1a |
| `vosk-model-en-us-0.22-compile` | **3.0 GB** zipped | The "source code" of the Vosk model: acoustic model, the phonetic decision `tree`, the lexicon, phone set, and the language-model sources. **Required to rebuild the graph with our own dictionary and language model.** | Stages 1b–5 |
| `vosk-model-en-us-0.22` (big) | 1.9 GB | Same acoustics with a huge static graph. **Optional**: an accuracy reference only. I suggest skipping it to save disk. | — |
| PyTorch for CPU | ~200 MB wheel, ~1 GB installed | Training engine for the adaptation stage | Stage 5 |

Vosk's own instructions for rebuilding the graph assume a 32 GB, 100 GB-disk server, because they rebuild the *full-size* language model. We will build a **pruned** language model instead: smaller, Dragon-sized. That fits in 16 GB.

**Memory needs.** Everything here runs on the CPU, and no graphics card is involved. The 32 GB figure is for the one-off *graph build*, not for recognising speech:

- **Recognising speech** with the compact model needs roughly 1–2 GB of RAM.
- **Building the graph** needs whatever the language model's size requires. We choose how hard to prune it so the build fits comfortably in this PC's 16 GB, aiming for 8 GB or less.
- **Adding a word** (Stage 2's grammar-FST approach) is meant to need very little.

The real figures will be measured in Stage 1b. The full-size language model can still be used where it's cheap: rescoring the lattice, i.e. re-ranking the first pass's shortlist of sentences.

**SRILM.** Vosk's scripts use SRILM, which has a restrictive licence. I'll swap it for **KenLM / OpenGrm** (open licences), so the whole project stays redistributable.

**Disk budget (approximate):**

| Item | Size |
|---|---|
| Kaldi + OpenFst build | 5–6 GB |
| Compile package unpacked | 5–8 GB (zip deleted after unpacking) |
| Our graphs | 1–2 GB |
| torch | 1 GB |
| Recordings | <0.5 GB |
| **Total** | **≈13–18 GB of the 29.8 GB free** |

It fits, but it would be worth freeing some space if you can.

## 4. Project layout

```
OpenDragon\
  PLAN.md               this document
  README.md             what it is, how to run it (written as stages land)
  STAGES.md             running log: what each stage did and its numbers
  opendragon\           Python package (Windows and WSL both import it)
    audio.py            wav loading, resampling
    score.py            word error rate, per-name accuracy, latency stats
    vosk_baseline.py    Stage 1a
    kaldi.py            thin wrappers that call Kaldi tools in WSL
    lexicon.py          Stage 2: your pronunciation dictionary + G2P
    lm.py               Stage 3: personal n-gram + interpolation
    ivector.py          Stage 4: persistent speaker i-vector
    adapt\              Stage 5: the continuous adaptation loop
      net.py            Kaldi TDNN-F → PyTorch port (checked against Kaldi)
      targets.py        frame targets from lattices / your corrections
      trainer.py        tethered fine-tuning (L2 + cap + relevance)
  recipes\              shell scripts run inside WSL (graph build, decode)
  data\
    lexicon\            user_lexicon.txt (yours, hand-editable)
    text\               your writing for the language model (git-ignored)
    audio\              enrol\, test\, sessions\ (git-ignored)
  models\               downloaded + built models (git-ignored)
  results\              one JSON + one short markdown table per run
  tests\
```

It will be a git repo, licensed Apache 2.0 to match Kaldi and Vosk. Your audio, your writing and the models are excluded from git, so the code can be published without any personal data.

## 5. How every stage is measured

- **Word error rate (WER):** the share of words the recogniser got wrong (substituted, dropped or inserted) against `prompts.json`. Lower is better.
- **Name accuracy:** the share of "hard" words (Siobhan, Ngaio, Kubernetes, and others you add) recognised exactly. WER hides these, because they are few.
- **Speed:**
  - *real-time factor*: processing time ÷ audio length
  - *finalisation latency*: ms from the end of speech to the final text, median and worst
- **Footprint:** model size on disk and peak memory.
- **No-harm check:** generic sentences must not get worse. This is the trap the personal-LM experiment fell into, where "Kubernetes" was written for "car next to the station".

**The test set is too small today.** `voice-test-70` has 9 usable clips (me_01 is silent and excluded), about 93 words, so one mistake moves WER by about 1 point. And to test learning your voice, the recogniser has to learn from one set of recordings and be scored on a *different* set.

So **Stage 0 asks you to record more.** A small recording tool will show one sentence at a time:

- **Enrolment set:** ~120 sentences (~10 minutes). Used for adaptation, never for scoring.
- **Test set:** ~60 different sentences (~5 minutes), recorded on a **different day**. It includes about 15 sentences with names and jargon.
- The existing 9 clips stay as a third check, so results can be compared with the earlier ones: Whisper small 15.2%, Parakeet 16.3%.

Every run writes a row to `results\`, so each stage shows a before-and-after table.

## 6. The stages

Each stage ends with a check-in: I show you the numbers and what changed, and you decide whether to go on.

### Stage 0: set-up and test audio (no big downloads)
- Create the folder, git repo, venv and scoring code.
- Build the recording tool, then you record the enrolment and test sets.
- Score the existing 9 clips with the small Vosk model OpenDictate already has, as a sanity check.

### Stage 1: a working hybrid baseline
- **1a. Vosk compact model on Windows (131 MB download).**
  - Decode all the test sets.
  - Get the **N-best list** (the recogniser's top alternative sentences) and per-word confidence.
  - Record WER and latency.
- **1b. Kaldi in WSL plus the compile package (3.0 GB download, plus Ubuntu build packages, about 0.5 GB).**
  - Compile Kaldi: roughly 1–2 hours, running in the background.
  - Build our *own* decoding graph ("HCLG": the sound units, pronunciation dictionary and language model compiled into one searchable network) from the Vosk sources, with a pruned language model.
  - Decode with Kaldi's online decoder and **write out real lattices**. A lattice is the compact graph of every word sequence the search seriously considered.
  - Check that WER is close to 1a.

  This graph is the foundation for everything after.

### Stage 2: a pronunciation dictionary you control
- `data\lexicon\user_lexicon.txt`, in plain text, e.g. `Siobhan SH IH V AO N` and `Ngaio N AY OW`.
- A "sounds like" helper: type *shiv-awn* and get phonemes back, using a small G2P model (grapheme-to-phoneme, i.e. spelling to sounds; Phonetisaurus, a few MB). You can then correct the result.
- The graph is rebuilt in minutes.
- Then try Kaldi's **grammar-FST** feature, which plugs a small word list into the big graph without a full rebuild. This mirrors Dragon's "Embedded" search variant, and it is how adding a word could become instant.
- **Success:** Siobhan and Ngaio are recognised, and the no-harm check holds.

### Stage 3: a personal language model from your writing
- **Your writing.** You choose the source: documents, sent mail, or OpenDictate's history.
- **Personal model.** Train an n-gram model on it with KenLM. An n-gram model gives the probability of each word given the previous 2–3 words.
- **Blend.** Mix it with the general model. The mixing weight is chosen by *perplexity*, a measure of how surprised a model is by held-out text: lower means a better fit.
- **Rebuild the graph** with the blended model.
- **Neural rescoring.** Re-rank the lattices with Vosk's neural language model (RNNLM), which matches Dragon's `NeuralNetLanguageModel`.
- **Difference from the failed Parakeet experiment:** there, the bonus was pushed into a spelling-by-spelling search. Here it is a proper probability inside the word graph, and you can set its weight.

### Stage 4: i-vector speaker adaptation
- **What an i-vector is:** a 100-number summary of your voice and microphone, fed into the acoustic network beside the sound. Dragon calls it the `IdentityVector`.
- **The problem:** the Vosk model already computes one, but it starts from zero each session.
- **The change:** a **persistent per-user i-vector**, estimated from your enrolment audio and updated slowly with MAP relevance smoothing.
  - MAP relevance smoothing: with little new data stay near the old estimate; with more data, trust the new one.
  - This matches Dragon's `InitialIdentityVector`, `AdaptationIdentityVectorRelevance` and `SDSigProc_TransferIdentityVectors`.
- **Measure:** cold start vs warm start on the test set.
- **What we skip:** fMLLR/CMLLR (Dragon's constrained feature transforms). It is designed for GMM models and is mostly replaced by i-vectors in chain models, so it's left as an optional experiment.

### Stage 5: continuous on-device adaptation of the acoustic network (the new part)

Modelled on Dragon's `AdaptSDNeuralNetScorer`.

**5a. Port the network.**
- Convert Vosk's acoustic network (TDNN-F, about 20 M weights) into PyTorch, by parsing Kaldi's text model format.
- **Check:** it must produce the same output as Kaldi to within rounding. If it doesn't, we stop.
- Decode by feeding PyTorch's scores into Kaldi's `latgen-faster-mapped`, so the search, lattices and graph stay pure Kaldi.

**5b. Targets: what the network should have said.**
- **Unsupervised:** take the recogniser's own best path through the lattice and convert it to per-frame sound-unit labels. Ignore words with low confidence (Dragon: `ErrorZeroingType`) and long silences (`InitialAndFinalSilenceMaxFrames`).
- **Supervised:** when you correct a word, force-align your corrected text to get clean labels. Corrections weigh more.

**5c. Tethered fine-tuning.** Every so many frames of new speech (Dragon: `CommitIntervalFrames`), run a few passes of training on recent audio with these safeguards:

- **Which weights move.** Mainly a per-neuron scale and offset in each layer, a published method called LHUC. Dragon's `RecomputeScaleAndOffset` parameters suggest a similar idea. Optionally, the last layer or two as well.
- **L2 pull to the original.** A penalty grows with distance from the speaker-independent weights (`SpeakerIndependentModelRelevance`). In the research literature this is "L2-SP".
- **KL tether.** The adapted network's outputs are also kept close to the original network's outputs. This is the KL-divergence regularised adaptation of Yu et al., 2013.
- **Per-weight movement cap.** No weight may move more than `min + range × |original|` away from its original value (the `WeightsRampFactorMin/Range` and `BiasesRampFactorMin/Range` parameters). Also a per-minibatch step cap (`MinibatchMaxWeightAdjustment`).
- **MAP relevance.** How far an update may go grows with the amount of your speech heard so far: `n / (n + τ)`.
- **Safety net.** The original model is never overwritten. Each update is kept only if it doesn't worsen recognition on a small "canary" set of your earlier, confirmed sentences. Otherwise it is rolled back.

**Hyper-parameters.** Every hyper-parameter is named after its Dragon counterpart in a config file, so you can see the correspondence.

**Measure:** a learning curve. Test-set WER after 0, 1, 2, 5 and 10 minutes of enrolment speech, separately for unsupervised and correction-based learning, plus a drift check (does it get worse over many sessions?).

### Later, not in this plan unless you want them
- A live dictation front end, e.g. as an OpenDictate engine.
- Dragon's other unusual parts: the tree pre-filter, explicit duration modelling, and a small sigmoid, integer-quantised "Dragon-shaped" network distilled from the Vosk one.

## 7. Things to agree before I start

1. **The split setup:** Windows for recording and scoring, the already-installed WSL Ubuntu 20.04 for Kaldi. Stage 1b will `apt install` Ubuntu build packages (about 0.5 GB): build-essential, automake, libtool, sox, zlib, OpenBLAS, python3-venv. I'll ask again just before.
2. **Downloads:**
   - Stage 1a: lgraph model, 131 MB
   - Stage 1b: compile package, 3.0 GB
   - Stage 5: PyTorch CPU, about 200 MB
   - Skip the 1.9 GB big model.

   I'll still ask at each one.
3. **Recording time:** about 15 minutes over two sittings for the enrolment and test sets.
4. **Your writing for Stage 3:** which folder or source? This can wait until Stage 3.
5. **Git and licence:** Apache 2.0, with audio, text and models never committed. Local only, no GitHub, unless you say otherwise.

## 8. Ground rules
- Dragon's model and settings files are never decrypted, extracted, converted, copied or redistributed. Only parameter *names* from the earlier analysis are used, as a reading list of published techniques.
- Everything runs offline, on this PC. Your audio and text stay in `OpenDragon\data`.

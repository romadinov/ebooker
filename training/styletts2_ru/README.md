# Russian StyleTTS2 fine-tune

**Status: abandoned (2026-10).** It never sounded native. The English-pretrained
text aligner credits trailing silence to the last vowel, which stretches word
endings («случая» came out as «случии») and slows speech by 13–33%. Fixing that means
retraining the aligner on Russian. The stress-tuned Qwen (`--backend
qwen-stress`) solved the stress problem without it. Kept for the record.

Why this exists: every neural front-end tested treats a stress mark as a
suggestion (DESIGN.md §6h). StyleTTS2 is phoneme-driven, so stress is decided
before the network runs, by this front-end:

    text → silero (ё restored, stress) → sparse acute → piper's espeak-ng → IPA

Only piper's espeak bridge honours the acute (`за́мок → zˈɑmʌk`, `замо́к → zamˈok`);
system espeak-ng 1.51/1.52 via `phonemizer` ignores it.

## Setup (Spark, `~/st2ru`)

* StyleTTS2 at `5cedc71`, with `styletts2.patch` applied
* PL-BERT: `papercup-ai/multilingual-pl-bert` in `Utils/PLBERT_ml` (Russian included)
* start from `yl4579/StyleTTS2-LibriTTS` `epochs_2nd_00020.pth`
* venv `~/st2ru/.venv`: torch 2.13.0+cu130, plus `piper-tts`, `silero-stress`

## The patch, and why each line is there

* **`ignore_modules=['bert']` when loading LibriTTS.** The checkpoint carries the
  English PL-BERT under `bert`; the same shapes plus `strict=False` meant it
  silently replaced the multilingual one.
* **`weights_only=False`** for the three official checkpoints, which predate the
  torch 2.6 default.
* **TF32 and `ST2_MEM_FRACTION`.** The Spark is shared. The training loop is
  plain fp32, and at batch 8 it holds 72 GB of live tensors.

## Data

`prep.py` writes 24 kHz wavs and `wav|phonemes|0` lists. The one rewrite is
`rɪ^ → rʲ`: `^` is espeak's soft-р spelling and is not in the 178-symbol table.
`u"` is a genuine espeak phoneme; it is kept, and it appears in the PL-BERT's own
training data too.

| corpus | hours | notes |
|---|---|---|
| RUSLAN | 30.5 | one male reader, studio 44.1 kHz, CC BY-NC-SA |
| CSS10-ru | 21.4 | one male reader, LibriVox 22 kHz |

## Cost on the Spark (GB10)

GPU-bound at 94% utilisation. At batch 6: 4.1 s/step and 3,639 steps per epoch,
so about 4 h per epoch before joint training starts; the SLM-adversarial stage
adds about 35%. Inference runs at RTF 0.03.

## Commands

    ~/st2ru/start.sh ruslan ~/st2ru/run_ruslan.sh   # detached; ruslan.pid / ruslan.log
    ~/st2ru/alive.sh ruslan                         # exit 0 while running
    cd ~/st2ru/StyleTTS2 && PYTHONPATH=~/ebooker/src ../.venv/bin/python \
        ~/ebooker/training/styletts2_ru/infer.py Configs/config_ru.yml \
        Models/ru_ruslan/epoch_2nd_0000N.pth ../data/ruslan24/wavs/ruslan_00040.wav ../out_eN

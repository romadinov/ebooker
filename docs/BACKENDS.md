# Backend environments

Some backends pin conflicting versions of torch, so they cannot share one
environment. Silero, Kokoro and Whisper coexist in the main venv; the others
need their own.

## Main environment

```bash
uv sync
```

Covers ingest, normalisation, mastering, packaging, Silero, Kokoro and
verification.

Kokoro additionally needs espeak-ng for out-of-vocabulary words. Its bundled
`espeakng-loader` ships a hard-coded path that does not exist, so set the data
path yourself:

```bash
brew install espeak-ng                      # or: apt install espeak-ng
export ESPEAK_DATA_PATH=/opt/homebrew/opt/espeak-ng/share/espeak-ng-data
uv pip install "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```

Silero's model is a separate download:

```bash
mkdir -p models && curl -L -o models/v5_5_ru.pt \
  https://models.silero.ai/models/tts/ru/v5_5_ru.pt
```

## ESpeech (Russian, voice cloning)

Needs its own venv because `f5-tts` pins torch.

```bash
uv venv esp --python 3.12
VIRTUAL_ENV=esp uv pip install f5-tts ruaccent soundfile num2words lxml "setuptools<81"
```

`torchaudio` 2.11 routes `load()` through torchcodec, whose bundled library will
not link against FFmpeg 9. The backend shims audio I/O to soundfile, so no
action is needed — but that is why the shim exists.

## Qwen stress (Russian, voice cloning, `--backend qwen-stress`)

The stress-tuned Qwen3-TTS fine-tune
[`siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf`](https://huggingface.co/siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf)
(Apache-2.0). It follows a combining acute after the stressed vowel, so the
backend restores ё and adds sparse stress marks with silero. `qwen-tts` pins
transformers 4.57.3, which the main environment does not use, so it needs its
own venv:

```bash
uv venv qwen --python 3.12
VIRTUAL_ENV=qwen uv pip install "qwen-tts==0.1.1" "transformers==4.57.3" accelerate \
  silero-stress lxml num2words ruaccent soundfile soxr openai-whisper
```

`--voice` is a short reference clip; its exact transcript goes next to it as
`<clip>.txt`. A transcript that does not match the audio makes the model babble.
It runs at about 0.75× real time on a DGX Spark without flash-attn.

## Shtorm-PocketTTS-RU (Russian, voice cloning, `--backend pocket`)

[`ArtShtorm/Shtorm-PocketTTS-RU`](https://huggingface.co/ArtShtorm/Shtorm-PocketTTS-RU)
(CC-BY-4.0) is a Russian book-reading fine-tune of Kyutai's
[pocket-tts](https://github.com/kyutai-labs/pocket-tts) (MIT). It follows
combining-acute stress marks, so the backend gives it the qwen-stress text
treatment (ё, sparse silero marks, mid-sentence dashes as «...»). `pocket-tts`
only needs `torch>=2.5`, so it lives in the main environment:

```bash
uv pip install "pocket-tts[audio]"
```

`--voice` is a reference clip, ideally about 5 s ending on a complete phrase; no
transcript is needed. `--variant slow` (default) is the author's calmer delivery,
`fast` the other checkpoint. The model is 100M parameters and renders about five
times faster than real time on CPU; on a DGX Spark it runs on the GPU with the
Mimi codec on CPU, as the library arranges.

`--variant english` is Kyutai's own English model (`TTSModel.load_model(language=
"english")`), which the pipeline loads as a second voice for the English
sentences of a Russian book (see "Latin script in a Russian book" in the README).
Its voice-cloning weights are gated on Hugging Face: without access the library
falls back to `kyutai/pocket-tts-without-voice-cloning`, whose preset voices
(`george`, `michael`, `peter_yearsley`, …) can be named instead of a clip. The
Russian fine-tune cannot read Latin script at all -- it skips the words -- which
is why the routing exists. The `cmudict` package supplies the pronunciations for
transliterating Latin words that stay in Russian sentences.

## Chatterbox

```bash
uv venv cbx --python 3.12
VIRTUAL_ENV=cbx uv pip install chatterbox-tts "setuptools<81" resampy
```

`setuptools<81` is required: the PerTh watermarker imports `pkg_resources`,
removed in later setuptools, and its `__init__` swallows the ImportError so the
failure surfaces as `TypeError: 'NoneType' object is not callable` at model load.

## NVIDIA / aarch64 (DGX Spark)

Install torch from the CUDA 13 index — the default wheels are built against
CUDA 12 and fail with `libcudart.so.12: cannot open shared object file`:

```bash
uv venv .venv --python 3.12
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install f5-tts ruaccent soundfile num2words lxml "setuptools<81" numpy
uv pip install openai-whisper
# f5-tts and openai-whisper pull a generic torch; put the CUDA build back
uv pip install --reinstall-package torch --reinstall-package torchaudio \
  torch torchaudio --index-url https://download.pytorch.org/whl/cu130
```

`torch.cuda.get_arch_list()` stops at `sm_120` while GB10 reports `sm_121`.
That is fine — the two are binary compatible.

**Use openai-whisper, not faster-whisper, for verification here.** The aarch64
CTranslate2 wheel has no CUDA support and its CPU path is unusable: Whisper
large-v3 measured RTF 4.38, making verification the bottleneck. Through torch on
CUDA, `turbo` runs at RTF 0.061. `Transcriber` picks the right one automatically.

Do **not** substitute `distil-large-v3`: it is English-only and silently returns
an English translation of Russian audio, which looks like a large speed win and
quietly breaks verification, since every transcript then mismatches its source.

If your account lacks sudo, a static ffmpeg build works:

```bash
curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz \
  | tar -xJ -C /tmp && cp /tmp/ffmpeg-*-arm64-static/{ffmpeg,ffprobe} ~/.local/bin/
```

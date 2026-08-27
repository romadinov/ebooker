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

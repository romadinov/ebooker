"""Synthesise test sentences from a Russian StyleTTS2 checkpoint.

Run from the StyleTTS2 checkout (it imports the repo's own modules):
    infer.py <config.yml> <checkpoint.pth> <reference.wav> <out_dir> [sentences.txt]

Uses prep.phonemise, i.e. exactly the training front-end. Adapted from the
repo's Demo/Inference_LibriTTS.ipynb.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

sys.path[:0] = [str(Path(__file__).parent), str(Path.cwd())]  # prep; the StyleTTS2 checkout
from front import phonemise  # noqa: E402

from models import build_model, load_ASR_models, load_F0_models  # noqa: E402
from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule  # noqa: E402
from text_utils import TextCleaner  # noqa: E402
from utils import recursive_munch  # noqa: E402
from Utils.PLBERT_ml.util import load_plbert  # noqa: E402

SR = 24000
DEFAULT_SENTENCES = [
    # stress minimal pairs, in context
    "Старый за́мок стоял на холме, а на двери висел тяжёлый замо́к.",
    "Му́ка — это страдание, а мука́ — это мелко смолотое зерно.",
    # the words reported wrong in earlier engines
    "После этого он встал на ноги и пошёл в чёрный отсек.",
    # questions and exclamations: the intonation complaint
    "Ах, так ты Весёлый Пёс? Подлый грабитель? Да на тебе преступлений — что блох на собаке! Вставай, пойдёшь с нами!",
    "Капитан знает, что делает. Я его спрошу. Но вот ещё что! Вы никогда не были в космосе, однако среди вас ходят сотни сплетен. Забудьте их, слышите?!",
    # plain narration
    "Утро выдалось серым и тихим. Над рекой висел туман, и лодки у причала казались вырезанными из бумаги.",
]

to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    return torch.gt(mask + 1, lengths.unsqueeze(1))


def load(config_path: str, ckpt: str, device: str):
    config = yaml.safe_load(open(config_path))
    text_aligner = load_ASR_models(config["ASR_path"], config["ASR_config"])
    pitch_extractor = load_F0_models(config["F0_path"])
    plbert = load_plbert(config["PLBERT_dir"])
    params = recursive_munch(config["model_params"])
    model = build_model(params, text_aligner, pitch_extractor, plbert)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["net"]
    for key in model:
        if key in state:
            sd = state[key]
            if next(iter(sd)).startswith("module."):
                sd = {k[7:]: v for k, v in sd.items()}
            model[key].load_state_dict(sd, strict=False)
        model[key].eval().to(device)
    sampler = DiffusionSampler(model.diffusion.diffusion, sampler=ADPM2Sampler(),
                               sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
                               clamp=False)
    return model, params, sampler


def style_of(model, path: str, device: str):
    wave, _ = librosa.load(path, sr=SR)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    mel = (torch.log(1e-5 + to_mel(torch.from_numpy(audio).float()).unsqueeze(0)) + 4) / 4
    mel = mel.to(device)
    with torch.no_grad():
        return torch.cat([model.style_encoder(mel.unsqueeze(1)),
                          model.predictor_encoder(mel.unsqueeze(1))], dim=1)


@torch.no_grad()
def synth(model, params, sampler, cleaner, ps: str, ref_s, device,
          alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1.0):
    # Pad at BOTH ends, as meldataset.py does in training. Upstream's demo
    # notebook pads only the start, which leaves the last real phoneme where the
    # model has never seen one; it swallowed the final ə ("случая" -> "случий").
    tokens = torch.LongTensor([0] + cleaner(ps) + [0]).to(device).unsqueeze(0)
    lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
    mask = length_to_mask(lengths).to(device)

    t_en = model.text_encoder(tokens, lengths, mask)
    bert_dur = model.bert(tokens, attention_mask=(~mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device), embedding=bert_dur,
                     embedding_scale=embedding_scale, features=ref_s,
                     num_steps=diffusion_steps).squeeze(1)
    ref = alpha * s_pred[:, :128] + (1 - alpha) * ref_s[:, :128]
    s = beta * s_pred[:, 128:] + (1 - beta) * ref_s[:, 128:]

    d = model.predictor.text_encoder(d_en, s, lengths, mask)
    x, _ = model.predictor.lstm(d)
    dur = torch.round(torch.sigmoid(model.predictor.duration_proj(x)).sum(-1).squeeze()).clamp(min=1)
    aln = torch.zeros(lengths, int(dur.sum()))
    c = 0
    for i in range(aln.size(0)):
        aln[i, c:c + int(dur[i])] = 1
        c += int(dur[i])
    aln = aln.unsqueeze(0).to(device)

    def shift(t):
        if params.decoder.type != "hifigan":
            return t
        out = torch.zeros_like(t)
        out[:, :, 0] = t[:, :, 0]
        out[:, :, 1:] = t[:, :, :-1]
        return out

    en = shift(d.transpose(-1, -2) @ aln)
    f0, n = model.predictor.F0Ntrain(en, s)
    out = model.decoder(shift(t_en @ aln), f0, n, ref.squeeze().unsqueeze(0))
    return out.squeeze().cpu().numpy()[..., :-50]  # upstream: pulse at the very end


def main() -> int:
    config, ckpt, ref, out = sys.argv[1:5]
    sentences = (Path(sys.argv[5]).read_text(encoding="utf-8").splitlines()
                 if len(sys.argv) > 5 else DEFAULT_SENTENCES)
    import os
    # alpha/beta = how much of the diffusion-sampled style to use (0 = reference only).
    # Until the diffusion stage has trained, the sampler is LibriTTS's, conditioned
    # on English PL-BERT features, so only 0/0 is meaningful.
    alpha, beta = float(os.environ.get("ST2_ALPHA", 0.3)), float(os.environ.get("ST2_BETA", 0.7))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model, params, sampler = load(config, ckpt, device)
    cleaner = TextCleaner()
    ref_s = style_of(model, ref, device)
    Path(out).mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(s for s in sentences if s.strip()):
        ps = phonemise(text)
        t = time.time()
        wav = synth(model, params, sampler, cleaner, ps, ref_s, device, alpha=alpha, beta=beta)
        sf.write(f"{out}/{i:02d}.wav", wav, SR)
        print(f"{i:02d} {len(wav) / SR:5.1f}s RTF {(time.time() - t) / (len(wav) / SR):.3f}  {ps}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

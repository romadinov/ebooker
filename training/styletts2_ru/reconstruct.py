"""Teacher-forced reconstruction: separates *what* the model says from *when*.

Given a real recording and its text, the aligner's own alignment, the real F0
and the real energy go to the decoder, exactly as in training
(train_finetune.py, "y_rec_gt_pred"). Nothing is predicted but the sound. Then:

* the reconstruction mispronounces a phoneme -> the acoustic model (or the
  alignment it learned from) is wrong for that phoneme;
* the reconstruction is right but free synthesis is wrong -> the durations or
  prosody predicted at inference are to blame.

Also prints the aligner's frames per phoneme, which is the alignment check.

Run from the StyleTTS2 checkout:
    reconstruct.py <config> <ckpt> <out_dir> <wav|text> [<wav|text> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path[:0] = [str(Path(__file__).parent), str(Path.cwd())]
from front import phonemise  # noqa: E402
from infer import length_to_mask, load, to_mel  # noqa: E402
from monotonic_align import mask_from_lens  # noqa: E402
from text_utils import TextCleaner  # noqa: E402
from utils import log_norm, maximum_path  # noqa: E402

SR = 24000


@torch.no_grad()
def reconstruct(model, cleaner, wav_path: str, text: str, device: str):
    wave, sr = sf.read(wav_path, dtype="float32")
    assert sr == SR, sr
    wave = np.concatenate([np.zeros(5000, np.float32), wave, np.zeros(5000, np.float32)])  # as meldataset
    mel = (torch.log(1e-5 + to_mel(torch.from_numpy(wave))) + 4) / 4          # (80, T)
    mel = mel[None, :, : mel.shape[-1] // 2 * 2].to(device)  # even: the aligner halves it
    T = mel.shape[-1]

    ps = phonemise(text)
    ids = [0] + cleaner(ps) + [0]
    texts = torch.LongTensor(ids)[None].to(device)
    in_len = torch.LongTensor([len(ids)]).to(device)
    n_down = model.text_aligner.n_down
    mel_len = torch.LongTensor([T]).to(device)

    mask = length_to_mask(mel_len // (2 ** n_down)).to(device)
    _, _, attn = model.text_aligner(mel, mask, texts)
    attn = attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
    mono = maximum_path(attn, mask_from_lens(attn, in_len, mel_len // (2 ** n_down)))

    t_en = model.text_encoder(texts, in_len, length_to_mask(in_len).to(device))
    asr = t_en @ mono
    L = min(asr.shape[-1], T // 2)
    asr, gt = asr[..., :L], mel[..., :2 * L]
    s = model.style_encoder(gt.unsqueeze(1))
    f0, _, _ = model.pitch_extractor(gt.unsqueeze(1))
    f0 = f0.reshape(1, -1)  # JDC squeezes the batch axis away when it is 1
    n = log_norm(gt.unsqueeze(1)).squeeze(1)
    y = model.decoder(asr, f0, n, s).squeeze().cpu().numpy()

    frames = mono[0].sum(-1).cpu().numpy().astype(int)  # per token, 25 ms each (2 mel hops)
    symbols = ["<pad>"] + list(ps) + ["<pad>"]
    return y, ps, list(zip(symbols, frames))


def main() -> int:
    config, ckpt, out = sys.argv[1:4]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = load(config, ckpt, device)
    cleaner = TextCleaner()
    Path(out).mkdir(parents=True, exist_ok=True)
    for i, pair in enumerate(sys.argv[4:]):
        wav, text = pair.split("|", 1)
        y, ps, frames = reconstruct(model, cleaner, wav, text, device)
        sf.write(f"{out}/{i:02d}_recon.wav", y, SR)
        print(f"{i:02d} {ps}")
        print("    start " + " ".join(f"{s}:{f}" for s, f in frames[:8]))
        print("    end   " + " ".join(f"{s}:{f}" for s, f in frames[-8:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Derive a Russian fine-tune config from StyleTTS2's Configs/config_ft.yml.

Run from the StyleTTS2 checkout:
    make_config.py <corpus_dir> <log_dir> <epochs> [batch] [diff_epoch] [joint_epoch]
        > Configs/config_ru.yml

Only the lines that differ from upstream are touched, by exact match, so an
upstream change to any of them fails loudly instead of being silently kept.
"""
import sys

corpus, log_dir, epochs = sys.argv[1], sys.argv[2], sys.argv[3]
batch, diff_epoch, joint_epoch = (sys.argv[4:] + ["8", "10", "30"][len(sys.argv) - 4:])[:3]
s = open("Configs/config_ft.yml", encoding="utf-8").read()
rep = {
    'log_dir: "Models/LJSpeech"': f'log_dir: "{log_dir}"',
    'pretrained_model: "Models/LibriTTS/epochs_2nd_00020.pth"':
        'pretrained_model: "../libritts/Models/LibriTTS/epochs_2nd_00020.pth"',
    "PLBERT_dir: 'Utils/PLBERT/'": "PLBERT_dir: 'Utils/PLBERT_ml/'",
    'train_data: "Data/train_list.txt"': f'train_data: "{corpus}/train_list.txt"',
    'val_data: "Data/val_list.txt"': f'val_data: "{corpus}/val_list.txt"',
    'root_path: "/local/LJSpeech-1.1/wavs"': f'root_path: "{corpus}/wavs"',
    'OOD_data: "Data/OOD_texts.txt"': f'OOD_data: "{corpus}/ood_texts.txt"',
    'epochs: 50 # number of finetuning epoch (1 hour of data)': f"epochs: {epochs}",
    "batch_size: 8": f"batch_size: {batch}",
    "save_freq: 5": "save_freq: 1",
    "diff_epoch: 10": f"diff_epoch: {diff_epoch}",
    "joint_epoch: 30": f"joint_epoch: {joint_epoch}",
}
for a, b in rep.items():
    if a not in s:
        sys.exit(f"upstream config changed: {a!r} not found")
    s = s.replace(a, b)
sys.stdout.write(s)

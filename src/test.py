import os
from pathlib import Path

import librosa

from dlbd.data import spectrogram
from dlbd.models.CityNetTF2 import CityNetTF2
from dlbd.utils import file as file_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_opts = file_utils.load_config(
    Path("results/models") / "CityNetTF2" / "1" / "network_opts.yaml"
)

model = CityNetTF2(model_opts, version="1")
spec_opts = {"type": "mel", "n_mels": 32, "n_fft": 2048, "hop_length": 1024}

wav, sample_rate = librosa.load(
    "/mnt/win/UMoncton/Doctorat/data/acoustic/reference/Arctic Tern.wav",
    spec_opts.get("sample_rate", None),
)


spec, opts = spectrogram.generate_spectrogram(wav, sample_rate, spec_opts)
print(len(wav) / sample_rate)
print(spec.shape)
print("n_iters:", spec.shape[1] / 20)

model.classify_spectrogram(spec)


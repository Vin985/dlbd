import librosa
import numpy as np

from PIL import Image

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 1024  # 512
DEFAULT_N_MELS = 32  # 128


def get_spec_subfolder(spec_opts):
    spec_folder = "_".join(
        [
            str(spec_opts.get("sample_rate", "original")),
            spec_opts["type"],
            str(spec_opts["n_mels"]),
            str(spec_opts["n_fft"]),
            str(spec_opts["hop_length"]),
        ]
    )
    return spec_folder


def generate_spectrogram(wav, sample_rate, spec_opts):
    opts = {
        "sr": sample_rate,
        "n_fft": spec_opts.get("n_fft", DEFAULT_N_FFT),
        "hop_length": spec_opts.get("hop_length", DEFAULT_HOP_LENGTH),
    }
    if spec_opts["type"] == "mel":
        opts["n_mels"] = spec_opts.get("n_mels", DEFAULT_N_MELS)
        spec = librosa.feature.melspectrogram(wav, **opts)
        spec = spec.astype(np.float32)
    else:
        raise AttributeError("No other spectrogram supported yet")
    return spec, opts


def resize_spectrogram(spec, size):
    img = Image.fromarray(spec)
    img = img.resize(size)
    return np.array(img)

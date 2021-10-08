import librosa
import numpy as np

import mouffet.utils.common as common_utils

from PIL import Image
from numpy.lib.function_base import copy

DEFAULT_OPTS = {
    "sample_rate": "original",
    "n_fft": 2048,
    "hop_length": None,
    "n_mels": 32,
    "window": "hann",
    "win_length": None,
    "pcen": {
        "gain": 0.8,
        "bias": 10,
        "power": 0.25,
        "time_constant": 0.06,
        "eps": 1e-06,
    },
    "type": "mel",
}


def generate_spectrogram(wav, sample_rate, spec_opts):
    opts = {
        "n_fft": spec_opts.get("n_fft", DEFAULT_OPTS["n_fft"]),
        "hop_length": spec_opts.get("hop_length", DEFAULT_OPTS["hop_length"]),
        "window": spec_opts.get("window", DEFAULT_OPTS["window"]),
    }

    spec = librosa.stft(wav, **opts)

    if spec_opts.get("type", DEFAULT_OPTS["type"]) == "mel":
        opts.update(
            {
                "n_mels": spec_opts.get("n_mels", DEFAULT_OPTS["n_mels"]),
                "sr": sample_rate,
            }
        )
        spec = librosa.feature.melspectrogram(S=np.abs(spec) ** 2, **opts)
        spec = spec.astype(np.float32)
    pcen = spec_opts.get("pcen", {})

    if pcen:
        pcen_opts = common_utils.deep_dict_update(DEFAULT_OPTS["pcen"], pcen, copy=True)
        opts["pcen"] = pcen_opts
        spec = librosa.pcen(spec * (2 ** 31), **pcen_opts)

    if spec_opts.get("to_db", False):
        spec = librosa.amplitude_to_db(spec, ref=np.max)

    opts["type"] = spec_opts.get("type", DEFAULT_OPTS["type"])
    return spec, opts


def resize_spectrogram(spec, size, resample_method="bicubic"):
    img = Image.fromarray(spec)
    if hasattr(Image, resample_method.upper()):
        resample_method = getattr(Image, resample_method.upper())
    else:
        resample_method = Image.BICUBIC
    img = img.resize(size, resample=resample_method)
    return np.array(img)

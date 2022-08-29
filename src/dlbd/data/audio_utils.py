import librosa
from mouffet import common_utils
import numpy as np
from PIL import Image

DEFAULT_PCEN_OPTS = {
    "gain": 0.8,
    "bias": 10,
    "power": 0.25,
    "time_constant": 0.06,
    "eps": 1e-06,
}

DEFAULT_SPEC_OPTS = {
    "sample_rate": "original",
    "n_fft": 2048,
    "hop_length": None,
    "n_mels": 32,
    "window": "hann",
    "win_length": None,
    "pcen": {},
    "type": "mel",
    "to_db": False,
    "center": True,
    "pad_mode": "reflect",
    "dtype": None,
}

STFT_OPTS_NAME = [
    "n_fft",
    "hop_length",
    "win_length",
    "window",
    "center",
    "dtype",
    "pad_mode",
]


def generate_spectrogram(wav, sample_rate, spec_opts):
    opts = common_utils.deep_dict_update(DEFAULT_SPEC_OPTS, spec_opts, copy=True)
    stft_opts = {k: v for k, v in opts.items() if k in STFT_OPTS_NAME}

    opts["win_length"] = (
        opts["win_length"] if opts["win_length"] is not None else opts["n_fft"]
    )
    opts["hop_length"] = (
        opts["hop_length"]
        if opts["hop_length"] is not None
        else opts["win_length"] // 4
    )

    spec = librosa.stft(wav, **stft_opts)

    if opts["type"] == "mel":
        stft_opts.update(
            {
                "n_mels": spec_opts.get("n_mels", DEFAULT_SPEC_OPTS["n_mels"]),
                "sr": sample_rate,
            }
        )
        spec = librosa.feature.melspectrogram(S=np.abs(spec) ** 2, **stft_opts)
        if opts["to_db"]:
            spec = librosa.power_to_db(spec, ref=np.max)
        spec = spec.astype(np.float32)

    pcen = spec_opts.get("pcen", {})

    if pcen:
        pcen_opts = common_utils.deep_dict_update(DEFAULT_PCEN_OPTS, pcen, copy=True)
        opts["pcen"] = pcen_opts
        spec = librosa.pcen(spec * (2 ** 31), **pcen_opts)

    return spec, opts


def resize_spectrogram(spec, size, resample_method="bicubic"):
    img = Image.fromarray(spec)
    if hasattr(Image, resample_method.upper()):
        resample_method = getattr(Image, resample_method.upper())
    else:
        resample_method = Image.BICUBIC
    img = img.resize(size, resample=resample_method)
    return np.array(img)


def load_audio_data(file_path, spec_opts):
    print("Loading audio file: " + str(file_path))
    sr = spec_opts.get("sample_rate", "original")
    if sr and sr == "original":
        sr = None
    # * NOTE: sample_rate can be different from sr if sr is None
    wav, sample_rate = librosa.load(str(file_path), sr=sr)
    # * NOTE: sp_opts can contain options not defined in spec_opts
    spec, sp_opts = generate_spectrogram(wav, sample_rate, spec_opts)
    metadata = {
        "file_path": file_path,
        "sample_rate": sample_rate,
        "length": len(wav),
        "duration": round(len(wav) / sample_rate, 1),
    }
    return spec, metadata, sp_opts

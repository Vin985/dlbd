import librosa
import numpy as np

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 1024  # 512
DEFAULT_N_MELS = 32  # 128


def generate_spectrogram(wav, sample_rate, spec_opts):
    if spec_opts["spec_type"] == "mel":
        spec = librosa.feature.melspectrogram(
            wav,
            sr=sample_rate,
            n_fft=spec_opts.get("n_fft", DEFAULT_N_FFT),
            hop_length=spec_opts.get("hop_length", DEFAULT_HOP_LENGTH),
            n_mels=spec_opts.get("n_mels", DEFAULT_N_MELS),
        )
        spec = spec.astype(np.float32)
    else:
        raise AttributeError("No other spectrogram supported yet")
    return spec

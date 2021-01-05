from time import time

import librosa
import numpy as np
from librosa.feature import melspectrogram
from tqdm import tqdm

from mouffet.models.dlmodel import DLModel


DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 1024  # 512
DEFAULT_N_MELS = 32  # 128


class AudioDLModel(DLModel):
    NAME = "AUDIODLMODEL"

    def __init__(self, opts=None):
        super().__init__(opts)
        self.wav = None
        self.sample_rate = None

    def classify(self, data, sampler):
        return self.classify_spectrogram(data, sampler)

    def classify_spectrogram(self, spectrogram, spec_sampler):
        """Apply the classifier"""
        tic = time()
        labels = np.zeros(spectrogram.shape[1])
        preds = []
        for data, _ in tqdm(spec_sampler([spectrogram], [labels])):
            pred = self.predict(data)
            preds.append(pred)
        print("Classified {0} in {1}".format("spectrogram", time() - tic))
        return np.vstack(preds)[:, 1]

    def load_wav(self, wavpath, loadmethod="librosa"):
        # tic = time()

        if loadmethod == "librosa":
            # a more correct and robust way -
            # this resamples any audio file to 22050Hz
            # TODO: downsample if higher than 22050
            sample_rate = self.opts.get("resample", None)
            print(sample_rate)
            return librosa.load(wavpath, sr=sample_rate)
        else:
            raise Exception("Unknown load method")

    def compute_spec(self, wav, sample_rate):
        # tic = time()
        spec = melspectrogram(
            wav,
            sr=sample_rate,
            n_fft=self.opts.get("n_fft", DEFAULT_N_FFT),
            hop_length=self.opts.get("hop_length", DEFAULT_HOP_LENGTH),
            n_mels=self.opts.get("n_mels", DEFAULT_N_MELS),
        )

        # if self.opts.remove_noise:
        #     spec = Spectrogram.remove_noise(spec)

        spec = np.log(self.opts["A"] + self.opts["B"] * spec)
        spec = spec - np.median(spec, axis=1, keepdims=True)
        return spec.astype(np.float32)

    def get_ground_truth(self, data):
        return data["tags_linear_presence"]

    def get_raw_data(self, data):
        return data["spectrograms"]

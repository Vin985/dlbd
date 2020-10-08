from pathlib import Path
from time import time

import librosa
import numpy as np
import yaml
from librosa.feature import melspectrogram

from ..data import utils
from ..training.spectrogram_sampler import SpectrogramSampler

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 1024  # 512
DEFAULT_N_MELS = 32  # 128


class DLModel:
    NAME = "DLMODEL"

    def __init__(self, opts=None, version=None):
        """Create the layers of the neural network, with the same options we used in training"""
        self.wav = None
        self.sample_rate = None
        self.model = None
        self._results_dir = None
        self._version = None
        self._opts = None
        self._model_name = ""
        self.results_dir_root = None
        self.version = version
        if opts:
            self.opts = opts

    @property
    def model_name(self):
        if not self._model_name:
            if self.version:
                self._model_name = self.NAME + "_v" + str(self.version)
            else:
                return self.NAME
        return self._model_name

    @property
    def version(self):
        if not self._version:
            self._version = self.get_model_version(self.results_dir_root)
        return self._version

    @version.setter
    def version(self, version):
        self._version = version

    @property
    def results_dir(self):
        return self.results_dir_root / str(self.version)

    @property
    def opts(self):
        return self._opts

    @opts.setter
    def opts(self, opts):
        self._opts = opts
        self.model = self.create_net()
        self.results_dir_root = Path(self.opts["model"]["model_dir"]) / self.NAME

    def create_net(self):
        return 0

    def predict(self, x):
        raise NotImplementedError("predict function not implemented for this class")

    def train(self, training_data, validation_data):
        raise NotImplementedError("train function not implemented for this class")

    def save_weights(self):
        raise NotImplementedError(
            "save_weights function not implemented for this class"
        )

    def load_weights(self, path=None):
        raise NotImplementedError(
            "load_weights function not implemented for this class"
        )

    def classify_spectrogram(self, spectrogram):
        raise NotImplementedError(
            "classify_spectrogram function not implemented for this class"
        )

    def prepare_data(self, data):
        return data

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

    def classify(self, wavpath=None):
        """Apply the classifier"""
        tic = time()

        if wavpath is None:
            raise AttributeError("No wave path is provided")

        wav, sr = self.load_wav(wavpath, loadmethod="librosa")
        spec = self.compute_spec(wav, sr)

        labels = np.zeros(spec.shape[1])
        # print("Took %0.3fs to load" % (time() - tic))
        tic = time()
        probas = []
        spec_sampler = SpectrogramSampler(self.opts)
        for x, _ in spec_sampler([spec], [labels]):
            pred = self.predict(x)
            probas.append(pred)
        # print("Took %0.3fs to classify" % (time() - tic))
        print("Classified {0} in {1}".format(wavpath, time() - tic))

        return {"preds": np.vstack(probas)[:, 1], "sr": sr}

    def save_params(self):
        utils.force_make_dir(self.results_dir)
        with open(self.results_dir / "network_opts.yaml", "w") as f:
            yaml.dump(self.opts, f, default_flow_style=False)

    @staticmethod
    def get_model_version(path):
        version = 1
        if path.exists():
            for item in path.iterdir():
                if item.is_dir():
                    try:
                        res = int(item.name)
                        if res >= version:
                            version = res + 1
                    except ValueError:
                        continue
        return version

    def save_model(self):
        self.save_params()
        self.save_weights()

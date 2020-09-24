import datetime
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

    def __init__(self, opts):
        """Create the layers of the neural network, with the same options we used in training"""
        self.opts = opts
        self.wav = None
        self.sample_rate = None
        self.model = self.create_net()
        self.results_dir = ""

    def create_net(self):
        raise NotImplementedError("create_net function not implemented for this class")

    def predict(self, x):
        raise NotImplementedError("predict function not implemented for this class")

    def train(self, training_data, validation_data):
        raise NotImplementedError("train function not implemented for this class")

    def save_weights(self, path):
        raise NotImplementedError(
            "save_weights function not implemented for this class"
        )

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
        if not self.results_dir:
            results_dir = (
                self.opts["model_dir"]
                + self.NAME
                + "/"
                + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                + "/"
            )
            utils.force_make_dir(results_dir)
            self.results_dir = results_dir
        # sys.stdout = ui.Logger(logging_dir + "log.txt")

        with open(self.results_dir + "network_opts.yaml", "w") as f:
            yaml.dump(self.opts, f, default_flow_style=False)

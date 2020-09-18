import datetime
from time import time


import librosa
import yaml
import numpy as np
from librosa.feature import melspectrogram

from analysis.detection.lib.spectrogram_sampler import SpectrogramSampler
import utils.commons as commons


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
        self.test_sampler = SpectrogramSampler(self.opts)

    def create_net(self):
        return 0

    def run_net(self, input):
        return 0

    def train(self, training_data, validation_data):
        return 0

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

        if wavpath is not None:
            wav, sr = self.load_wav(wavpath, loadmethod="librosa")
            spec = self.compute_spec(wav, sr)

        labels = np.zeros(spec.shape[1])
        # print("Took %0.3fs to load" % (time() - tic))
        tic = time()
        probas = []
        for Xb, _ in self.test_sampler([spec], [labels]):
            pred = self.run_net(
                input=Xb
            )  # self.sess.run(self.net["output"], feed_dict={self.net["input"]: Xb})
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
            commons.force_make_dir(results_dir)
            self.results_dir = results_dir
        # sys.stdout = ui.Logger(logging_dir + "log.txt")

        with open(self.results_dir + "network_opts.yaml", "w") as f:
            yaml.dump(self.opts, f, default_flow_style=False)

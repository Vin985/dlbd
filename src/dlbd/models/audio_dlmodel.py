from time import time

import librosa
import numpy as np

# from librosa.feature import melspectrogram
from mouffet.models.dlmodel import DLModel
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

from ..data.spectrogram import resize_spectrogram

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 1024  # 512
DEFAULT_N_MELS = 32  # 128


class AudioDLModel(DLModel):
    NAME = "AUDIODLMODEL"

    def __init__(self, opts=None):
        super().__init__(opts)
        # self.wav = None
        # self.sample_rate = None

    # def load_wav(self, wavpath, loadmethod="librosa"):
    #     # tic = time()

    #     if loadmethod == "librosa":
    #         # a more correct and robust way -
    #         # this resamples any audio file to 22050Hz
    #         # TODO: downsample if higher than 22050
    #         sample_rate = self.opts.get("resample", None)
    #         print(sample_rate)
    #         return librosa.load(wavpath, sr=sample_rate)
    #     else:
    #         raise Exception("Unknown load method")

    # def compute_spec(self, wav, sample_rate):
    #     # tic = time()
    #     spec = melspectrogram(
    #         wav,
    #         sr=sample_rate,
    #         n_fft=self.opts.get("n_fft", DEFAULT_N_FFT),
    #         hop_length=self.opts.get("hop_length", DEFAULT_HOP_LENGTH),
    #         n_mels=self.opts.get("n_mels", DEFAULT_N_MELS),
    #     )

    #     # if self.opts.remove_noise:
    #     #     spec = Spectrogram.remove_noise(spec)

    #     spec = np.log(self.opts["A"] + self.opts["B"] * spec)
    #     spec = spec - np.median(spec, axis=1, keepdims=True)
    #     return spec.astype(np.float32)

    def get_ground_truth(self, data):
        return data["tags_linear_presence"]

    def get_raw_data(self, data):
        return data["spectrograms"]

    def modify_spectrogram(self, spec, resize_width):
        spec = np.log(self.opts["model"]["A"] + self.opts["model"]["B"] * spec)
        spec = spec - np.median(spec, axis=1, keepdims=True)
        if resize_width > 0:
            spec = resize_spectrogram(spec, (resize_width, spec.shape[0]))
        return spec

    def prepare_data(self, data):
        if not self.opts["model"]["learn_log"]:
            for i, spec in enumerate(data["spectrograms"]):
                resize_width = self.get_resize_width(data["infos"][i])

                # Issue a warning if the number of pixels desired is too far from the original size
                original_pps = (
                    data["infos"][i]["sample_rate"]
                    / data["infos"][i]["spec_opts"]["hop_length"]
                )
                new_pps = self.opts["model"]["pixels_in_sec"]
                if new_pps / original_pps > 2 or new_pps / original_pps < 0.5:
                    common_utils.print_warning(
                        (
                            "WARNING: The number of pixels per seconds when resizing -{}-"
                            + " is far from the original resolution -{}-. Consider changing the pixels_per_sec"
                            + " option or the hop_length of the spectrogram so the two values can be closer"
                        ).format(new_pps, original_pps)
                    )
                data["spectrograms"][i] = self.modify_spectrogram(spec, resize_width)
                if resize_width > 0:
                    data["tags_linear_presence"][i] = zoom(
                        data["tags_linear_presence"][i],
                        float(resize_width) / spec.shape[1],
                        order=1,
                    ).astype(int)
        return data

    def get_resize_width(self, infos):
        resize_width = -1
            pix_in_sec = self.opts["model"].get("pixels_in_sec", 20)
            resize_width = int(pix_in_sec * infos["length"] / infos["sample_rate"])
        return resize_width

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

    def classify(self, data, sampler):
        spectrogram, infos = data
        spectrogram = self.modify_spectrogram(spectrogram, self.get_resize_width(infos))
        return self.classify_spectrogram(spectrogram, sampler)

from time import time

import numpy as np

# from librosa.feature import melspectrogram
from mouffet.models.dlmodel import DLModel
import mouffet.utils.common as common_utils
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

from ..data.spectrogram import resize_spectrogram

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 1024  # 512
DEFAULT_N_MELS = 32  # 128


class AudioDLModel(DLModel):
    NAME = "AUDIODLMODEL"

    def __init__(self, opts=None):
        input_size = opts["net"].get("input_size", opts["model"]["pixels_in_sec"])
        if input_size % 2:
            common_utils.print_warning(
                (
                    "Network input size is odd. For performance reasons,"
                    + " input_size should be even. {} will be used as input size instead of {}."
                    + " Consider changing the pixel_per_sec or input_size options in the configuration file"
                ).format(input_size + 1, input_size)
            )
            if input_size == opts["model"]["pixels_in_sec"]:
                common_utils.print_warning(
                    (
                        "Input size and pixels per seconds were identical, using {} pixels per seconds as well"
                    ).format(input_size + 1)
                )
            input_size += 1
            opts["model"]["pixels_in_sec"] = input_size
        opts["net"]["input_size"] = input_size
        super().__init__(opts)

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

                # * Issue a warning if the number of pixels desired is too far from the original size
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

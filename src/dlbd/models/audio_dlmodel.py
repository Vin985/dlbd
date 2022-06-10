from time import time

import numpy as np

from mouffet.models import DLModel
from mouffet import common_utils
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

from ..data.audio_utils import resize_spectrogram


class AudioDLModel(DLModel):
    NAME = "AUDIODLMODEL"

    def __init__(self, opts=None):
        input_width = opts.get("input_width", opts["pixels_per_sec"])
        if input_width % 2:
            common_utils.print_warning(
                (
                    "Network input size is odd. For performance reasons,"
                    + " input_width should be even. {} will be used as input size instead of {}."
                    + " Consider changing the pixel_per_sec or input_width options in the configuration file"
                ).format(input_width + 1, input_width)
            )
            if input_width == opts["pixels_per_sec"]:
                common_utils.print_warning(
                    (
                        "Input size and pixels per seconds were identical, using {} pixels per seconds as well"
                    ).format(input_width + 1)
                )
            input_width += 1
            opts.add_option("pixels_per_sec", input_width)
        opts.add_option("input_width", input_width)
        super().__init__(opts)

    # def get_ground_truth(self, data):
    #     return data["tags_linear_presence"]

    # def get_raw_data(self, data):
    #     return data["spectrograms"]

    # def modify_spectrogram(self, spec, resize_width, to_db=False):
    #     if not to_db:
    #         spec = np.log(self.opts["A"] + self.opts["B"] * spec)
    #     spec = spec - np.median(spec, axis=1, keepdims=True)
    #     if resize_width > 0:
    #         spec = resize_spectrogram(spec, (resize_width, spec.shape[0]))
    #     return spec

    # def prepare_data(self, dataset):
    #     if not self.opts["learn_log"]:
    #         for i, spec in enumerate(dataset.data["spectrograms"]):
    #             infos = dataset.data["infos"][i]
    #             spec_opts = infos["spec_opts"]
    #             infos["duration"] = infos["length"] / infos["sample_rate"]
    #             resize_width = self.get_resize_width(infos)

    #             if (
    #                 spec_opts["type"] == "mel"
    #                 and self.opts.get("input_height", -1) != spec_opts["n_mels"]
    #             ):
    #                 self.opts.add_option("input_height", spec_opts["n_mels"])

    #             # * Issue a warning if the number of pixels desired is too far from the original size
    #             original_pps = infos["sample_rate"] / spec_opts["hop_length"]
    #             new_pps = self.opts["pixels_per_sec"]
    #             if self.opts.get("verbose", False) and (
    #                 new_pps / original_pps > 2 or new_pps / original_pps < 0.5
    #             ):
    #                 common_utils.print_warning(
    #                     (
    #                         "WARNING: The number of pixels per seconds when resizing -{}-"
    #                         + " is far from the original resolution -{}-. Consider changing the pixels_per_sec"
    #                         + " option or the hop_length of the spectrogram so the two values can be closer"
    #                     ).format(new_pps, original_pps)
    #                 )
    #             dataset.data["spectrograms"][i] = self.modify_spectrogram(
    #                 spec, resize_width, to_db=spec_opts["to_db"]
    #             )
    #             if resize_width > 0:
    #                 dataset.data["tags_linear_presence"][i] = zoom(
    #                     dataset.data["tags_linear_presence"][i],
    #                     float(resize_width) / spec.shape[1],
    #                     order=1,
    #                 ).astype(int)
    #     return dataset

    # def get_resize_width(self, infos):
    #     resize_width = -1
    #     pix_in_sec = self.opts.get("pixels_per_sec", 20)
    #     resize_width = int(pix_in_sec * infos["duration"])
    #     return resize_width

    def classify_spectrogram(self, spectrogram, spec_sampler):
        """Apply the classifier"""
        tic = time()
        labels = np.zeros(spectrogram.shape[1])
        preds = []
        # count = 0
        # ig = image.ImageGenerator(image_options=image.ImageOptions())
        for data, _ in tqdm(spec_sampler([spectrogram], [labels])):
            # for i in range(0, data.shape[0]):
            #     count += 1
            #     img = ig.spec2img(data[i], size=None, is_array=True)
            #     img.save(
            #         file_utils.ensure_path_exists(
            #             "test/specs2/test_spectrogram_{}.png".format(count),
            #             is_file=True,
            #         )
            #     )
            pred = self.predict(data)
            preds.append(pred)
        print("Classified {0} in {1}".format("spectrogram", time() - tic))
        return np.vstack(preds)[:, 1]

    def classify(self, data, sampler):
        spectrogram, infos = data
        spectrogram = self.modify_spectrogram(spectrogram, self.get_resize_width(infos))

        # ig = image.ImageGenerator(image_options=image.ImageOptions())
        # img = ig.spec2img(spectrogram[:, 0:500], size=None, is_array=True)
        # img.save("test_spectrogram.png")

        return self.classify_spectrogram(spectrogram, sampler)

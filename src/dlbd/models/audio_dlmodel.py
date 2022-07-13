# from time import time

# import numpy as np

# from mouffet.models import DLModel
# from mouffet import common_utils
# from tqdm import tqdm

# from .TF2Model import TF2Model
# from mouffet.models import Model


# class AudioTF2Model(TF2Model):
#     NAME = "AUDIOMODEL"

#     def __init__(self, opts=None):
#         input_width = opts.get("input_width", opts["pixels_per_sec"])
#         if input_width % 2:
#             common_utils.print_warning(
#                 (
#                     "Network input size is odd. For performance reasons,"
#                     + " input_width should be even. {} will be used as input size instead of {}."
#                     + " Consider changing the pixel_per_sec or input_width options in the configuration file"
#                 ).format(input_width + 1, input_width)
#             )
#             if input_width == opts["pixels_per_sec"]:
#                 common_utils.print_warning(
#                     (
#                         "Input size and pixels per seconds were identical, using {} pixels per seconds as well"
#                     ).format(input_width + 1)
#                 )
#             input_width += 1
#             opts.add_option("pixels_per_sec", input_width)
#         opts.add_option("input_width", input_width)
#         super().__init__(opts)

#     # def predict_spectrogram(self, spectrogram, spec_sampler):
#     #     """Apply the classifier"""
#     #     tic = time()
#     #     labels = np.zeros(spectrogram.shape[1])
#     #     preds = []
#     #     for data, _ in tqdm(spec_sampler([spectrogram], [labels])):
#     #         pred = self.predict(data)
#     #         preds.append(pred)
#     #     print("Classified {0} in {1}".format("spectrogram", time() - tic))
#     #     return np.vstack(preds)[:, 1]

#     def predict_spectrogram(self, data):
#         spectrogram, spec_sampler = data
#         """Apply the classifier"""
#         tic = time()
#         labels = np.zeros(spectrogram.shape[1])
#         preds = []
#         for x, _ in tqdm(spec_sampler([spectrogram], [labels])):
#             pred = self.predict(x)
#             preds.append(pred)
#         print("Classified {0} in {1}".format("spectrogram", time() - tic))
#         return np.vstack(preds)[:, 1]

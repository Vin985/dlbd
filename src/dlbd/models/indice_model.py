from . import AudioDetector
import numpy as np
from time import time
from tqdm import tqdm


class IndiceModel(AudioDetector):
    def create_model(self):
        pass

    @staticmethod
    def compute_ACI(spec):
        # time_step = self.opts.get("time_step", None)
        # if time_step is None:
        j_bin = int(spec.shape[1] / 10)
        # else:
        #     if self.unit == "seconds":
        #         j_bin = int(self.time_step * self.spec.shape[1] / self.duration)
        #     elif self.unit == "frames":
        #         j_bin = self.time_step

        # alternative time indices to follow the R code
        times = range(0, spec.shape[1] - 10, j_bin)
        # sub-spectros of temporal size j
        jspecs = [np.array(spec[:, i : i + j_bin]) for i in times]
        # list of ACI values on each jspecs
        aci = [
            sum((np.sum(abs(np.diff(jspec)), axis=1) / np.sum(jspec, axis=1)))
            for jspec in jspecs
        ]
        # self.temporal_values = aci
        # self.ACI = sum(aci)
        return sum(aci)

    def predict(self, x):
        res = np.zeros(x.shape[0])
        for i, data in enumerate(x):
            res[i] = self.compute_ACI(data)
        # res = (res - res.mean()) / res.std()
        res = (res - np.min(res)) / (np.max(res) - np.min(res))
        # res = np.log(res / (1 - res))
        return res

    def save_model(self, path=None):
        pass

    def load_weights(self):
        pass

    def predict_spectrogram(self, data):
        spectrogram, spec_sampler = data
        """Apply the classifier"""
        tic = time()
        labels = np.zeros(spectrogram.shape[1])
        preds = []
        for x, _ in tqdm(spec_sampler([spectrogram], [labels])):
            pred = self.predict(x)
            preds.append(pred)
        print("Classified {0} in {1}".format("spectrogram", time() - tic))
        return np.concatenate(preds)

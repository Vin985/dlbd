from abc import ABC, abstractmethod
from pathlib import Path

import yaml

from ..utils import file as file_utils

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 1024  # 512
DEFAULT_N_MELS = 32  # 128


class DLModel(ABC):
    NAME = "DLMODEL"

    STEP_TRAINING = "train"
    STEP_VALIDATION = "validation"

    def __init__(self, opts=None, version=None):
        """Create the layers of the neural network, with the same options we used in training"""
        self.model = None
        self._results_dir = None
        # self._version = None
        self._opts = None
        # self._model_name = ""
        # self.results_dir_root = None
        # self.version = version
        if opts:
            self.opts = opts

    # @property
    # def model_name(self):
    #     if not self._model_name:
    #         if self.version is not None:
    #             self._model_name = self.NAME + "_v" + str(self.version)
    #         else:
    #             return self.NAME
    #     return self._model_name

    # @property
    # def version(self):
    #     if self._version is None:
    #         v = self.get_model_version(self.results_dir_root)
    #         if self.opts["model"].get("from_epoch", 0) and v > 0:
    #             v -= 1
    #         self._version = v
    #     return self._version

    # @version.setter
    # def version(self, version):
    #     self._version = version

    # @property
    # def results_dir(self):
    #     return self.results_dir_root / str(self.version)

    @property
    def opts(self):
        return self._opts

    @opts.setter
    def opts(self, opts):
        self._opts = opts
        self.model = self.create_net()
        # self.results_dir_root = Path(self.opts["model_dir"]) / self.NAME

    @abstractmethod
    def create_net(self):
        return 0

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError("predict function not implemented for this class")

    @abstractmethod
    def train(self, training_data, validation_data):
        raise NotImplementedError("train function not implemented for this class")

    @abstractmethod
    def save_weights(self, path=None):
        raise NotImplementedError(
            "save_weights function not implemented for this class"
        )

    @abstractmethod
    def load_weights(self, path=None):
        raise NotImplementedError(
            "load_weights function not implemented for this class"
        )

    @abstractmethod
    def classify(self, data, sampler):
        return None

    @abstractmethod
    def get_ground_truth(self, data):
        return data

    @abstractmethod
    def get_raw_data(self, data):
        return data["spectrograms"]

    def prepare_data(self, data):
        return data

    def save_params(self):
        file_utils.ensure_path_exists(self.results_dir)
        with open(self.results_dir / "network_opts.yaml", "w") as f:
            yaml.dump(self.opts, f, default_flow_style=False)

    # @staticmethod
    # def get_model_version(path):
    #     version = 1
    #     if path.exists():
    #         for item in path.iterdir():
    #             if item.is_dir():
    #                 try:
    #                     res = int(item.name)
    #                     if res >= version:
    #                         version = res + 1
    #                 except ValueError:
    #                     continue
    #     return version

    def save_model(self, path=None):
        self.save_params()
        self.save_weights(path)

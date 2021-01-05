import re

from .options import Options
from ..utils import common as common_utils


class ModelOptions(Options):

    DEFAULT_VALUES = {
        "id": "",
        "id_prefixes": {},
        "intermediate_save_dir": "intermediate",
    }

    def __init__(self, opts):
        super().__init__(opts)
        self._model_id = ""
        self._version = None

    @property
    def results_dir_root(self):
        return self.model_dir / self.model_id

    @property
    def results_dir(self):
        return self.results_dir_root / str(self.version)

    def get_intermediate_path(self, epoch, version=None, as_string=True):
        """Get the path where intermediate results for a specific epoch are saved

        Args:
            epoch (int): The epoch where the results are saved
            version (int, optional): An optional version number to provide. If None,
            use current version number (for saving). If provided and positive, use that
            version number. If provided and negative, use the previous version number.
            Defaults to None.
            as_string (bool, optional): Returns the result as a string instead of a pathlib.Path.
            Defaults to True.

        Returns:
            [type]: [description]
        """
        # * By default, use the current version results dir
        res_dir = self.results_dir
        if version:
            if version > 0:
                # * A positive version number is provided, use this number
                res_dir = self.results_dir_root / str(version)
            else:
                # * The version number is negative, use previous version
                res_dir = self.results_dir_root / str(self.version - 1)
        path = res_dir / self.intermediate_save_dir / ("epoch_" + str(epoch))
        if as_string:
            return str(path)
        return path

    @property
    def model_id(self):
        if not self._model_id:
            self._model_id = self.name + self.resolve_id(self.id)
            # self.opts["model_id"] = self._model_id
        return self._model_id

    def resolve_id(self, model_id):
        prefixes = self.id_prefixes
        to_replace = re.findall("\\{(.+?)\\}", model_id)
        res = {}
        for key in to_replace:
            mid = ""
            if prefixes:
                prefix = prefixes.get(key, prefixes.get("default", ""))
                mid += str(prefix)
            mid += str(common_utils.get_dict_path(self.opts, key, key))
            res[key] = mid

        mid = model_id.format(**res)
        return mid

    @property
    def version(self):
        if self._version is None:
            v = self.opts.get("version", None)
            if not v:
                v = self.get_model_version(self.results_dir_root)
            self._version = v
        return self._version

    def get_model_version(self, path):
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
        if self.opts["model"].get("from_epoch", 0) and version > 0:
            version -= 1
        return version

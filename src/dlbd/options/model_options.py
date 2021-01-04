import re

from .options import Options
from ..utils import common as common_utils


class ModelOptions(Options):

    DEFAULT_VALUES = {"id": "{version}", "id_prefixes": {"version": "_v"}}

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

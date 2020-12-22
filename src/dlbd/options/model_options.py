from .options import Options


class ModelOptions(Options):

    DEFAULT_VALUES = {"id": ""}

    def __init__(self, opts):
        super().__init__(opts)
        self._model_name = ""
        self._model_id = ""
        self._version = None

    @property
    def results_dir_root(self):
        return self.model_dir / self.name

    @property
    def results_dir(self):
        return self.results_dir_root / str(self.version)

    @property
    def model_name(self):
        if not self._model_name:
            self._model_name = self.name + "_v" + str(self.version)
        return self._model_name

    @property
    def model_id(self):
        if not self._model_id:
            self._model_id = self.model_name
            mid = self.id
            if mid:
                self._model_id += "_" + self.resolve_id(mid)
            self.opts["model_id"] = self._model_id
        return self._model_id

    def resolve_id(self, model_id):
        mid = model_id
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

from .options import Options


class ModelOptions(Options):

    DEFAULT_VALUES = {"id": "", "version": 1}

    def __init__(self, opts):
        super().__init__(opts)
        self._model_name, self._model_id = "", ""

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
                self._model_id += "_" + mid
            self.opts["model_id"] = self._model_id
        return self._model_id

from pathlib import Path


class Options:

    DEFAULT_VALUES = {"options": {}}

    def __init__(self, opts, updated_opts=None):
        self.opts = opts
        self.updated_opts = updated_opts

    def __bool__(self):
        return bool(self.opts)

    def __getstate__(self):
        return self.opts

    def __getattr__(self, name):
        value = self.opts.get(name, self.DEFAULT_VALUES.get(name, None))
        if value is None:
            raise ValueError(
                "No option {} is present in this {} object. Please check the config file.".format(
                    name, self.__class__
                )
            )
        if isinstance(value, str) and (name.endswith("_dir") or name.endswith("_path")):
            value = Path(value)
        return value

    def __repr__(self) -> str:
        return repr(self.opts)

    def __str__(self) -> str:
        return str(self.opts)

    def __format__(self, format_spec: str) -> str:
        return super().__format__(format_spec)

    def __contains__(self, item):
        return item in self.opts

    def get_option(self, option):
        if "options" in self.opts:
            return self.opts["options"].get(
                option, self.DEFAULT_VALUES["options"][option]
            )
        return self.DEFAULT_VALUES["options"][option]

    def get(self, value, default):
        return self.opts.get(value, default)

    def __getitem__(self, key):
        return getattr(self, key)

from .options import Options
from ..utils import common as common_utils


class DatabaseOptions(Options):

    DB_TYPES = ["test", "training", "validation"]

    DEFAULT_VALUES = {
        "class_type": "",
        "classes_file": "classes.csv",
        "db_types": DB_TYPES,
        "data_extensions": [""],
        "generate_file_lists": False,
        "overwrite": False,
        "recursive": False,
        "save_intermediates": False,
        "tags_suffix": "-sceneRect.csv",
        "use_subfolders": None,
    }

    @property
    def types(self):
        return self.opts.get("db_types", self.DB_TYPES)

    def has_type(self, db_type):
        return db_type in self.types


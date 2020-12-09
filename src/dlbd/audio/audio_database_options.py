from ..options.database_options import DatabaseOptions


class AudioDatabaseOptions(DatabaseOptions):

    DEFAULT_VALUES = DatabaseOptions.DEFAULT_VALUES.copy()
    DEFAULT_VALUES.update(
        {
            "class_type": "biotic",
            "data_extensions": [".wav"],
            "classes_file": "classes.csv",
            "tags_suffix": "-sceneRect.csv",
        }
    )


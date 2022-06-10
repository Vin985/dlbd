from mouffet.data import Database
from .audio_dataset import AudioDataset
from .split import citynet_split
from mouffet.data.split import split_folder


class AudioDatabase(Database):

    # DEFAULT_VALUES.update(
    #     {
    #         "class_type": "biotic",
    #         "data_extensions": [".wav"],
    #         "classes_file": "classes.csv",
    #         "tags_suffix": "-sceneRect.csv",
    #     }
    # )

    DATASET = AudioDataset

    SPLIT_FUNCS = {"arctic": split_folder, "citynet": citynet_split}

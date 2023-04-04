from mouffet.data import Database
from .audio_dataset import AudioDataset
from .split import citynet_split
from mouffet.data.split import split_folder


class AudioDatabase(Database):

    DATASET = AudioDataset

    SPLIT_FUNCS = {
        "arctic": split_folder,
        "citynet": citynet_split,
    }

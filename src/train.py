import os

import yaml

from training.CityNet_trainer import CityNetTrainer
from training.utils import create_detection_dataset, train_citynet

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# try:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# except:
#     pass


os.sys.path.insert(0, "/mnt/win/UMoncton/Doctorat/dev/ecosongs/src")
try:
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from analysis.detection.models.CityNetTF2 import CityNetTF2
    from analysis.detection.models.DCASE_SpeechLab import DCASESpeechLab
except Exception:
    print("Woops, module not found")

if __name__ == "__main__":

    stream = open("src/test/citynet2/CONFIG.yaml", "r")
    opts = yaml.load(stream, Loader=yaml.Loader)
    opts["model_name"] = "citynet_augmented1"
    print(opts)

    # trainer = CityNetTrainer(opts)
    # trainer.create_detection_datasets()

    model = CityNetTF2(opts)

    trainer = CityNetTrainer(opts, model)

    trainer.train()
    # trainer.train_model2()
    # spec_dir = generate_spectrograms(extracted_dir, extracted_dir + "spectrograms/")

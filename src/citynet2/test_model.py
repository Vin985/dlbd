#%%
import os
import yaml

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, "/mnt/win/UMoncton/Doctorat/dev/ecosongs/src")
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from analysis.detection.models.CityNetTF2 import CityNetTF2
    from analysis.detection.models.DCASE_SpeechLab import DCASESpeechLab
except Exception:
    print("Woops, module not found")


stream = open(
    "/mnt/win/UMoncton/Doctorat/dev/ecosongs/src/test/citynet2/CONFIG.yaml", "r"
)
opts = yaml.load(stream, Loader=yaml.Loader)
opts["model_name"] = "test_citynettf2"
print(opts)


model = CityNetTF2(opts)
model2 = DCASESpeechLab(opts)

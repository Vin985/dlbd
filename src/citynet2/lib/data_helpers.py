import pandas as pd
import librosa
import os
import yaml

# base = yaml.load(open("CONFIG.yaml"), Loader=yaml.Loader)["base_dir"]
# labels_dir = base + "/annotations/"
# wav_dir = base + "/wavs/"

HUMAN_NOISES = set(
    [
        "mix traffic",
        "braking",
        "voices",
        "electrical",
        "anthropogenic unknown",
        "airplane",
        "beep",
        "metal",
        "bus emitting",
        "footsteps",
        "mower",
        "whistle",
        "siren",
        "coughing",
        "music",
        "horn",
        "startthecar",
        "bells",
        "applause",
        "dog bark",
        "road traffic",
        "braking vehicle (road or rail)",
        "human voices",
        "mechanical",
        "vehicle horn (road or rail)",
        "air traffic",
        "vehicle alarm",
        "human voice",
        "machinery",
        "church bell",
        "breaking vehicle",
        "deck lowering",
        "car horn",
        "rail traffic",
        "alarm",
        "vehicle horn",
        "building ventilation system",
        "car alarm",
        "rock",
        "church bells",
        "train horn",
        "mobile phone",
        "train station announcement",
        "hammering",
        "door opening",
        "dog barking",
        "vehicle breaking",
        "cat",
        "glass into bins",
        "barking dog",
        "television",
        "sweeping broom",
        "ball bouncing",
        "bat hitting ball",
        "laughing",
        "clapping",
        "camera",
        "train doors (beeping)",
        "lawnmower",
    ]
)

ANIMAL_NOISES = set(
    [
        "bird",
        "wing beats",
        "bat",
        "fox",
        "grey squirrel",
        "invertebrate",
        "insect",
        "animal",
        "wings beating",
        "russling leaves (animal)",
        "amphibian",
        "squirrel",
        "russling vegetation (animal)",
        # NEW LABELS
        # TODO: find better way to implement this
        "amgp_1(tu-tit)",
        "amgp_2(tuuut)",
        "amgp_3(other)",
        "bran",
        "cago",
        "dunl(descending trill)",
        "dunl(rising trill)",
        "goose",
        "gull",
        "hegu",
        "kiei",
        "lalo(chirp)",
        "lalo(song)",
        "loon",
        "ltdu",
        "ltdu(yodel)",
        "passerine(chirp)",
        "passerine(song)",
        "pesa(hoohoohoo)",
        "plover",
        "pusa(tututu)",
        "pusa(kreeekreeekreee)",
        "reph(chirp)",
        "rtlo",
        "rtlo(duet)",
        "sagu",
        "sand(chirp)",
        "sand(trill)",
        "sesa_2(motorboat)",
        "sesa_3(titutu)",
        "shorebird(chirp)",
        "shorebird(song)",
        "snbu(chirp)",
        "snbu(song)",
        "sngo",
        "tusw",
        "unkn",
        "wrsa_1(boingboingboing)",
    ]
)

OTHER = set(
    [
        "rain",
        "unknown sound",
        "electrical disturbance",
        "vegetation",
        "wind",
        "unknown",
        "metalic sound",
        "dripping water",
        "shower",
        "metalic",
        "rubbish bag",
        "water dripping",
        "water splashing",
        "rainfall on vegetation",
    ]
)

BIRDS = set(["SESA", "AMGP", "Shorebird", "Passerine"])


def load_annotations(audio_file_path, labels_dir):

    # load file and convert to spectrogram
    wav, sample_rate = librosa.load(str(audio_file_path), None)

    # create label vector...
    biotic = 0 * wav
    anthropogenic = 0 * wav

    csv_file_path = labels_dir / (audio_file_path.stem + "-sceneRect.csv")
    print("Loading annotations for file: " + str(audio_file_path))
    if os.path.exists(csv_file_path):
        pd_annots = pd.read_csv(csv_file_path, skip_blank_lines=True)
        # loop over each annotation...
        tmp = pd_annots.loc[~pd_annots.Filename.isna()]
        for _, annot in tmp.iterrows():
            # fill in the label vector
            start_point = int(float(annot["LabelStartTime_Seconds"]) * sample_rate)
            end_point = int(float(annot["LabelEndTime_Seconds"]) * sample_rate)

            label = annot["Label"].lower()
            if label in HUMAN_NOISES:
                anthropogenic[start_point:end_point] = 1
            elif label in ANIMAL_NOISES:
                biotic[start_point:end_point] = 1
            elif label in OTHER:
                pass
            else:
                print("Unknown label ", annot["Label"])
                # raise Exception("Unknown label ", annot["Label"])
    else:
        pd_annots = pd.DataFrame()
        print("Warning - no annotations found for %s" % str(audio_file_path))

    return {"anthrop": anthropogenic, "biotic": biotic}, wav, sample_rate

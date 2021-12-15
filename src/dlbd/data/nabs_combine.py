#%%

import math
import pickle
import random
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile
from mouffet import file_utils


def get_white_noise(signal, SNR, shape=None):
    # RMS value of signal
    RMS_s = math.sqrt(np.mean(signal ** 2))
    # RMS values of noise
    RMS_n = math.sqrt(RMS_s ** 2 / (pow(10, SNR / 10)))
    # Additive white gausian noise. Thereore mean=0
    # Because sample length is large (typically > 40000)
    # we can use the population formula for standard daviation.
    # because mean=0 STD=RMS
    STD_n = RMS_n
    if not shape:
        shape = signal.shape[0]
    noise = np.random.normal(0, STD_n, shape)
    return noise


def NABS_to_df(file_list):
    df = pd.DataFrame({"path": [str(path) for path in file_list]})
    splits = df.path.str.split("/")

    df["dataset"] = splits.str[-3]
    df["tag"] = splits.str[-2]
    df["tag_index"] = splits.str[-1].str.extract(".+\\((.+)\\).*").astype("int")
    df = df.reset_index().rename(columns={"index": "id"})
    return df


def load_audios(file_list, opts):
    sr = opts.get("sr", 16000)
    preloaded_file = Path(opts.get("src_dir")) / "preloaded_audios_{}.pkl".format(sr)
    if preloaded_file.exists():
        print("Loading preloaded file {}".format(preloaded_file))
        return pickle.load(open(preloaded_file, "rb"))

    res = [librosa.load(str(file_path), sr=sr)[0] for file_path in file_list]

    if opts.get("load_preloaded", True) and not preloaded_file.exists():
        with open(
            file_utils.ensure_path_exists(preloaded_file, is_file=True), "wb"
        ) as f:
            pickle.dump(res, f, -1)
            print("Saved file: ", preloaded_file)

    return res


def generate_mix(audios, file_df, opts):
    # Shuffle all files
    tags = np.random.permutation(file_df.id)
    # Index of used file
    idx = 0
    file_count = opts.get("start_file_count", 1)
    sr = opts.get("sr", 16000)
    duration = opts.get("duration", 30)
    nframes = sr * duration

    print(len(audios))
    print(len(tags))
    print(max(tags))
    print(max(file_df.id))

    # While we still have unused files, create new mixes
    for i in range(opts.get("n_iter", 1)):
        print("Performing iteration no: ", i + 1)
        while idx < tags.shape[0]:
            tic = time.time()
            tag_infos = []

            # Create empty audio file
            generated = np.zeros(sr * duration)
            # Create empty labels
            labels = np.zeros(sr * duration)
            noisy = False
            start_pos = np.random.choice(range(0, generated.shape[0]), 1000)
            max_filled = random.uniform(0, opts.get("max_filled", 0.3))

            for start in start_pos:
                if idx > tags.shape[0]:
                    break
                audio = audios[tags[idx]]
                if not noisy:
                    snr = random.uniform(20, 50)
                    noise = get_white_noise(audio, snr, len(generated))
                    generated += noise
                    noisy = True

                end = min(start + len(audio), len(generated) - 1)

                if not opts.get("allow_overlap", False) and (
                    labels[start] == 1 or labels[end] == 1
                ):
                    continue

                dur = min(end - start, len(audio))
                generated[start:end] += audio[0:dur]
                labels[start:end] = 1
                tag_infos.append(
                    {
                        "tag": file_df.tag.iloc[tags[idx]],
                        "start": start / sr,
                        "end": end / sr,
                    }
                )
                prop_filled = sum(labels) / nframes
                # print(prop_filled)
                idx += 1
                if idx >= tags.shape[0]:
                    break
                if prop_filled > max_filled:
                    break

            file_name = Path(opts.get("dest_dir", ".")) / "mix_{}.wav".format(
                file_count
            )
            soundfile.write(
                file_name,
                generated,
                opts.get("sr", 16000),
            )
            pd.DataFrame(tag_infos).to_csv(
                Path(opts.get("dest_dir", ".")) / "mix_{}_tags.csv".format(file_count)
            )

            labels_file = Path(opts.get("dest_dir", ".")) / "mix_{}_labels.pkl".format(
                file_count
            )
            with open(
                file_utils.ensure_path_exists(labels_file, is_file=True), "wb"
            ) as f:
                pickle.dump(labels, f, -1)
            print("Generated file {} in {}s.".format(file_name, time.time() - tic))
            file_count += 1
        idx = 0


#%%

opts = {
    "sr": 16000,
    "duration": 30,
    "noise": True,
    "class:overlap": 0.3,
    "max_filled": 0.3,
    "allow_overlap": False,
    "dest_dir": "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Reference/generated/NABS/2",
    "src_dir": "/home/vin/Doctorat/data/dl_training/raw/NABS",
    "n_iter": 10,
    "extensions": [".wav", ".WAV"],
    "recursive": True,
    "start_file_count": 697,
}

file_list = file_utils.list_files(
    Path(opts["src_dir"]),
    opts.get("extensions", [".wav", ".WAV"]),
    opts.get("recursive", True),
)


file_df = NABS_to_df(file_list)
#%%


audios = load_audios(file_list, opts)


#%%


generate_mix(audios, file_df, opts)

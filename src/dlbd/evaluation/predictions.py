import time
from pathlib import Path

import numpy as np
import pandas as pd
from dlbd.data import audio_utils

from ..training.spectrogram_sampler import SpectrogramSampler


def classify_elements(elements, model, spec_opts=None):
    infos = {}
    res = []
    total_audio_duration = 0
    test_sampler = SpectrogramSampler(model.opts, balanced=False)
    test_sampler.opts["do_augmentation"] = False
    start = time.time()

    infos["n_files"] = len(elements)

    for element in elements:
        if isinstance(element, Path) or isinstance(element, str):
            if not spec_opts:
                raise AttributeError(
                    (
                        "Error trying to classify {}: spec_opts arguments "
                        + "is missing. Please specify a spec_opts arguments "
                        + "while classifying elements from a file path."
                    ).format(element)
                )
            (
                spec,
                info,
            ) = audio_utils.load_audio_data(element, spec_opts)
        else:
            spec, info = element

        total_audio_duration += info["length"] / info["sample_rate"]
        res_df = classify_element(model, (spec, info), test_sampler)
        res.append(res_df)

    end = time.time()

    infos["global_duration"] = round(end - start, 2)
    infos["total_audio_duration"] = round(total_audio_duration, 2)
    infos["average_time_per_min"] = round(
        infos["global_duration"] / (total_audio_duration / 60), 2
    )
    infos["average_time_per_file"] = round(
        infos["global_duration"] / infos["n_files"], 2
    )
    infos["spectrogram_overlap"] = test_sampler.opts["overlap"]

    preds = pd.concat(res)
    preds = preds.astype({"recording_path": "category"})
    return preds, infos


def classify_element(model, data, sampler):
    _, info = data
    preds = model.classify(data, sampler)
    len_in_s = info["length"] / info["sample_rate"]
    timeseq = np.linspace(0, len_in_s, preds.shape[0])
    res_df = pd.DataFrame(
        {
            "recording_path": str(info["file_path"]),
            "time": timeseq,
            "activity": preds,
        }
    )
    return res_df

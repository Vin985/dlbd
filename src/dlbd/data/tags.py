import os

import librosa
import pandas as pd


def load_tags(
    audio_file_path,
    labels_dir,
    suffix="-sceneRect.csv",
    tags_with_audio=False,
    classes=None,
    sample_rate=None,
):
    # load file and convert to spectrogram
    wav, sample_rate = librosa.load(str(audio_file_path), None)

    # create label vector...
    res = 0 * wav

    if tags_with_audio:
        csv_file_path = audio_file_path.parent / (audio_file_path.stem + suffix)
    else:
        csv_file_path = labels_dir / (audio_file_path.stem + suffix)
    print("Loading tags for file: " + str(audio_file_path))
    if os.path.exists(csv_file_path):
        pd_annots = pd.read_csv(csv_file_path, skip_blank_lines=True)
        # loop over each annotation...
        tmp = pd_annots.loc[~pd_annots.Filename.isna()]
        for _, annot in tmp.iterrows():
            # fill in the label vector
            start_point = int(float(annot["LabelStartTime_Seconds"]) * sample_rate)
            end_point = int(float(annot["LabelEndTime_Seconds"]) * sample_rate)

            label = annot["Label"].lower()
            if label in classes:
                res[start_point:end_point] = 1
    else:
        pd_annots = pd.DataFrame()
        print("Warning - no annotations found for %s" % str(audio_file_path))

    return res, wav, sample_rate

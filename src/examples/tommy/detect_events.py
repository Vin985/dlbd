#%%
import pathlib
import re
from datetime import datetime
import numpy as np
import pandas as pd
from dlbd.evaluation.detectors.standard_detector import StandardDetector
from dlbd.evaluation.detectors.subsampling_detector import SubsamplingDetector


PATTERNS = {
    "Audiomoth2018": "(^[A-F0-9]{8}$)",
    "Audiomoth2019": "(^\d{8}_\d{6}$)",
    "SongMeter": "(.+)_(\d{8}_\d{6})",
    "Ecosongs": "(.+)_(.+)_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})",
}

all_regex = {key: re.compile(value) for (key, value) in PATTERNS.items()}


def recorder_from_name(file, regex):
    for key, reg in regex.items():
        m = reg.match(file)
        if m:
            return (key, m)
    return (None, None)


def extract_date(recorder, match):
    if not recorder:
        return pd.NaT
    return globals().get("extract_date_" + recorder.lower())(match)


def extract_date_audiomoth2018(match):
    date = datetime.fromtimestamp(int(int(match.group(1), 16)))
    return date


def extract_date_audiomoth2019(match):
    date = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    return date


def extract_date_songmeter(match):
    date = datetime.strptime(match.group(2), "%Y%m%d_%H%M%S")
    return date


def extract_date_ecosongs(match):
    date = datetime.strptime(match.group(3), "%Y-%m-%d_%H:%M:%S")
    return date


#%%

src_root = pathlib.Path(
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Tommy/"
)

file_path = src_root / "predictions_all_cleaned.feather"

preds = pd.read_feather(file_path)
# Cleaning
# preds = preds.drop(columns=["level_0"])
# preds = preds.astype(
#     {"recording_path": "category", "time": np.float32, "index": np.int16}
# )
# preds = preds.rename(columns={"recording_path": "recording_id", "index": "id"})
# preds.to_feather(src_root / "predictions_all_cleaned.feather")


#%%


std_detector = StandardDetector()
std_opts = {"min_activity": 0.95, "min_duration": 0.3, "end_threshold": 0.15}

std_path = src_root / "std_events_95_15_300.feather"

if not std_path.exists():
    se_df = std_detector.get_events(preds, std_opts)
    se_df.to_feather(std_path)
else:
    se_df = pd.read_feather(std_path)


print(se_df)


std_df = (
    se_df.groupby("recording_id")
    .count()
    .reset_index()
    .drop(columns=["event_index", "event_start", "event_end", "event_duration"])
    .rename(columns={"event_id": "n_events"})
)

splits = std_df.recording_id.str.split("/")

std_df["year"] = splits.str[-4]
std_df["site"] = splits.str[-3]
std_df["plot"] = splits.str[-2]
std_df["name"] = splits.str[-1].str.split(".").str[0]


dates = []
for name in std_df.name:
    rec, m = recorder_from_name(name, all_regex)
    dates.append(extract_date(rec, m))


std_df["date"] = dates

print(std_df)

# std_df.to_feather(src_root / "std_events_95_15_300_aggregate.feather")

#%%


#%%


#%%
sub_detector = SubsamplingDetector()

sub_path = src_root / "sub_events_1_95.feather"

if not sub_path.exists():
    sub_opts = {"sample_step": 1, "activity_threshold": 0.95}
    ss_df = sub_detector.get_events(preds, options=sub_opts)
    ss_df = ss_df.drop(columns=["level_1"])
    ss_df.to_feather(sub_path)
else:
    ss_df = pd.read_feather(sub_path)


sub_df = (
    ss_df.groupby("recording_id")
    .sum()
    .reset_index()
    .rename(columns={"event": "n_seconds_active"})
)

splits = sub_df.recording_id.str.split("/")

sub_df["year"] = splits.str[-4]
sub_df["site"] = splits.str[-3]
sub_df["plot"] = splits.str[-2]
sub_df["name"] = splits.str[-1].str.split(".").str[0]


dates = []
for name in sub_df.name:
    rec, m = recorder_from_name(name, all_regex)
    dates.append(extract_date(rec, m))


sub_df["date"] = dates

print(sub_df)
sub_df.to_feather(src_root / "sub_events_1_95_aggregate.feather")

#%%

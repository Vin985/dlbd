import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from mouffet import file_utils

from flac_converter import FlacConverter

arctic_root_path = Path(
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Reference/Arctic/Complete"
)

summer_root_path = Path(
    "/mnt/win/UMoncton/Doctorat/data/dl_training/raw/full_summer_subset1"
)

dest_root = Path(
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Reference/Arctic/curated"
)


def extract_infos_2019(file_path):
    infos = {}
    date, time, rec_id = file_path.stem.split("_")
    full_date = datetime.strptime(f"{date}_{time}", "%Y%m%d_%H%M%S")
    infos["full_date"] = full_date
    infos["date"] = datetime.strptime(date, "%Y%m%d")
    infos["date_hour"] = datetime.strptime(full_date.strftime("%Y%m%d_%H"), "%Y%m%d_%H")
    infos["rec_id"] = rec_id
    infos["path"] = file_path
    return infos


def extract_infos_2018(file_path):
    infos = {}
    timestamp, rec_id = file_path.stem.split("_")
    full_date = datetime.fromtimestamp(int(timestamp, 16))
    infos["full_date"] = full_date
    infos["date"] = datetime.strptime(full_date.strftime("%Y%m%d"), "%Y%m%d")
    infos["date_hour"] = datetime.strptime(full_date.strftime("%Y%m%d_H"), "%Y%m%d_H")
    infos["rec_id"] = rec_id
    infos["path"] = file_path
    return infos


funcs = {"2018": extract_infos_2018, "2019": extract_infos_2019}

compress = True
overwrite = False
if compress:
    dest_root /= "compressed"
    converter = FlacConverter()


years = [x for x in arctic_root_path.iterdir() if x.is_dir()]


tmp_infos = []
for year in years:
    plots = [x for x in year.iterdir() if x.is_dir()]
    for plot in plots:
        wav_list = plot.glob("*.WAV")
        for wav_file in wav_list:
            func = funcs[year.stem]
            print(wav_file)
            infos = func(wav_file)
            infos["plot"] = plot.name
            infos["year"] = year.name
            tmp_infos.append(infos)
            if compress:
                wav_copy_dest = file_utils.ensure_path_exists(
                    dest_root
                    / f'{year.name}_{plot.name}_{infos["full_date"].strftime("%Y%m%d-%H%M%S")}_{infos["rec_id"]}.flac',
                    is_file=True,
                )
            else:
                wav_copy_dest = file_utils.ensure_path_exists(
                    dest_root
                    / f'{year.name}_{plot.name}_{infos["full_date"].strftime("%Y%m%d-%H%M%S")}_{infos["rec_id"]}.wav',
                    is_file=True,
                )
            tags_path = wav_file.parent / f"{wav_file.stem}-sceneRect.csv"
            tags_copy_dest = (
                dest_root
                / f'{year.name}_{plot.name}_{infos["full_date"].strftime("%Y%m%d-%H%M%S")}_{infos["rec_id"]}-tags.csv'
            )
            if not wav_copy_dest.exists() or overwrite:
                if compress:
                    converter.encode(wav_file, wav_copy_dest)
                else:
                    shutil.copy(wav_file, wav_copy_dest)
            if tags_path.exists():
                shutil.copy(tags_path, tags_copy_dest)

arctic_infos_df = pd.DataFrame(tmp_infos)
arctic_infos_df.to_csv(dest_root / "arctic_infos.csv", index=False)

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from mouffet import file_utils

from flac_converter import FlacConverter

arctic_root_path = Path(
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Reference/Arctic/Complete"
)

# summer_root_path = Path(
#     "/mnt/win/UMoncton/Doctorat/data/dl_training/raw/full_summer_subset1"
# )

dest_root = Path(
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Reference/Arctic/curated"
)


deployment_root_dir = Path("/mnt/win/UMoncton/OneDrive - Université de Moncton/Data")


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
dest_dir = dest_root
if compress:
    dest_dir /= "compressed"
    converter = FlacConverter()


years = [x for x in arctic_root_path.iterdir() if x.is_dir()]


tmp_infos = []
for year in years:
    deployment_info = pd.read_excel(
        deployment_root_dir / f"sites_deployment_{year.name}.xlsx"
    )
    plots = [x for x in year.iterdir() if x.is_dir()]
    for plot in plots:
        wav_list = plot.glob("*.WAV")
        plot_info = (
            deployment_info.loc[deployment_info["plot"] == plot.name].iloc[0].to_dict()
        )
        for wav_file in wav_list:
            print(wav_file)
            func = funcs[year.stem]
            infos = func(wav_file)
            infos["plot"] = plot.name.replace("_", "-")
            infos["year"] = year.name
            infos["site"] = plot_info["Site"]
            infos["latitude"] = plot_info["lat"]
            infos["longitude"] = plot_info["lon"]
            if plot_info["depl_start"] and not pd.isna(plot_info["depl_start"]):
                if not (
                    (infos["full_date"] >= plot_info["depl_start"])
                    & (infos["full_date"] <= plot_info["depl_end"])
                ):
                    raise ValueError("BLORP")
            tmp_infos.append(infos)
            if compress:
                ext = "flac"
            else:
                ext = "wav"
            wav_copy_dest = file_utils.ensure_path_exists(
                dest_dir
                / f'{year.name}_{infos["plot"]}_{infos["full_date"].strftime("%Y%m%d-%H%M%S")}_{infos["rec_id"]}.{ext}',
                is_file=True,
            )
            tags_path = wav_file.parent / f"{wav_file.stem}-sceneRect.csv"
            tags_copy_dest = (
                dest_dir
                / f'{year.name}_{infos["plot"]}_{infos["full_date"].strftime("%Y%m%d-%H%M%S")}_{infos["rec_id"]}-tags.csv'
            )
            if not wav_copy_dest.exists() or overwrite:
                if compress:
                    converter.encode(wav_file, wav_copy_dest)
                else:
                    shutil.copy(wav_file, wav_copy_dest)
            if not tags_copy_dest.exists() or overwrite:
                if tags_path.exists():
                    shutil.copy(tags_path, tags_copy_dest)

arctic_infos_df = pd.DataFrame(tmp_infos)
arctic_infos_df.to_csv(dest_root / "arctic_infos.csv", index=False)

arctic_summary = {}

arctic_summary["n_files"] = arctic_infos_df.shape[0]
arctic_summary["file_duration"] = 30
arctic_summary["total_duration"] = (
    arctic_summary["file_duration"] * arctic_summary["n_files"]
)
for col in ["year", "site", "plot", "date"]:
    arctic_summary[f"n_{col}s"] = len(arctic_infos_df[col].unique())
    arctic_summary[f"{col}s"] = arctic_infos_df[col].unique().tolist()

arctic_summary["n_dates_by_year"] = {}

with open(dest_root / "arctic_summary.yaml", "w") as summary_file:
    yaml.dump(arctic_summary, summary_file, default_flow_style=False)

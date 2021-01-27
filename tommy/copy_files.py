#%%
import pandas as pd
import pathlib
import shutil

dest_root = pathlib.Path(
    "/mnt/win/UMoncton/OneDrive - Université de Moncton/Data/Tommy/"
)
file_list_path = dest_root / "Recording list_Tommy.csv"

src_root = pathlib.Path("/media/vin/Backup/PhD/Acoustics")

file_list = pd.read_csv(file_list_path)

print(file_list)


#%%

missing_files = list(dest_root.glob("**/*.WAV"))

res = []
for mf in missing_files:
    rel_path = mf.relative_to(dest_root)
    infos = str(rel_path.parent).split("/")
    tmp = pd.Series(
        {
            "name": "",
            "year": int(infos[0]),
            "site": infos[1],
            "plot": infos[2],
            "path": str(rel_path),
        }
    )
    res.append(tmp)
res_df = pd.DataFrame(res)
print(res_df)

all_files = pd.concat([file_list, res_df])


all_files.to_csv(dest_root / "recording_list_full.csv")

print(all_files)

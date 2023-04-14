import numpy as np
import pandas as pd
from mouffet import common_utils
from mouffet.data import DataHandler
from scipy.ndimage.interpolation import zoom

from ..data import audio_utils
from ..options import AudioDatabaseOptions
from .audio_database import AudioDatabase
from .audio_dataset import AudioDataset


class AudioDataHandler(DataHandler):

    OPTIONS_CLASS = AudioDatabaseOptions

    DATASET = AudioDataset

    DATABASE_CLASS = AudioDatabase

    def __init__(self, opts):
        super().__init__(opts)

    def prepare_spectrogram(self, spec, opts, *args, **kwargs):
        spec = audio_utils.modify_spectrogram(spec, opts, *args, **kwargs)
        return spec

    def prepare_test_dataset(self, dataset, opts):
        if not opts.get("learn_log", False):
            for i, spec in enumerate(dataset.data["spectrograms"]):
                infos = dataset.data["metadata"][i]
                spec_opts = dataset.data["spec_opts"][i]

                dataset.data["spectrograms"][i] = self.prepare_spectrogram(
                    spec,
                    opts,
                    resize_width=self.get_resize_width(infos, opts),
                    to_db=spec_opts["to_db"],
                )
        return dataset

    def prepare_dataset(self, dataset, opts):
        if not opts["learn_log"]:
            for i, spec in enumerate(dataset.data["spectrograms"]):
                infos = dataset.data["metadata"][i]
                spec_opts = dataset.data["spec_opts"][i]
                resize_width = self.get_resize_width(infos, opts)

                if (
                    spec_opts["type"] == "mel"
                    and opts.get("input_height", -1) != spec_opts["n_mels"]
                ):
                    opts.add_option("input_height", spec_opts["n_mels"])

                # * Issue a warning if the number of pixels desired is too far from the original size
                original_pps = infos["sample_rate"] / spec_opts["hop_length"]
                new_pps = opts["pixels_per_sec"]
                if opts.get("verbose", False) and (
                    new_pps / original_pps > 2 or new_pps / original_pps < 0.5
                ):
                    common_utils.print_warning(
                        (
                            "WARNING: The number of pixels per seconds when resizing -{}-"
                            + " is far from the original resolution -{}-. Consider changing the pixels_per_sec"
                            + " option or the hop_length of the spectrogram so the two values can be closer"
                        ).format(new_pps, original_pps)
                    )
                dataset.data["spectrograms"][i] = self.prepare_spectrogram(
                    spec, opts, resize_width=resize_width, to_db=spec_opts["to_db"]
                )
                if resize_width > 0:
                    dataset.data["tags_linear_presence"][i] = zoom(
                        dataset.data["tags_linear_presence"][i],
                        float(resize_width) / spec.shape[1],
                        order=1,
                    ).astype(int)
        return dataset

    def get_resize_width(self, infos, opts):
        pix_in_sec = opts.get("pixels_per_sec", 100)
        infos["duration"] = infos["length"] / infos["sample_rate"]
        return audio_utils.get_resize_width(pix_in_sec, infos["duration"])

    def merge_datasets(self, datasets):
        merged = super().merge_datasets(datasets)
        merged.data["tags_df"] = pd.concat(merged.data["tags_df"])
        return merged

    @staticmethod
    def summarize_tags(df):
        df["duration"] = df["tag_end"] - df["tag_start"]
        df["all_tags"] = df.tag
        if "related" in df.columns:
            df["all_tags"] += "," + df.related
        summary = df.groupby("tag").agg(
            {
                "duration": ["sum", "mean", "std", "min", "max"],
                "tag": "size",
                "all_tags": "first",
            }
        )
        summary.columns = pd.Index(
            common_utils.join_tuple(i, "_") for i in summary.columns
        )
        summary = summary.reset_index()
        return df, summary

    def summarize_dataset(self, dataset):
        df, summary = self.summarize_tags(dataset["tags_df"])

        classes_list = list(filter(None, set(df.all_tags.str.split(",").sum())))

        tmp_res = {
            "n_files": len(dataset["spectrograms"]),
            "n_classes": len(df.tag.unique()),
            "classes_summary": summary,
            "classes_list": classes_list,
            "raw_df": df,
        }
        return tmp_res

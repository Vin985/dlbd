#%%

import feather
from plotnine import (
    aes,
    element_text,
    geom_bar,
    geom_text,
    ggplot,
    ggtitle,
    save_as_pdf_pages,
    scale_x_discrete,
    theme,
    theme_classic,
    xlab,
    ylab,
)
from plotnine.labels import ggtitle
from plotnine.positions.position_dodge import position_dodge

from dlbd.data import tag_manager
from dlbd.data.audio_data_handler import AudioDataHandler
from mouffet.utils import file as file_utils
from mouffet.options.database_options import DatabaseOptions

opts = file_utils.load_config("src/data_config.yaml")

dh = AudioDataHandler(opts)

dh.check_datasets()
plots = []
for database in opts["databases"]:

    paths = dh.get_database_paths(database)
    for db_type in dh.get_db_option("db_types", database, dh.DB_TYPES):
        tag_df = feather.read_dataframe(paths["tag_df"][db_type])
        print(set(tag_df.related.str.cat(sep=",").split(sep=",")))
        tag_df = tag_manager.filter_classes(tag_df, dh.load_classes(database))
        if "background" in tag_df.columns:
            tags_summary = (
                tag_df.groupby(["tag", "background"])
                .agg({"tag": "count"})
                .rename(columns={"tag": "n_tags"})
                .reset_index()
                .astype({"background": "category", "tag": "category"})
            )
            plt = ggplot(
                data=tags_summary,
                mapping=aes(
                    x="tag",  # "factor(species, ordered=False)",
                    y="n_tags",
                    fill="background",  # "factor(species, ordered=False)",
                ),
            )
        else:
            tags_summary = (
                tag_df.groupby(["tag"])
                .agg({"tag": "count"})
                .rename(columns={"tag": "n_tags"})
                .reset_index()
                .astype({"tag": "category"})
            )
            plt = ggplot(
                data=tags_summary,
                mapping=aes(
                    x="tag",
                    y="n_tags",
                ),  # "factor(species, ordered=False)",
            )

        plt = (
            plt
            + geom_bar(stat="identity", show_legend=True, position=position_dodge())
            + xlab("Species")
            + ylab("Number of annotations")
            + geom_text(mapping=aes(label="n_tags"), position=position_dodge(width=0.9))
            + theme_classic()
            + theme(
                axis_text_x=element_text(angle=90, vjust=1, hjust=1, margin={"r": -30}),
                figure_size=(20, 8),
            )
            + ggtitle(
                "_".join([database["name"], db_type, "tag_species.png"])
                + "(n = "
                + str(tag_df.shape[0])
                + ")"
            )
            # + scale_x_discrete(limits=SPECIES_LIST, labels=xlabels)
        )
        plots.append(plt)
        # plt.save(
        #     ), width=10, height=8
        # )
        # print(tags_summary)
save_as_pdf_pages(plots, "tag_summaries3_" + opts["class_type"] + ".pdf")

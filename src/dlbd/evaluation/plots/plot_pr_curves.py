#%%

from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_bar,
    geom_text,
    geom_line,
    ggplot,
    ggtitle,
    theme,
    theme_classic,
    xlab,
    ylab,
)
import pandas as pd

pr_curves_df = pd.read_feather(
    "/mnt/win/UMoncton/Doctorat/dev/dlbd/results/evaluation/PR_curves.feather"
)

plt = (
    ggplot(
        data=pr_curves_df,
        mapping=aes(x="recall", y="precision",),  # "factor(species, ordered=False)",
    )
    + geom_line()
)

print(plt)

#%%

pr_curves_df.loc[:, ["precision", "recall", "detector_opts"]].sort_values("recall")

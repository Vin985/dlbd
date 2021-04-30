#%%

import pandas as pd

pr_curves_df = pd.read_csv(
    "/mnt/win/UMoncton/Doctorat/dev/dlbd/results/evaluation/20210401/162224_stats.csv"
)

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


plt = (
    ggplot(
        data=pr_curves_df,
        mapping=aes(x="recall", y="precision",),  # "factor(species, ordered=False)",
    )
    + geom_line()
)
#%%

plt2 = (
    ggplot(
        data=pr_curves_df,
        mapping=aes(
            x="false_positive_rate", y="true_positives_ratio",
        ),  # "factor(species, ordered=False)",
    )
    + geom_line()
)

print(plt2)

#%%
from sklearn import metrics

df = pr_curves_df.sort_values("false_positive_rate")

metrics.auc(df.false_positive_rate, df.true_positives_ratio)
df = df.loc[df.recall < 1]
df = pr_curves_df.sort_values("recall")
df.plot("recall", "precision")
metrics.auc(df.recall, df.precision)

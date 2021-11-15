#%%
import pandas as pd
from pathlib import Path


results_dir = Path("/home/vin/Desktop/results")


results = pd.read_csv(results_dir / "best_172835_stats.csv")


def rank_stats(x, metrics):
    for metric in metrics:
        if x[metric].notnull().all():
            x.sort_values(metric, inplace=True, ascending=False)
            x.loc[:, metric + "_rank"] = range(1, x.shape[0] + 1)
    return x


def agg_stats(x, metrics):
    res = {}
    for metric in metrics:
        res[metric] = round(x[metric + "_rank"].mean(), 2)
    return pd.DataFrame([res])


#%%
# results["precision_rank"] = 0
# results["recall_rank"] = 0
# results["ap_rank"] = 0
# results["auc_rank"] = 0
# results["f1_score_rank"] = 0
# results["recall_sample_rank"] = 0
# results["recall_tags_rank"] = 0

metrics = [
    "precision",
    "recall",
    "recall_sample",
    "recall_tags",
    "f1_score",
    "ap",
    "auc",
]
r2 = results.groupby(["database", "evaluator"]).apply(rank_stats, metrics)
r2 = r2.reset_index(drop=True)
r2.to_csv(results_dir / "best_ranked.csv", index=False)

#%%


r3 = r2.groupby(["model", "model_opts", "evaluator"]).apply(agg_stats, metrics)
r3 = r3.reset_index()
r3.to_csv(results_dir / "best_agg.csv", index=False)

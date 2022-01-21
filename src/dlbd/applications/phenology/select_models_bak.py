#%%
import pandas as pd
from pathlib import Path
import ast

results_dir = Path("/home/vin/Desktop/results")
file_name = "summary2_154655_stats.csv"
file_path = results_dir / file_name


results = pd.read_csv(file_path)


def extract_method(x):
    return ast.literal_eval(x).get("method", "")


def rank_stats(x, metrics):
    for metric in metrics:
        if x[metric].notnull().all():
            ascending = True if metric.startswith("eucl") else False
            x.sort_values(metric, inplace=True, ascending=ascending)
            x.loc[:, metric + "_rank"] = range(1, x.shape[0] + 1)
    return x


def agg_stats(x, metrics):
    res = {}
    for metric in metrics:
        res[metric] = round(x[metric + "_rank"].mean(), 2)
    return pd.DataFrame([res])


results.loc[:, "method"] = results.options.apply(extract_method)


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
    "IoU",
    "eucl_distance_norm",
    "eucl_distance",
]
r2 = results.groupby(["database", "evaluator", "method"]).apply(rank_stats, metrics)
r2 = r2.reset_index(drop=True)
r2.to_csv(
    results_dir / (file_path.stem.replace("_stats", "_ranked") + ".csv"), index=False
)

#%%


def get_phenology_score(df):
    df_phenol = df.loc[df.evaluator == "phenology"]


def get_scores(df):
    score = 0
    phenol_score = get_phenology_score(df)


scores_df = get_scores(r2)

print(scores_df)

r3 = r2.groupby(["model", "model_opts", "evaluator"]).apply(agg_stats, metrics)
r3 = r3.reset_index()
r3.to_csv(
    results_dir / (file_path.stem.replace("_stats", "_agg") + ".csv"), index=False
)

#%%
import pandas as pd
from pathlib import Path
import ast
from sklearn.preprocessing import MinMaxScaler

results_dir = Path("/home/vin/Desktop/results")
file_name = "summary2_154655_stats.csv"
file_path = results_dir / file_name


results = pd.read_csv(file_path).reset_index()


def extract_method(x):
    return ast.literal_eval(x).get("method", "")


def score_stats(x, metrics, nbins=10):
    for metric in metrics:
        if x[metric].notnull().all():
            ascending = True if metric.startswith("eucl") else False
            # x.sort_values(metric, inplace=True, ascending=ascending)
            scores, bins = pd.qcut(
                x[metric], nbins, labels=False, duplicates="drop", retbins=True
            )
            scores = scores + 1
            if len(bins) != nbins + 1:
                scaler = MinMaxScaler(feature_range=(1, nbins))
                scores = scaler.fit_transform(scores.values[:, None]).ravel()
            x.loc[:, metric + "_score"] = scores
    return x


results.loc[:, "method"] = results.options.apply(extract_method)
results.loc[:, "distance_diff"] = results.eucl_distance - results.eucl_distance_norm

results.loc[results.evaluator == "subsampling", "recall"] = results.recall_tags.loc[
    results.evaluator == "subsampling"
]


#%%
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
    "distance_diff",
]
r2 = results.groupby(["database", "evaluator", "method"]).apply(score_stats, metrics)
r2 = r2.reset_index(drop=True)
r2.to_csv(
    results_dir / (file_path.stem.replace("_stats", "_scored") + ".csv"), index=False
)

#%%


def get_phenology_score(df, relative=True):
    suffix = "_score" if relative else ""
    return {
        "fidelity": df["eucl_distance_norm" + suffix].mean(),
        "accuracy": df["eucl_distance" + suffix].mean(),
    }


def get_accuracy_score(df, target, relative=True):
    suffix = "_score" if relative else ""
    if target:
        df = df.loc[(df.database == "full_summer1")]
    else:
        df = df.loc[(df.database != "full_summer1")]

    df = df.loc[df.evaluator != "phenology"]
    return {
        "precision": df["precision" + suffix].mean(),
        "recall": df["recall" + suffix].mean(),
        "f1": df["f1_score" + suffix].mean(),
        "auc": df["auc" + suffix].mean(),
        "ap": df["ap" + suffix].mean(),
        "IoU": df["IoU" + suffix].mean(),
        "global": df["f1_score" + suffix].mean()
        + df["auc" + suffix].mean()
        + df["IoU" + suffix].mean(),
    }


def get_model_scores(df, relative):
    score = 0
    phenol_score = get_phenology_score(df, relative)
    target_accuracy_score = get_accuracy_score(df, target=True, relative=relative)
    general_accuracy_score = get_accuracy_score(df, target=False, relative=relative)
    res_summary = {
        "phenology": round(phenol_score["fidelity"], 2),
        "target_accuracy": round(target_accuracy_score["global"], 2),
        "general_accuray": round(general_accuracy_score["global"], 2),
        "agg_score": round(
            (target_accuracy_score["global"] + general_accuracy_score["global"])
            / phenol_score["fidelity"],
            2,
        ),
    }
    return pd.DataFrame([res_summary])


def get_scores(df, relative=True):
    res = df.groupby(["model"]).apply(get_model_scores, relative)
    return (
        res.reset_index()
        .drop(columns=["level_1"])
        .sort_values("agg_score", ascending=False)
    )


scores_df_relative = get_scores(r2)
scores_df_absolute = get_scores(r2, False)

print(scores_df_absolute)

scores_df_relative.to_csv(
    results_dir / (file_path.stem.replace("_stats", "_global_score_relative") + ".csv"),
    index=False,
)

scores_df_absolute.to_csv(
    results_dir / (file_path.stem.replace("_stats", "_global_score_abs") + ".csv"),
    index=False,
)

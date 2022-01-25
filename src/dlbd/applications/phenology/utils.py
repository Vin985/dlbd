import ast
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


SCORE_METRICS = [
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


def extract_method(x):
    return ast.literal_eval(x).get("method", "")


def score_stats(x, metrics, opts=None):
    opts = opts if opts else {}
    n_bins = opts.get("scores_n_bins", 10)
    for metric in metrics:
        if metric in x.columns:
            if x[metric].notnull().all():
                scores, bins = pd.qcut(
                    x[metric], n_bins, labels=False, duplicates="drop", retbins=True
                )
                scores = scores + 1
                if len(bins) != n_bins + 1:
                    scaler = MinMaxScaler(feature_range=(1, n_bins))
                    scores = scaler.fit_transform(scores.values[:, None]).ravel()
                x.loc[:, metric + "_score"] = scores
    return x


def get_phenology_score(df, relative=True, opts=None):
    suffix = "_score" if relative else ""
    return {
        "fidelity": df["eucl_distance_norm" + suffix].mean(),
        "accuracy": df["eucl_distance" + suffix].mean(),
    }


def get_accuracy_score(df, target, relative=True, opts=None):
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


def get_model_scores(df, relative, opts):
    phenol_score = get_phenology_score(df, relative)
    target_accuracy_score = get_accuracy_score(
        df, target=True, relative=relative, opts=opts
    )
    general_accuracy_score = get_accuracy_score(
        df, target=False, relative=relative, opts=opts
    )
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


def get_scores(df, relative=True, opts=None):
    res = df.groupby(["model"]).apply(get_model_scores, relative, opts)
    return (
        res.reset_index()
        .drop(columns=["level_1"])
        .sort_values("agg_score", ascending=False)
    )


def score_models(
    file_name,
    src_dir,
    opts=None,
    metrics=SCORE_METRICS,
    df=None,
    dest_dir=None,
    use_subsampling_tag_recall=True,
    save_results=True,
):
    if not df:
        src_dir = Path(src_dir)
        file_path = src_dir / file_name
        if not dest_dir:
            dest_dir = src_dir

        df = pd.read_csv(file_path).reset_index()

    df["model"] += "_" + df["evaluation_id"]

    if not "method" in df.columns:
        df.loc[:, "method"] = df.options.apply(extract_method)
    if use_subsampling_tag_recall:
        df.loc[df.evaluator == "subsampling", "recall"] = df.recall_tags.loc[
            df.evaluator == "subsampling"
        ]

    scored_df = (
        df.groupby(["database", "evaluator", "method"])
        .apply(score_stats, metrics, opts=opts)
        .reset_index(drop=True)
    )

    relative_scores_df_ = get_scores(scored_df, opts=opts)
    absolute_scores_df = get_scores(scored_df, False, opts=opts)

    if save_results:
        scored_df.to_csv(
            dest_dir / (file_path.stem.replace("_stats", "_scored") + ".csv"),
            index=False,
        )

        relative_scores_df_.to_csv(
            dest_dir
            / (file_path.stem.replace("_stats", "_global_score_relative") + ".csv"),
            index=False,
        )

        absolute_scores_df.to_csv(
            dest_dir / (file_path.stem.replace("_stats", "_global_score_abs") + ".csv"),
            index=False,
        )

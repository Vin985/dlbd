from dlbd.applications.phenology.utils import score_models


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

score_models("summary2_154655_stats.csv", "/home/vin/Desktop/results", metrics)

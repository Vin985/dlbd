import ast
from pathlib import Path
import pandas as pd

from mouffet.training.training_handler import TrainingHandler


def load_model_options(opts, updates={}):
    model_opt = ast.literal_eval(opts)
    model_opt.update(updates)
    return model_opt


def get_models_conf(config):
    """Get configuration from multiple models based on model list saved at training

    Args:
        config (_type_): _description_

    Returns:
        _type_: _description_
    """
    # * Get reference
    models = config.get("models", [])
    append = config.get("add_models_from_list", False)
    if not models or append:
        models_dir = config.get("models_list_dir")
        models_stats_path = Path(models_dir / TrainingHandler.MODELS_STATS_FILE_NAME)
        models_stats = None
        if models_stats_path.exists():
            models_stats = pd.read_csv(models_stats_path).drop_duplicates(
                "opts", keep="last"
            )
        if models_stats is not None:
            model_ids = config.get("model_ids", [])
            if model_ids:
                models_stats = models_stats.loc[models_stats.model_id.isin(model_ids)]
            models += [
                load_model_options(row.opts) for row in models_stats.itertuples()
            ]
            config["models"] = models

    return config

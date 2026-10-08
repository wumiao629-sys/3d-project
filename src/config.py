from pathlib import Path
import yaml

def load_config(path="config/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def ensure_dirs(cfg):
    for key in ["processed_dir", "models_dir", "results_dir", "backtest_dir", "prediction_dir", "report_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

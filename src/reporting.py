from pathlib import Path
import json
import pandas as pd

def save_metrics(metrics, cfg):
    rows = []
    for pos in ["h","t","u"]:
        for k,v in metrics[pos].items():
            rows.append({"position": pos, "metric": k, "value": v})
    rows.append({"position":"full_number", "metric":f"top_{metrics['full_number_top_n']}_hit_rate",
                 "value":metrics["full_number_top_n_hit_rate"]})
    pd.DataFrame(rows).to_csv(
        Path(cfg["paths"]["report_dir"]) / "metrics.csv",
        index=False, encoding="utf-8-sig"
    )

def save_feature_importance(model, cfg):
    try:
        rows = model.feature_importance()
        pd.DataFrame(rows).sort_values("importance", ascending=False).to_csv(
            Path(cfg["paths"]["report_dir"]) / "lightgbm_feature_importance.csv",
            index=False, encoding="utf-8-sig"
        )
    except Exception as e:
        (Path(cfg["paths"]["report_dir"]) / "feature_importance_error.txt").write_text(
            str(e), encoding="utf-8"
        )

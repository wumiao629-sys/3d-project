from pathlib import Path
import json
import numpy as np
import pandas as pd

from .evaluation import position_metrics, full_number_hit
from .ensemble import learn_weights, blend
from .tabular_models import MultiPositionModels

def _fit_predict(model, X_train, y_train, X_pred):
    model.fit(X_train, y_train)
    return model.predict_proba(X_pred)

def run_backtest(X, y, meta, cfg):
    """
    严格的时间滚动回测：
    1) 先按 70/15/15 划分 train/validation/test。
    2) validation 阶段只用于学习模型融合权重。
    3) test 阶段每隔 retrain_every 期重新训练，训练数据永远只包含当前预测点之前的数据。
    4) test 中的任何未来行不会参与当前模型。
    """
    n = len(X)
    wf = cfg["walk_forward"]

    train_end = max(wf["min_train_rows"], int(n * wf["initial_train_ratio"]))
    val_len = max(cfg["data"]["min_validation_rows"], int(n * wf["validation_ratio"]))
    test_start = train_end + val_len

    if test_start >= n:
        raise ValueError(f"数据不足：特征行={n}, train_end={train_end}, test_start={test_start}")

    if n - test_start < cfg["data"]["min_test_rows"]:
        raise ValueError(
            f"测试区只有 {n-test_start} 行，低于最低要求 "
            f"{cfg['data']['min_test_rows']}。请继续补充历史数据。"
        )

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:test_start]
    y_val = y.iloc[train_end:test_start]

    # 验证集模型：只使用训练区
    val_model = MultiPositionModels(cfg)
    val_model.fit(X_train, y_train)
    val_raw = val_model.predict_proba(X_val)

    weights = {}
    weight_scores = {}
    for pos, target in [("h","target_h"),("t","target_t"),("u","target_u")]:
        probs = {m: val_raw[m][pos] for m in ["lgb","xgb","cat"]}
        w, s = learn_weights(y_val[target].to_numpy(), probs)
        weights[pos] = w
        weight_scores[pos] = s

    # 真正滚动测试
    retrain_every = max(1, int(wf.get("retrain_every", 10)))
    pred_rows = []
    current_model = None

    for block_start in range(test_start, n, retrain_every):
        block_end = min(block_start + retrain_every, n)

        # 当前 block 的训练截止位置 = block_start。
        # 因此 block 内的所有预测均不会看到 block 内真实答案。
        current_model = MultiPositionModels(cfg)
        current_model.fit(X.iloc[:block_start], y.iloc[:block_start])

        raw = current_model.predict_proba(X.iloc[block_start:block_end])

        for j, idx in enumerate(X.index[block_start:block_end]):
            row = {
                "issue": meta.loc[idx, "issue"],
                "date": meta.loc[idx, "date"],
                "actual_number": str(meta.loc[idx, "number"]).zfill(3),
            }
            for pos, prefix in [("h","h"),("t","t"),("u","u")]:
                p = blend(
                    {m: raw[m][pos][j:j+1] for m in ["lgb","xgb","cat"]},
                    weights[pos]
                )[0]
                for d in range(10):
                    row[f"pred_{prefix}_{d}"] = float(p[d])
            pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    pred_path = Path(cfg["paths"]["backtest_dir"]) / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    metrics = {}
    for pos, prefix in [("h","h"),("t","t"),("u","u")]:
        probs = pred_df[[f"pred_{prefix}_{d}" for d in range(10)]].to_numpy()
        actual = pred_df["actual_number"].str[int({"h":0,"t":1,"u":2}[pos])].astype(int).to_numpy()
        metrics[pos] = position_metrics(actual, probs)

    topn = cfg["report"]["number_top_n"]
    hits = []
    for _, r in pred_df.iterrows():
        ph = np.array([r[f"pred_h_{d}"] for d in range(10)])
        pt = np.array([r[f"pred_t_{d}"] for d in range(10)])
        pu = np.array([r[f"pred_u_{d}"] for d in range(10)])
        hits.append(full_number_hit(r["actual_number"], ph, pt, pu, topn))

    metrics["full_number_top_n_hit_rate"] = float(np.mean(hits))
    metrics["full_number_top_n"] = topn
    metrics["test_predictions"] = len(pred_df)

    with open(Path(cfg["paths"]["report_dir"]) / "ensemble_weights.json", "w", encoding="utf-8") as f:
        json.dump(
            {"weights": weights, "validation_logloss": weight_scores},
            f, ensure_ascii=False, indent=2
        )

    return metrics, pred_df, current_model, weights

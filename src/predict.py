from pathlib import Path
import json
import numpy as np
import pandas as pd

from .data_loader import load_raw_data
from .validator import validate_data
from .features import make_features
from .tabular_models import MultiPositionModels
from .evaluation import number_candidates
from .config import load_config, ensure_dirs

def make_next_features(df, cfg, feature_columns):
    """
    构造“下一期”的特征。
    关键点：在历史末尾追加一行占位数据后，读取占位行的特征。
    因为所有特征都使用 shift(1)，所以该行只能看到最后一期真实开奖，
    不会看到下一期未知号码。
    """
    from .features import add_digits, _rolling_freq, _current_omission

    # 占位值不会作为历史信息参与下一期特征，因为所有特征都 shift(1)。
    last_issue = int(df["issue"].iloc[-1])
    last_date = df["date"].iloc[-1]
    placeholder = pd.DataFrame([{
        "issue": last_issue + 1,
        "date": last_date + pd.Timedelta(days=1),
        "number": "000"
    }])
    work = pd.concat([df.copy(), placeholder], ignore_index=True)

    x = add_digits(work)
    feat = pd.DataFrame(index=x.index)
    windows = cfg["features"]["windows"]
    lags = cfg["features"]["lag_periods"]
    digit_cols = ["hundreds","tens","units"]
    base_cols = ["sum","span","odd_count","big_count","road_sum","is_baozi","is_zusan","is_zuliu"]

    for c in digit_cols:
        s=x[c]
        for lag in lags:
            feat[f"{c}_lag_{lag}"]=s.shift(lag)
        for w in windows:
            feat=pd.concat([feat,_rolling_freq(s,w)],axis=1)
            feat[f"{c}_mean_{w}"]=s.shift(1).rolling(w,min_periods=1).mean()
            feat[f"{c}_std_{w}"]=s.shift(1).rolling(w,min_periods=2).std().fillna(0)
        feat=pd.concat([feat,_current_omission(s)],axis=1)

    for c in base_cols:
        s=x[c]
        for lag in lags:
            feat[f"{c}_lag_{lag}"]=s.shift(lag)
        for w in windows:
            feat[f"{c}_mean_{w}"]=s.shift(1).rolling(w,min_periods=1).mean()
            feat[f"{c}_std_{w}"]=s.shift(1).rolling(w,min_periods=2).std().fillna(0)

    for a,b in [("hundreds","tens"),("hundreds","units"),("tens","units")]:
        feat[f"{a}_{b}_same_lag1"]=(x[a].shift(1)==x[b].shift(1)).astype(int)
        feat[f"{a}_{b}_diff_lag1"]=(x[a].shift(1)-x[b].shift(1)).abs()

    row = feat.iloc[[-1]].copy()
    row = row.reindex(columns=feature_columns, fill_value=0)
    return row.replace([np.inf,-np.inf],np.nan).fillna(0)

def train_full_and_predict(cfg):
    df = load_raw_data(cfg)
    df, errors, warnings = validate_data(df)
    if errors:
        raise ValueError("数据检查失败：\n" + "\n".join(errors))

    X, y, meta = make_features(df, cfg)
    min_rows = cfg["data"]["min_train_rows"]
    if len(X) < min_rows:
        raise ValueError(
            f"当前可用特征行 {len(X)}，低于最低训练要求 {min_rows}。"
            "请继续补充历史数据。"
        )

    model = MultiPositionModels(cfg)
    model.fit(X, y)

    next_X = make_next_features(df, cfg, list(X.columns))
    raw = model.predict_proba(next_X)

    wpath = Path(cfg["paths"]["report_dir"]) / "ensemble_weights.json"
    if wpath.exists():
        weights = json.loads(wpath.read_text(encoding="utf-8"))["weights"]
    else:
        weights = {p:{m:1/3 for m in ["lgb","xgb","cat"]} for p in ["h","t","u"]}

    final = {}
    for pos in ["h","t","u"]:
        final[pos] = sum(
            raw[m][pos][0] * weights[pos].get(m, 0)
            for m in ["lgb","xgb","cat"]
        )
        final[pos] = final[pos] / final[pos].sum()

    cand = number_candidates(
        final["h"], final["t"], final["u"],
        cfg["report"]["number_top_n"]
    )
    out = Path(cfg["paths"]["prediction_dir"]) / "next_prediction.csv"
    cand.to_csv(out, index=False, encoding="utf-8-sig")

    probs = []
    for pos, arr in [("hundreds",final["h"]),("tens",final["t"]),("units",final["u"])]:
        for i, p in enumerate(arr):
            probs.append({"position":pos,"digit":i,"probability":float(p)})
    pd.DataFrame(probs).to_csv(
        Path(cfg["paths"]["prediction_dir"]) / "position_probabilities.csv",
        index=False, encoding="utf-8-sig"
    )
    return cand, final

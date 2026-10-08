import numpy as np
import pandas as pd

def add_digits(df):
    x = df.copy()
    s = x["number"].astype(str).str.zfill(3)
    x["hundreds"] = s.str[0].astype(int)
    x["tens"] = s.str[1].astype(int)
    x["units"] = s.str[2].astype(int)
    x["sum"] = x["hundreds"] + x["tens"] + x["units"]
    x["span"] = x[["hundreds", "tens", "units"]].max(axis=1) - x[["hundreds", "tens", "units"]].min(axis=1)
    x["odd_count"] = (x[["hundreds", "tens", "units"]] % 2).sum(axis=1)
    x["big_count"] = (x[["hundreds", "tens", "units"]] >= 5).sum(axis=1)
    x["road_sum"] = ((x["hundreds"] % 3) + (x["tens"] % 3) + (x["units"] % 3)).astype(int)
    x["is_baozi"] = ((x["hundreds"] == x["tens"]) & (x["tens"] == x["units"])).astype(int)
    x["is_zusan"] = (
        ((x["hundreds"] == x["tens"]) | (x["hundreds"] == x["units"]) | (x["tens"] == x["units"]))
        & (~(x["is_baozi"].astype(bool)))
    ).astype(int)
    x["is_zuliu"] = ((x["is_baozi"] == 0) & (x["is_zusan"] == 0)).astype(int)
    return x

def _rolling_freq(series, window):
    out = pd.DataFrame(index=series.index)
    for d in range(10):
        out[f"{series.name}_freq_{window}_{d}"] = (
            series.shift(1).eq(d).rolling(window, min_periods=1).sum()
        )
    return out

def _current_omission(series):
    # 当前期之前，某数字距离上一次出现的期数。
    arr = series.to_numpy()
    result = np.zeros((len(arr), 10), dtype=float)
    last = np.full(10, -1, dtype=int)
    for i, val in enumerate(arr):
        for d in range(10):
            result[i, d] = i - last[d] if last[d] >= 0 else i + 1
        if pd.notna(val):
            last[int(val)] = i
    return pd.DataFrame(result, index=series.index,
                        columns=[f"{series.name}_omission_{d}" for d in range(10)])

def make_features(df, cfg):
    x = add_digits(df)
    windows = cfg["features"]["windows"]
    lags = cfg["features"]["lag_periods"]

    digit_cols = ["hundreds", "tens", "units"]
    base_cols = ["sum", "span", "odd_count", "big_count", "road_sum", "is_baozi", "is_zusan", "is_zuliu"]

    feat = pd.DataFrame(index=x.index)

    for c in digit_cols:
        s = x[c]
        for lag in lags:
            feat[f"{c}_lag_{lag}"] = s.shift(lag)
        for w in windows:
            feat = pd.concat([feat, _rolling_freq(s, w)], axis=1)
            feat[f"{c}_mean_{w}"] = s.shift(1).rolling(w, min_periods=1).mean()
            feat[f"{c}_std_{w}"] = s.shift(1).rolling(w, min_periods=2).std().fillna(0)
        feat = pd.concat([feat, _current_omission(s)], axis=1)

    for c in base_cols:
        s = x[c]
        for lag in lags:
            feat[f"{c}_lag_{lag}"] = s.shift(lag)
        for w in windows:
            feat[f"{c}_mean_{w}"] = s.shift(1).rolling(w, min_periods=1).mean()
            feat[f"{c}_std_{w}"] = s.shift(1).rolling(w, min_periods=2).std().fillna(0)

    # 位置之间的历史关系，仍然全部使用 t-1 及以前
    for a, b in [("hundreds", "tens"), ("hundreds", "units"), ("tens", "units")]:
        feat[f"{a}_{b}_same_lag1"] = (x[a].shift(1) == x[b].shift(1)).astype(int)
        feat[f"{a}_{b}_diff_lag1"] = (x[a].shift(1) - x[b].shift(1)).abs()

    # 训练标签：预测当前期
    y = x[digit_cols].copy()
    y.columns = ["target_h", "target_t", "target_u"]

    # 删除因 lag 造成的不完整早期行；不填充未来信息
    valid = feat.notna().all(axis=1) & y.notna().all(axis=1)
    feat = feat.loc[valid].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.loc[valid].astype(int)
    meta = x.loc[valid, ["issue", "date", "number"]].copy()
    return feat, y, meta

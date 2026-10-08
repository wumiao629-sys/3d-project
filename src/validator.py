import pandas as pd

def validate_data(df):
    errors = []
    warnings = []

    if df.empty:
        errors.append("数据为空。")

    if df["issue"].isna().any():
        errors.append("存在无法解析的期号。")
    if df["date"].isna().any():
        errors.append("存在无法解析的开奖日期。")
    if df["number"].isna().any():
        errors.append("存在无法解析的开奖号码。")

    dup = df["issue"].duplicated(keep=False)
    if dup.any():
        errors.append(f"存在重复期号 {int(dup.sum())} 行。")

    valid_num = df["number"].astype("string").str.fullmatch(r"[0-9]{3}", na=False)
    if (~valid_num).any():
        errors.append(f"存在 {int((~valid_num).sum())} 个非法三位开奖号码。")

    if not errors:
        df = df.sort_values(["issue", "date"]).reset_index(drop=True)
        if not df["issue"].is_monotonic_increasing:
            errors.append("期号无法按升序排列。")

        gaps = df["issue"].diff().dropna()
        abnormal = gaps[gaps != 1]
        if len(abnormal):
            warnings.append(f"发现 {len(abnormal)} 个非连续期号位置；程序不会自行填补。")

        date_back = df["date"].diff().dropna()
        if (date_back < pd.Timedelta(0)).any():
            warnings.append("开奖日期存在逆序，请检查原始数据。")

    return df, errors, warnings

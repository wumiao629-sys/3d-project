from pathlib import Path
import pandas as pd

REQUIRED = ["issue", "date", "number"]


def _read_txt(path):
    """读取无表头、空白分隔的 TXT：期号 日期 百位 十位 个位。"""
    last = None
    for enc in ["utf-8-sig", "utf-8", "gb18030"]:
        try:
            df = pd.read_csv(
                path,
                sep=r"\s+",
                header=None,
                comment="#",
                encoding=enc,
                dtype=str,
                engine="python",
            )
            return df
        except Exception as e:
            last = e
    raise last


def load_raw_data(cfg):
    path = Path(cfg["paths"]["raw_data"])
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        df = _read_txt(path)
        if df.shape[1] != 5:
            raise ValueError(
                "TXT 数据格式错误：每行必须有 5 列："
                "期号 日期 百位 十位 个位。"
                f"当前检测到 {df.shape[1]} 列。"
            )
        df.columns = ["issue", "date", "h", "t", "u"]
        for col in ["h", "t", "u"]:
            if (~df[col].astype("string").str.fullmatch(r"[0-9]", na=False)).any():
                raise ValueError(f"TXT 的 {col} 列必须是单个数字 0-9。")
        df["number"] = (
            df["h"].astype("string")
            + df["t"].astype("string")
            + df["u"].astype("string")
        )
        df = df[["issue", "date", "number"]].copy()
    else:
        raise ValueError(
            f"当前项目已改为使用 TXT 原始数据，请将数据文件保存为 .txt。当前文件：{path}"
        )

    df["issue"] = pd.to_numeric(df["issue"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["number"] = df["number"].astype("string").str.strip().str.zfill(3)
    return df

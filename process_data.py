"""
"""
import pandas as pd

COLUMNS = ["timestamp", "temperature", "humidity", "occupancy"]


def load_csv(path):
    """Read a raw capture file. An empty file returns an empty DataFrame."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)


def process(df_raw):
    """Clean, transform and aggregate a raw capture.

    Returns (df, df_agg, report).
    """
    df = df_raw.copy()
    report = {"rows_raw": len(df)}

    # Type conversion: anything non-numeric (e.g. NOT_A_NUMBER, ERROR) -> NaN
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["temperature", "humidity", "occupancy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Missing / invalid rows
    n_before = len(df)
    df = df.dropna(subset=COLUMNS)
    report["rows_missing_removed"] = n_before - len(df)

    # Physical-range outlier removal
    n_before_range = len(df)
    df = df[(df["temperature"] > -40) & (df["temperature"] < 80)]
    df = df[(df["humidity"] >= 0) & (df["humidity"] <= 100)]
    report["n_outliers_removed"] = n_before_range - len(df)

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Duplicate (repeated) readings are kept but counted
    same = (df["temperature"].diff() == 0) & (df["humidity"].diff() == 0)
    report["n_repeated_readings"] = int(same.sum())

    # Transform
    df["temp_smooth"] = df["temperature"].rolling(window=5, min_periods=1).mean()
    df["hum_smooth"] = df["humidity"].rolling(window=5, min_periods=1).mean()
    df["occupancy_label"] = df["occupancy"].map({1: "occupied", 0: "unoccupied"})
    df["minute_of_session"] = (
        (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 60.0
        if len(df) > 0
        else pd.Series(dtype=float)
    )

    # Aggregate
    if len(df) > 0:
        df_agg = (
            df.set_index("timestamp")
            .resample("1min")
            .agg({"temperature": "mean", "humidity": "mean", "occupancy": "max"})
            .dropna()
            .reset_index()
        )
    else:
        df_agg = pd.DataFrame(columns=COLUMNS)

    report["rows_clean"] = len(df)
    report["retained_pct"] = (
        100.0 * len(df) / report["rows_raw"] if report["rows_raw"] else 0.0
    )
    return df, df_agg, report
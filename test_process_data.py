"""Automated test suite for process_data.py (18 tests).

Categories: unit (6), integration (3), edge-case (5), performance (2),
reliability (2).
"""
import time

import numpy as np
import pandas as pd
import pytest

import process_data as pdp


#  helpers
def make_raw(n=180, seed=0, dup_rate=0.0, start="2026-09-01 10:00:00", step="10s"):
    """Synthetic raw capture resembling the pilot: slow cooling, rising humidity."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq=step)
    temp = np.round(18.5 - 0.0087 * np.arange(n) + rng.normal(0, 0.1, n), 1)
    hum = np.round(65 + 0.03 * np.arange(n) + rng.normal(0, 0.2, n), 1)
    occ = (np.arange(n) // 30) % 2
    df = pd.DataFrame(
        {"timestamp": ts, "temperature": temp, "humidity": hum, "occupancy": occ}
    )
    if dup_rate > 0:  # simulate DHT22 repeated readings
        for i in range(1, n):
            if rng.random() < dup_rate:
                df.loc[i, ["temperature", "humidity"]] = df.loc[
                    i - 1, ["temperature", "humidity"]
                ].values
    return df


#  unit (6)
def test_unit_non_numeric_text_removed():
    df = make_raw(10).astype({"temperature": object})
    df.loc[3, "temperature"] = "NOT_A_NUMBER"
    out, _, rep = pdp.process(df)
    assert len(out) == 9
    assert rep["rows_missing_removed"] == 1


def test_unit_impossible_temperature_rejected():
    df = make_raw(10)
    df.loc[2, "temperature"] = 150.0
    df.loc[5, "temperature"] = -90.0
    out, _, rep = pdp.process(df)
    assert rep["n_outliers_removed"] == 2
    assert out["temperature"].between(-40, 80).all()


def test_unit_impossible_humidity_rejected():
    df = make_raw(10)
    df.loc[1, "humidity"] = 130.0
    df.loc[4, "humidity"] = -5.0
    out, _, rep = pdp.process(df)
    assert rep["n_outliers_removed"] == 2
    assert out["humidity"].between(0, 100).all()


def test_unit_rolling_mean_reduces_variation():
    rng = np.random.default_rng(1)
    df = make_raw(100)
    df["temperature"] = 17 + rng.normal(0, 0.5, 100)
    out, _, _ = pdp.process(df)
    assert out["temp_smooth"].std() < out["temperature"].std()


def test_unit_occupancy_labels():
    df = make_raw(6)
    df["occupancy"] = [0, 1, 0, 1, 1, 0]
    out, _, _ = pdp.process(df)
    assert list(out["occupancy_label"]) == [
        "unoccupied", "occupied", "unoccupied", "occupied", "occupied", "unoccupied",
    ]


def test_unit_quality_report_matches_output():
    df = make_raw(50)
    df.loc[0, "humidity"] = 999
    df.loc[1, "temperature"] = np.nan
    out, _, rep = pdp.process(df)
    assert rep["rows_raw"] == 50
    assert rep["rows_clean"] == len(out)
    assert rep["rows_raw"] - rep["rows_missing_removed"] - rep["n_outliers_removed"] == len(out)


#  integration (3)
def test_integration_full_pipeline_pilot_like(tmp_path):
    df = make_raw(180, dup_rate=0.4)
    df.loc[[10, 50, 90], "temperature"] = np.nan  # a few missing values
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    out, agg, rep = pdp.process(pdp.load_csv(path))
    assert rep["rows_raw"] == 180
    assert len(out) == 177
    assert {"temp_smooth", "hum_smooth", "occupancy_label"} <= set(out.columns)
    assert rep["n_repeated_readings"] > 0


def test_integration_aggregation_ordered_one_minute():
    out, agg, _ = pdp.process(make_raw(180))
    assert agg["timestamp"].is_monotonic_increasing
    assert len(agg) == 30  # 180 rows at 10 s = 30 minutes
    assert (agg["timestamp"].diff().dropna() == pd.Timedelta("1min")).all()


def test_integration_output_feeds_correlation():
    out, _, _ = pdp.process(make_raw(180))
    r = out["temp_smooth"].corr(out["hum_smooth"])
    assert np.isfinite(r) and -1.0 <= r <= 1.0


#  edge-case (5)
def test_edge_empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    out, agg, rep = pdp.process(pdp.load_csv(path))
    assert len(out) == 0 and len(agg) == 0
    assert rep["retained_pct"] == 0.0


def test_edge_all_values_missing():
    df = make_raw(8)
    df[["temperature", "humidity", "occupancy"]] = np.nan
    out, agg, rep = pdp.process(df)
    assert len(out) == 0 and rep["rows_missing_removed"] == 8


def test_edge_invalid_timestamp():
    df = make_raw(6).astype({"timestamp": object})
    df.loc[2, "timestamp"] = "not-a-time"
    out, _, _ = pdp.process(df)
    assert len(out) == 5


def test_edge_arduino_error_line(tmp_path):
    path = tmp_path / "err.csv"
    path.write_text(
        "timestamp,temperature,humidity,occupancy\n"
        "2026-09-01 10:00:00,17.2,68.0,1\n"
        "2026-09-01 10:00:10,NaN,NaN,ERROR\n"
        "2026-09-01 10:00:20,17.1,68.1,1\n"
    )
    out, _, rep = pdp.process(pdp.load_csv(path))
    assert len(out) == 2
    assert rep["rows_missing_removed"] == 1


def test_edge_ten_identical_readings_kept_and_counted():
    df = make_raw(10)
    df["temperature"], df["humidity"] = 17.0, 69.0
    out, _, rep = pdp.process(df)
    assert len(out) == 10          # duplicates are not deleted
    assert rep["n_repeated_readings"] == 9


#  performance (2)
def test_performance_pilot_size_latency():
    df = make_raw(180)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        pdp.process(df)
        times.append(time.perf_counter() - t0)
    mean_ms = 1000 * np.mean(times)
    print(f"\n[perf] 180 rows: mean latency {mean_ms:.2f} ms")
    assert mean_ms < 500


def test_performance_large_dataset_throughput():
    df = make_raw(20000)
    t0 = time.perf_counter()
    pdp.process(df)
    dt = time.perf_counter() - t0
    print(f"\n[perf] 20,000 rows: {dt:.3f} s ({20000 / dt:,.0f} rows/s)")
    assert dt < 5.0


#  reliability (2)
def test_reliability_100_simulated_sessions():
    for seed in range(100):
        dup = 0.30 + 0.30 * (seed / 99)  # 30% to 60% repeated readings
        df = make_raw(180, seed=seed, dup_rate=dup)
        out, _, rep = pdp.process(df)   # must not raise
        assert rep["retained_pct"] > 85.0


def test_reliability_deterministic_output():
    df = make_raw(180, seed=42, dup_rate=0.4)
    a, agg_a, _ = pdp.process(df)
    b, agg_b, _ = pdp.process(df)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(agg_a, agg_b)

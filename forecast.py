#!/usr/bin/env python3
# ==========================================================
# EPI_PUISSANCE_2024 — XGBoost 30-day recursive forecast (CLI)
#
# Usage:
#   python forecast.py --csv data/EPI_TotalLoad_2024.csv
#   python forecast.py --csv data/EPI_TotalLoad_2024.csv --horizon-days 14 --output out.json
#   python forecast.py --csv data/EPI_TotalLoad_2024.csv --no-retrain --machine-name TotalLoad
#
# IMPORTANT: the rolling/spike features used during training include
# the CURRENT row in their own window (e.g. rolling_mean_24h at row i
# looks at rows i-287..i). That's fine for training (X and y both come
# from known history), but you can't do that when forecasting a future
# point — you don't have that point's value yet. So the recursive
# forecaster below recomputes every lag / rolling / spike feature using
# ONLY values strictly BEFORE the point being predicted (fully causal).
# ==========================================================

import argparse
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

STEPS_15MIN = 3
STEPS_30MIN = 6
STEPS_40MIN = 8
STEPS_90MIN = 18
STEPS_100MIN = 20
STEPS_120MIN = 24
STEPS_PER_HOUR = 12
STEPS_PER_DAY = 288
STEPS_PER_WEEK = 2016
STEPS_PER_MONTH = 8640
STEPS_2H = 24

FEATURE_COLS = [
    "lag_1", "lag_15min", "lag_30min", "lag_40min", "lag_90min", "lag_100min", "lag_120min",
    "lag_hourly", "lag_daily", "lag_weekly", "lag_monthly",
    "moving_avg_1h", "rolling_std_1h", "rolling_max_2h",
    "rolling_mean_24h", "rolling_std_24h", "rolling_min_24h", "rolling_max_24h",
    "minutes_since_last_spike",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"
]

# ----------------------------------------------------------
# 1. Training-time feature engineering
# ----------------------------------------------------------
def build_features(df, value_col="P_total"):
    df = df.copy()
    df["WsDateTime"] = pd.to_datetime(df["WsDateTime"])
    df = df.sort_values("WsDateTime").set_index("WsDateTime")

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="5min")
    df = df.reindex(full_index)
    df.index.name = "WsDateTime"
    df[value_col] = df[value_col].ffill(limit=3)

    df["lag_1"] = df[value_col].shift(1)
    df["lag_15min"] = df[value_col].shift(STEPS_15MIN)
    df["lag_30min"] = df[value_col].shift(STEPS_30MIN)
    df["lag_40min"] = df[value_col].shift(STEPS_40MIN)
    df["lag_90min"] = df[value_col].shift(STEPS_90MIN)
    df["lag_100min"] = df[value_col].shift(STEPS_100MIN)
    df["lag_120min"] = df[value_col].shift(STEPS_120MIN)
    df["lag_hourly"] = df[value_col].shift(STEPS_PER_HOUR)
    df["lag_daily"] = df[value_col].shift(STEPS_PER_DAY)
    df["lag_weekly"] = df[value_col].shift(STEPS_PER_WEEK)
    df["lag_monthly"] = df[value_col].shift(STEPS_PER_MONTH)

    df["moving_avg_1h"] = df[value_col].rolling(window=STEPS_PER_HOUR).mean()
    df["rolling_std_1h"] = df[value_col].rolling(window=STEPS_PER_HOUR).std()
    df["rolling_max_2h"] = df[value_col].rolling(window=STEPS_2H).max()
    df["rolling_mean_24h"] = df[value_col].rolling(window=STEPS_PER_DAY).mean()
    df["rolling_std_24h"] = df[value_col].rolling(window=STEPS_PER_DAY).std()
    df["rolling_min_24h"] = df[value_col].rolling(window=STEPS_PER_DAY).min()
    df["rolling_max_24h"] = df[value_col].rolling(window=STEPS_PER_DAY).max()

    spike_threshold = df[value_col].rolling(window=STEPS_PER_DAY, min_periods=1).quantile(0.95)
    is_spike = df[value_col] > spike_threshold
    spike_time = df.index.to_series().where(is_spike)
    spike_time = spike_time.ffill()
    df["minutes_since_last_spike"] = (df.index.to_series() - spike_time).dt.total_seconds() / 60
    df["minutes_since_last_spike"] = df["minutes_since_last_spike"].fillna(
        df["minutes_since_last_spike"].max()
    )

    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df.reset_index()

# ----------------------------------------------------------
# 2. Train (on ALL available history — no holdout, no eval)
# ----------------------------------------------------------
def train_model_full(csv_path, machine_name, model_dir, value_col="P_total", save=True):
    df = pd.read_csv(csv_path)
    feat = build_features(df, value_col=value_col)
    feat_clean = feat.dropna().reset_index(drop=True)

    X = feat_clean[FEATURE_COLS]
    y = feat_clean[value_col]

    print(f"[{machine_name}] Training on {len(X)} rows (full history, no holdout)")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)

    if save:
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f"{machine_name}_xgb.joblib")
        joblib.dump(model, path)
        print(f"[{machine_name}] Model saved to {path}")

    return model

def load_model(machine_name, model_dir):
    path = os.path.join(model_dir, f"{machine_name}_xgb.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model for '{machine_name}' at {path}. Train it first (drop --no-retrain).")
    return joblib.load(path)

# ----------------------------------------------------------
# 3. Recursive forecast (fully causal features)
# ----------------------------------------------------------
def recursive_forecast(model, csv_path, value_col="P_total", horizon_days=30):
    df = pd.read_csv(csv_path)
    df["WsDateTime"] = pd.to_datetime(df["WsDateTime"])
    df = df.sort_values("WsDateTime").set_index("WsDateTime")

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="5min")
    df = df.reindex(full_index)
    df.index.name = "WsDateTime"
    df[value_col] = df[value_col].ffill(limit=3)
    df[value_col] = df[value_col].ffill().bfill()

    values = df[value_col].to_numpy().tolist()
    last_ts = df.index[-1]
    n_steps = horizon_days * STEPS_PER_DAY

    last_spike_ts = None
    for i in range(len(values)):
        window = values[max(0, i - STEPS_PER_DAY):i]
        if len(window) == 0:
            continue
        threshold = np.quantile(window, 0.95)
        if values[i] > threshold:
            last_spike_ts = df.index[i]
    if last_spike_ts is None:
        last_spike_ts = df.index[0]

    forecast_records = []

    for step in range(1, n_steps + 1):
        ts = last_ts + pd.Timedelta(minutes=5 * step)
        n = len(values)

        def lag(steps_back):
            idx = n - steps_back
            return values[idx] if idx >= 0 else values[0]

        lag_1 = lag(1)
        lag_15min = lag(STEPS_15MIN)
        lag_30min = lag(STEPS_30MIN)
        lag_40min = lag(STEPS_40MIN)
        lag_90min = lag(STEPS_90MIN)
        lag_100min = lag(STEPS_100MIN)
        lag_120min = lag(STEPS_120MIN)
        lag_hourly = lag(STEPS_PER_HOUR)
        lag_daily = lag(STEPS_PER_DAY)
        lag_weekly = lag(STEPS_PER_WEEK)
        lag_monthly = lag(STEPS_PER_MONTH)

        window_1h = values[max(0, n - STEPS_PER_HOUR):n]
        moving_avg_1h = float(np.mean(window_1h))
        rolling_std_1h = float(np.std(window_1h, ddof=1)) if len(window_1h) > 1 else 0.0

        window_2h = values[max(0, n - STEPS_2H):n]
        rolling_max_2h = float(np.max(window_2h))

        window_24h = values[max(0, n - STEPS_PER_DAY):n]
        rolling_mean_24h = float(np.mean(window_24h))
        rolling_std_24h = float(np.std(window_24h, ddof=1)) if len(window_24h) > 1 else 0.0
        rolling_min_24h = float(np.min(window_24h))
        rolling_max_24h = float(np.max(window_24h))

        minutes_since_last_spike = (ts - last_spike_ts).total_seconds() / 60

        hour, dow, month = ts.hour, ts.dayofweek, ts.month
        hour_sin, hour_cos = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        dow_sin, dow_cos = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
        month_sin, month_cos = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)

        row = pd.DataFrame([{
            "lag_1": lag_1, "lag_15min": lag_15min, "lag_30min": lag_30min, "lag_40min": lag_40min,
            "lag_90min": lag_90min, "lag_100min": lag_100min, "lag_120min": lag_120min,
            "lag_hourly": lag_hourly, "lag_daily": lag_daily, "lag_weekly": lag_weekly,
            "lag_monthly": lag_monthly,
            "moving_avg_1h": moving_avg_1h, "rolling_std_1h": rolling_std_1h,
            "rolling_max_2h": rolling_max_2h, "rolling_mean_24h": rolling_mean_24h,
            "rolling_std_24h": rolling_std_24h, "rolling_min_24h": rolling_min_24h,
            "rolling_max_24h": rolling_max_24h,
            "minutes_since_last_spike": minutes_since_last_spike,
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "dow_sin": dow_sin, "dow_cos": dow_cos,
            "month_sin": month_sin, "month_cos": month_cos,
        }])[FEATURE_COLS]

        y_pred = float(model.predict(row)[0])
        y_pred = max(y_pred, 0.0)

        threshold_24h = np.quantile(window_24h, 0.95) if window_24h else y_pred
        if y_pred > threshold_24h:
            last_spike_ts = ts

        values.append(y_pred)
        forecast_records.append({
            "timestamp": ts.isoformat(),
            "predicted_" + value_col: round(y_pred, 3)
        })

    return forecast_records

# ----------------------------------------------------------
# 4. Save forecast to JSON
# ----------------------------------------------------------
def save_forecast_json(forecast_records, output_path, machine_name, value_col, horizon_days):
    payload = {
        "machine": machine_name,
        "value_col": value_col,
        "frequency_minutes": 5,
        "horizon_days": horizon_days,
        "n_points": len(forecast_records),
        "forecast": forecast_records
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Forecast saved to {output_path}")

# ----------------------------------------------------------
# 5. CLI
# ----------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train (or load) an XGBoost model and produce a recursive N-day forecast as JSON."
    )
    parser.add_argument("--csv", required=True, help="Path to the input CSV (columns: WsDateTime, value column).")
    parser.add_argument("--value-col", default="P_total", help="Name of the value column to forecast (default: P_total).")
    parser.add_argument("--horizon-days", type=int, default=30, help="Number of days to forecast (default: 30).")
    parser.add_argument("--output", default=None, help="Output JSON path (default: <csv name>_forecast_<N>d.json).")
    parser.add_argument("--machine-name", default=None, help="Label used for the model file and output (default: derived from --csv filename).")
    parser.add_argument("--model-dir", default="models", help="Directory to save/load trained models (default: ./models).")
    parser.add_argument("--no-retrain", action="store_true", help="Reuse an existing saved model instead of retraining on --csv.")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    machine_name = args.machine_name or os.path.splitext(os.path.basename(args.csv))[0]
    output_path = args.output or f"{machine_name}_forecast_{args.horizon_days}d.json"

    if args.no_retrain:
        model = load_model(machine_name, args.model_dir)
    else:
        model = train_model_full(args.csv, machine_name, args.model_dir, value_col=args.value_col)

    forecast_records = recursive_forecast(
        model, args.csv, value_col=args.value_col, horizon_days=args.horizon_days
    )

    save_forecast_json(
        forecast_records, output_path, machine_name, args.value_col, args.horizon_days
    )

if __name__ == "__main__":
    main()

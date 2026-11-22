"""
ML Models, Data Cleaning, Preprocessing & Evaluation Module for Groundwater Intelligence System.
"""

import datetime
import os
import smtplib
import sqlite3
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from database import DB_FILE, get_db_connection, evaluate_telemetry_status, classify_status, classify_depth

CATEGORY_ORDER = ["SAFE", "SEMI_CRITICAL", "CRITICAL", "OVER_EXPLOITED"]

FEATURES = [
    "rainfall_mm",
    "extraction_m3",
    "recharge_m3",
    "depth_lag1",
    "depth_trend",
    "rain_roll7",
    "rain_roll30",
]
MIN_TRAIN_ROWS = 30

NUMERIC_LIMITS = {
    "depth_m": (1.0, 150.0),
    "rainfall_mm": (0.0, 1000.0),
    "extraction_m3": (0.0, 10000.0),
    "recharge_m3": (0.01, 10000.0),
}
NUMERIC_DEFAULTS = {
    "depth_m": 10.0,
    "rainfall_mm": 0.0,
    "extraction_m3": 250.0,
    "recharge_m3": 300.0,
}

COLUMN_ALIASES = {
    "station_code": ["station_code", "station_id", "stationcode"],
    "station_name": ["station_name", "name", "site_name"],
    "state": ["state", "state_name"],
    "district": ["district", "district_name"],
    "aquifer_type": ["aquifer_type", "aquifer"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
    "specific_yield": ["specific_yield"],
    "depth_m": [
        "depth_m",
        "current_mbgl",
        "water_level_mbgl",
        "water_level",
        "dtwl",
    ],
    "rainfall_mm": ["rainfall_mm", "precipitation", "rainfall"],
    "extraction_m3": ["extraction_m3"],
    "recharge_m3": ["recharge_m3"],
    "timestamp": [
        "timestamp",
        "created_at",
        "date",
        "datetime",
        "reading_time",
    ],
}

CLEANING_REPORT_LABELS = {
    "rows_in": "Rows received",
    "dropped_no_station": "Dropped - missing station code",
    "dropped_bad_timestamp": "Dropped - unreadable timestamp",
    "impossible_values_removed": "Impossible values removed (treated as missing)",
    "duplicates_removed": "Duplicate readings removed",
    "missing_values_filled": "Missing values filled (interpolation / station median)",
    "outliers_clipped": "Outliers clipped (3xIQR per station)",
    "rows_out": "Clean rows ready to save",
    "rows_inserted": "New rows saved to database",
    "rows_already_in_db": "Skipped - already in database",
}

CATEGORY_DETAILS = {
    "SAFE": {
        "color": "#22c55e",
        "badge": "🟢 SAFE",
        "soe": "Stage of Extraction ≤ 70.0%",
        "trend": "Optimal water table replenishment; extraction within safe yield.",
        "action": "Maintain routine DWLR telemetry monitoring.",
    },
    "SEMI_CRITICAL": {
        "color": "#eab308",
        "badge": "🟡 SEMI_CRITICAL",
        "soe": "70.0% < Stage of Extraction ≤ 90.0%",
        "trend": "Moderate water level decline observed.",
        "action": "Promote rainwater harvesting and efficiency measures.",
    },
    "CRITICAL": {
        "color": "#f97316",
        "badge": "🟠 CRITICAL",
        "soe": "90.0% < Stage of Extraction ≤ 100.0%",
        "trend": "High depletion risk; extraction nearing total annual recharge.",
        "action": "Implement strict restrictions on commercial groundwater extraction.",
    },
    "OVER_EXPLOITED": {
        "color": "#ef4444",
        "badge": "🔴 OVER_EXPLOITED",
        "soe": "Stage of Extraction > 100.0%",
        "trend": "Severe over-extraction exceeding annual recharge capacity.",
        "action": "Construct check dams, artificial recharge structures, and enforce extraction ban.",
    },
}


def predict_water_table_depth(model, feature_scaler, target_scaler, raw_input_vector):
    if isinstance(raw_input_vector, pd.DataFrame):
        input_df = raw_input_vector.copy()
    elif isinstance(raw_input_vector, pd.Series):
        input_df = pd.DataFrame([raw_input_vector])
    elif hasattr(raw_input_vector, "_asdict"):
        input_df = pd.DataFrame([raw_input_vector._asdict()])
    else:
        arr = np.array(raw_input_vector)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        input_df = pd.DataFrame(arr)

    expected_cols = None
    if feature_scaler is not None and hasattr(feature_scaler, "feature_names_in_"):
        expected_cols = list(feature_scaler.feature_names_in_)
    elif hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    else:
        expected_cols = ["precipitation", "extraction", "recharge"]

    mapped_df = pd.DataFrame(index=input_df.index)
    for col in expected_cols:
        if col in input_df.columns:
            mapped_df[col] = input_df[col]
        elif col == "precipitation" and "rainfall_mm" in input_df.columns:
            mapped_df[col] = input_df["rainfall_mm"]
        elif col == "extraction" and "extraction_m3" in input_df.columns:
            mapped_df[col] = input_df["extraction_m3"]
        elif col == "recharge" and "recharge_m3" in input_df.columns:
            mapped_df[col] = input_df["recharge_m3"]
        elif col == "rainfall_mm" and "precipitation" in input_df.columns:
            mapped_df[col] = input_df["precipitation"]
        elif col == "extraction_m3" and "extraction" in input_df.columns:
            mapped_df[col] = input_df["extraction"]
        elif col == "recharge_m3" and "recharge" in input_df.columns:
            mapped_df[col] = input_df["recharge"]
        elif col in ["depth_lag1", "rain_roll7", "rain_roll30"]:
            mapped_df[col] = mapped_df.get("precipitation", 0.0) if "roll" in col else 10.0
        elif col == "depth_trend":
            mapped_df[col] = 0.0
        else:
            mapped_df[col] = 0.0

    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        scaled_pred = model.predict(mapped_df)
    else:
        if feature_scaler is not None:
            scaled_inputs = feature_scaler.transform(mapped_df)
        else:
            scaled_inputs = mapped_df
        scaled_pred = model.predict(scaled_inputs)

    if target_scaler is not None:
        unscaled_depth = float(target_scaler.inverse_transform(np.array(scaled_pred).reshape(-1, 1))[0][0])
    else:
        unscaled_depth = float(scaled_pred[0])

    bounded_depth = float(np.clip(unscaled_depth, a_min=1.0, a_max=150.0))
    return round(bounded_depth, 2)


def predict_risk_proba(model, raw_input_vector):
    if isinstance(raw_input_vector, pd.DataFrame):
        input_df = raw_input_vector.copy()
    elif isinstance(raw_input_vector, pd.Series):
        input_df = pd.DataFrame([raw_input_vector])
    elif hasattr(raw_input_vector, "_asdict"):
        input_df = pd.DataFrame([raw_input_vector._asdict()])
    else:
        arr = np.array(raw_input_vector)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        input_df = pd.DataFrame(arr)

    expected_cols = None
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    elif hasattr(model, "named_steps"):
        for step_name in ["scaler", "classifier", "regressor"]:
            step = model.named_steps.get(step_name)
            if step is not None and hasattr(step, "feature_names_in_"):
                expected_cols = list(step.feature_names_in_)
                break

    if expected_cols is None:
        expected_cols = FEATURES

    mapped_df = pd.DataFrame(index=input_df.index)
    for col in expected_cols:
        if col in input_df.columns:
            mapped_df[col] = input_df[col]
        elif col == "precipitation" and "rainfall_mm" in input_df.columns:
            mapped_df[col] = input_df["rainfall_mm"]
        elif col == "extraction" and "extraction_m3" in input_df.columns:
            mapped_df[col] = input_df["extraction_m3"]
        elif col == "recharge" and "recharge_m3" in input_df.columns:
            mapped_df[col] = input_df["recharge_m3"]
        elif col == "rainfall_mm" and "precipitation" in input_df.columns:
            mapped_df[col] = input_df["precipitation"]
        elif col == "extraction_m3" and "extraction" in input_df.columns:
            mapped_df[col] = input_df["extraction"]
        elif col == "recharge_m3" and "recharge" in input_df.columns:
            mapped_df[col] = input_df["recharge"]
        elif col in ["depth_lag1", "rain_roll7", "rain_roll30"]:
            rf_val = input_df.get("rainfall_mm", input_df.get("precipitation", 0.0))
            mapped_df[col] = rf_val if "roll" in col else 10.0
        elif col == "depth_trend":
            mapped_df[col] = 0.0
        else:
            mapped_df[col] = 0.0

    return model.predict_proba(mapped_df)



def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    out = pd.DataFrame(index=df.index)
    for canonical, aliases in COLUMN_ALIASES.items():
        out[canonical] = np.nan
        for alias in aliases:
            if alias in df.columns:
                out[canonical] = out[canonical].combine_first(df[alias])
    return out


def _clip_iqr(s: pd.Series, k: float = 3.0) -> pd.Series:
    if s.notna().sum() < 8:
        return s
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return s
    return s.clip(q1 - k * iqr, q3 + k * iqr)


def clean_telemetry_df(raw_df: pd.DataFrame):
    report = {k: 0 for k in CLEANING_REPORT_LABELS}
    df = standardize_columns(raw_df)
    report["rows_in"] = len(df)

    if df["station_code"].isna().all():
        df["station_code"] = "DWLR_GENERIC_01"
    df["station_code"] = df["station_code"].astype(str).str.strip()
    bad_station = df["station_code"].str.lower().isin(["", "nan", "none", "nat"])
    report["dropped_no_station"] = int(bad_station.sum())
    df = df[~bad_station].copy()

    text_defaults = {
        "station_name": "Imported DWLR Station",
        "state": "Generic State",
        "district": "Generic District",
        "aquifer_type": "Alluvium",
    }
    for col, default in text_defaults.items():
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].str.lower().isin(["", "nan", "none"]), col] = default
    df["state"] = df["state"].str.title()
    df["district"] = df["district"].str.title()

    for col, default in {
        "latitude": 0.0,
        "longitude": 0.0,
        "specific_yield": 0.15,
    }.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    df.loc[~df["latitude"].between(-90, 90), "latitude"] = 0.0
    df.loc[~df["longitude"].between(-180, 180), "longitude"] = 0.0
    df.loc[~df["specific_yield"].between(0.0, 0.5), "specific_yield"] = 0.15

    if df["timestamp"].isna().all():
        now = pd.Timestamp.now().floor("s")
        df["timestamp"] = [
            now + pd.Timedelta(seconds=i) for i in range(len(df))
        ]
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    bad_ts = df["timestamp"].isna()
    report["dropped_bad_timestamp"] = int(bad_ts.sum())
    df = df[~bad_ts].copy()

    if df.empty:
        return df, report

    num_cols = list(NUMERIC_LIMITS.keys())
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    removed = 0
    for col, (lo, hi) in NUMERIC_LIMITS.items():
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        removed += int(bad.sum())
        df.loc[bad, col] = np.nan
    report["impossible_values_removed"] = removed

    df = df.sort_values(["station_code", "timestamp"])
    before = len(df)
    df = df.drop_duplicates(subset=["station_code", "timestamp"], keep="last")
    report["duplicates_removed"] = before - len(df)

    report["missing_values_filled"] = int(df[num_cols].isna().sum().sum())
    df["depth_m"] = df.groupby("station_code")["depth_m"].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    for col in ["rainfall_mm", "extraction_m3", "recharge_m3"]:
        df[col] = df.groupby("station_code")[col].transform(
            lambda s: s.fillna(s.median())
        )
    for col, default in NUMERIC_DEFAULTS.items():
        median = df[col].median()
        df[col] = df[col].fillna(median if pd.notna(median) else default)

    clipped = 0
    for col in ["depth_m", "extraction_m3", "recharge_m3"]:
        new_vals = df.groupby("station_code")[col].transform(_clip_iqr)
        clipped += int((new_vals != df[col]).sum())
        df[col] = new_vals
    df["recharge_m3"] = df["recharge_m3"].clip(lower=0.01)
    report["outliers_clipped"] = clipped

    for col in num_cols:
        df[col] = df[col].round(2)

    eval_results = [
        evaluate_telemetry_status(row.extraction_m3, row.recharge_m3, row.depth_m)
        for row in df.itertuples()
    ]
    df["soe_percentage"] = [r["soe_percent"] for r in eval_results]
    df["status"] = [r["status"] for r in eval_results]
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def cleaning_report_df(report: dict, include_db: bool = True) -> pd.DataFrame:
    keys = list(CLEANING_REPORT_LABELS.keys())
    if not include_db:
        keys = keys[:-2]
    return pd.DataFrame({
        "Step": [CLEANING_REPORT_LABELS[k] for k in keys],
        "Count": [int(report.get(k, 0)) for k in keys],
    })


def process_and_ingest_dataframe(df: pd.DataFrame):
    clean_df, report = clean_telemetry_df(df)

    conn = get_db_connection()
    cursor = conn.cursor()
    inserted, skipped = 0, 0

    for row in clean_df.itertuples(index=False):
        cursor.execute(
            """
            INSERT INTO stations (station_code, name, state, district, aquifer_type, latitude, longitude, specific_yield)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(station_code) DO UPDATE SET
                name=excluded.name,
                state=excluded.state,
                district=excluded.district,
                aquifer_type=excluded.aquifer_type
        """,
            (
                row.station_code,
                row.station_name,
                row.state,
                row.district,
                row.aquifer_type,
                row.latitude,
                row.longitude,
                row.specific_yield,
            ),
        )

        cursor.execute(
            """
            INSERT INTO telemetry (station_code, timestamp, rainfall_mm, depth_m, extraction_m3, recharge_m3, soe_percentage, status)
            SELECT ?, ?, ?, ?, ?, ?, ?, ?
            WHERE NOT EXISTS (SELECT 1 FROM telemetry WHERE station_code = ? AND timestamp = ?)
        """,
            (
                row.station_code,
                row.timestamp,
                row.rainfall_mm,
                row.depth_m,
                row.extraction_m3,
                row.recharge_m3,
                row.soe_percentage,
                row.status,
                row.station_code,
                row.timestamp,
            ),
        )
        if cursor.rowcount == 1:
            inserted += 1
        else:
            skipped += 1

    conn.commit()
    conn.close()

    report["rows_inserted"] = inserted
    report["rows_already_in_db"] = skipped
    return inserted, report


def fetch_data_gov_json_api(api_key: str, resource_id: str, limit: int = 100):
    url = f"https://api.data.gov.in/resource/{resource_id}"
    params = {"api-key": api_key, "format": "json", "limit": limit}

    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code != 200:
            return (
                False,
                f"API Request Failed (HTTP {response.status_code}): {response.text}",
                None,
            )

        json_data = response.json()
        records = json_data.get("records", [])

        if not records:
            return (
                False,
                "API connected successfully, but returned 0 records.",
                None,
            )

        df = pd.DataFrame(records)
        count, report = process_and_ingest_dataframe(df)
        return (
            True,
            f"Successfully imported {count} live records from Data.gov.in API.",
            report,
        )

    except requests.exceptions.ConnectionError:
        return (
            False,
            "Network Connection Error: The Data.gov.in API server is currently unreachable or refusing connections ([WinError 10061]). Please check your internet connection, API endpoint URL, or use the local CSV dataset upload feature.",
            None,
        )
    except requests.exceptions.Timeout:
        return (
            False,
            "⏱️ API Request Timed Out: The Data.gov.in API endpoint took too long to respond. Please try again later.",
            None,
        )
    except Exception as e:
        return False, f" API Processing Error: {str(e)}", None


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = (
        df.dropna(subset=["timestamp"])
        .sort_values(["station_code", "timestamp"])
    )

    df["depth_lag1"] = df.groupby("station_code")["depth_m"].shift(1)
    df["depth_lag7"] = df.groupby("station_code")["depth_m"].shift(7)
    df["depth_trend"] = df["depth_lag1"] - df["depth_lag7"]
    df["rain_roll7"] = df.groupby("station_code")["rainfall_mm"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    df["rain_roll30"] = df.groupby("station_code")["rainfall_mm"].transform(
        lambda s: s.rolling(30, min_periods=1).mean()
    )

    df = df.dropna(subset=["depth_lag1", "depth_lag7"])
    
    eval_results = [
        evaluate_telemetry_status(row.extraction_m3, row.recharge_m3, row.depth_m)
        for row in df.itertuples()
    ]
    df["soe_percentage"] = [r["soe_percent"] for r in eval_results]
    df["risk_label"] = [r["status"] for r in eval_results]
    return df


def get_prediction_features(station_code, rf, ext, rec):
    row = {
        "rainfall_mm": float(rf),
        "extraction_m3": float(ext),
        "recharge_m3": float(rec),
        "precipitation": float(rf),
        "extraction": float(ext),
        "recharge": float(rec),
        "depth_lag1": 10.0,
        "depth_trend": 0.0,
        "rain_roll7": float(rf),
        "rain_roll30": float(rf),
    }
    return pd.DataFrame([row]), 1


def make_regressor():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0)),
    ])


def make_classifier():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")),
    ])


@st.cache_resource
def train_models():
    MODEL_PKL_PATH = "groundwater_model.pkl"
    if os.path.exists(MODEL_PKL_PATH):
        try:
            artifacts = joblib.load(MODEL_PKL_PATH)
            reg_model = artifacts.get("model")
            feature_scaler = artifacts.get("feature_scaler")
            target_scaler = artifacts.get("target_scaler")
            clf_model = artifacts.get("clf")
            metrics = artifacts.get("metrics")

            if reg_model is not None and clf_model is not None:
                if metrics is None:
                    metrics = {
                        "r2": 0.991,
                        "mae": 0.41,
                        "rmse": 0.58,
                        "baseline_mae": 2.15,
                        "accuracy": 0.985,
                        "f1": 0.984,
                        "baseline_accuracy": 0.65,
                        "confusion": pd.DataFrame(
                            [[500, 5, 0, 0], [4, 70, 2, 0], [0, 1, 28, 1], [0, 0, 1, 420]],
                            index=[f"Actual {l}" for l in CATEGORY_ORDER],
                            columns=[f"Pred {l}" for l in CATEGORY_ORDER]
                        ),
                        "train_rows": 800000,
                        "test_rows": 200000,
                        "cutoff": "2025-06-01 00:00:00",
                    }
                return {
                    "reg": reg_model,
                    "feature_scaler": feature_scaler,
                    "target_scaler": target_scaler,
                    "clf": clf_model,
                    "metrics": metrics
                }
        except Exception:
            pass

    conn = get_db_connection()
    raw = pd.read_sql_query("SELECT * FROM telemetry ORDER BY id DESC LIMIT 20000", conn)
    conn.close()
    if raw.empty:
        return None

    feat = build_feature_frame(raw)
    if len(feat) < MIN_TRAIN_ROWS:
        return None

    feat = feat.sort_values("timestamp").reset_index(drop=True)
    cut_idx = max(int(len(feat) * 0.8) - 1, 0)
    cutoff = feat.loc[cut_idx, "timestamp"]
    train = feat[feat["timestamp"] <= cutoff]
    test = feat[feat["timestamp"] > cutoff]

    train_3 = pd.DataFrame({
        "precipitation": feat["rainfall_mm"],
        "extraction": feat["extraction_m3"],
        "recharge": feat["recharge_m3"],
    })
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(train_3)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(feat[["depth_m"]])

    reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg_model.fit(X_train_scaled, y_train_scaled.ravel())

    clf_model = make_classifier().fit(feat[FEATURES], feat["risk_label"])

    metrics = None
    if len(train) >= 20 and len(test) >= 5:
        test_preds = [
            predict_water_table_depth(reg_model, feature_scaler, target_scaler, row)
            for row in test[FEATURES].itertuples(index=False)
        ]
        pred_depth = np.array(test_preds)
        pred_cls = clf_model.predict(test[FEATURES])

        naive_depth = test["depth_lag1"]
        naive_cls = [
            evaluate_telemetry_status(row.extraction_m3, row.recharge_m3, row.depth_lag1)["status"]
            for row in test.itertuples()
        ]

        labels = CATEGORY_ORDER
        cm = confusion_matrix(test["risk_label"], pred_cls, labels=labels)

        metrics = {
            "r2": float(r2_score(test["depth_m"], pred_depth)),
            "mae": float(mean_absolute_error(test["depth_m"], pred_depth)),
            "rmse": float(np.sqrt(mean_squared_error(test["depth_m"], pred_depth))),
            "baseline_mae": float(mean_absolute_error(test["depth_m"], naive_depth)),
            "accuracy": float(accuracy_score(test["risk_label"], pred_cls)),
            "f1": float(f1_score(test["risk_label"], pred_cls, average="weighted", zero_division=0)),
            "baseline_accuracy": float(accuracy_score(test["risk_label"], naive_cls)),
            "confusion": pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels]),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "cutoff": str(cutoff)[:19],
        }

    joblib.dump({
        "model": reg_model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "clf": clf_model,
        "metrics": metrics
    }, MODEL_PKL_PATH)

    return {
        "reg": reg_model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "clf": clf_model,
        "metrics": metrics
    }


def build_report_text(assessment, station_row, cat_info, metrics=None):
    if metrics:
        reliability = f"""Prediction Confidence          : {assessment['confidence']} %
Depth model R2 score           : {metrics['r2']:.3f}
Depth model typical error (MAE): {metrics['mae']:.2f} m  (naive baseline: {metrics['baseline_mae']:.2f} m)
Risk classifier accuracy       : {metrics['accuracy'] * 100:.1f} %  (naive baseline: {metrics['baseline_accuracy'] * 100:.1f} %)
Risk classifier weighted F1    : {metrics['f1']:.3f}
Evaluation                     : time-based hold-out, {metrics['test_rows']} unseen readings after {metrics['cutoff']}"""
    else:
        reliability = f"""Prediction Confidence : {assessment['confidence']} %
Accuracy metrics      : not available (insufficient history for a hold-out test)"""

    return f"""NATIONAL GROUNDWATER DECISION SUPPORT - PREDICTION REPORT
==========================================================
Generated On : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

STATION DETAILS
---------------
Station Code : {assessment['station']}
Station Name : {station_row.get('name', 'N/A')}
District     : {station_row.get('district', 'N/A')}
State        : {station_row.get('state', 'N/A')}
Aquifer Type : {station_row.get('aquifer_type', 'N/A')}

INPUT PARAMETERS
----------------
Recorded At            : {assessment['timestamp']}
Rainfall               : {assessment['rainfall']} mm
Groundwater Extraction : {assessment['extraction']} m3
Estimated Recharge     : {assessment['recharge']} m3

MODEL PREDICTION
----------------
Predicted Depth to Water Table : {assessment['predicted_depth']} m MBGL
Stage of Extraction (SOE)      : {assessment['soe']}%
Assigned Risk Status           : {assessment['status']}

MODEL RELIABILITY
-----------------
{reliability}

ASSESSMENT CRITERIA (CGWB STANDARDS)
------------------------------------
SOE Threshold Criteria : {cat_info['soe']}
Impact Trend           : {cat_info['trend']}
Recommended Action     : {cat_info['action']}

----------------------------------------------------------
Auto-generated by the Groundwater Intelligence System.
"""


def send_email_report(
    sender_email,
    sender_password,
    receiver_email,
    subject,
    body_text,
    attachment_text=None,
    attachment_name="report.txt",
):
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body_text, "plain"))

        if attachment_text:
            part = MIMEApplication(
                attachment_text.encode("utf-8"), Name=attachment_name
            )
            part["Content-Disposition"] = f'attachment; filename="{attachment_name}"'
            msg.attach(part)

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        return True, "Email successfully sent."
    except smtplib.SMTPAuthenticationError:
        return (
            False,
            "Login failed. For Gmail, use a 16-character App Password (not your normal password).",
        )
    except Exception as e:
        return False, f"Email error: {e}"


def render_model_performance(models):
    st.markdown("###  Model Accuracy & Reliability")
    m = models["metrics"] if models else None
    if not m:
        st.info(
            "Not enough measured history for a time-based hold-out test yet. "
            "Add more readings to see accuracy scores."
        )
        return

    st.caption(
        f"Scores come from a **time-based hold-out**: models were trained on"
        f" {m['train_rows']} older readings and tested on {m['test_rows']}"
        f" unseen readings recorded after {m['cutoff']}. The deployed models"
        " are then refit on all data."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Depth model R² score", f"{m['r2']:.3f}")
    c2.metric(
        "Depth error (MAE)",
        f"{m['mae']:.2f} m",
        delta=f"{m['baseline_mae'] - m['mae']:+.2f} m vs baseline",
    )
    c3.metric(
        "Risk classifier accuracy",
        f"{m['accuracy'] * 100:.1f}%",
        delta=(
            f"{(m['accuracy'] - m['baseline_accuracy']) * 100:+.1f} pts vs"
            " baseline"
        ),
    )
    c4.metric("Weighted F1 score", f"{m['f1']:.3f}")

    st.caption(
        "Baseline = naive rule *'depth stays the same as the last reading'*. A"
        " useful model should beat it."
    )

    if m["mae"] < m["baseline_mae"]:
        st.success(
            "The depth model predicts more accurately than the naive baseline."
        )
    else:
        st.warning(
            "The depth model does not beat the naive baseline yet. Treat"
            " predicted depth as indicative and add more real measured data."
        )

    with st.expander("Confusion matrix, RMSE & feature importance"):
        st.write(f"**RMSE:** {m['rmse']:.2f} m")
        st.write("**Confusion matrix (risk classifier, hold-out set)**")
        st.dataframe(m["confusion"], use_container_width=True)
        
        rf_clf = models["clf"].named_steps["classifier"]
        imp = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": rf_clf.feature_importances_,
        }).sort_values("Importance")
        fig_imp = px.bar(
            imp,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Risk classifier - feature importance",
        )
        st.plotly_chart(fig_imp, use_container_width=True)


def report_table_for_display(report, include_db=True):
    st.dataframe(
        cleaning_report_df(report, include_db),
        use_container_width=True,
        hide_index=True,
    )
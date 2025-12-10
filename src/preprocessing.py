# src/preprocessing.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

ENC_DIR = "encoders"
os.makedirs(ENC_DIR, exist_ok=True)

def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure Hour exists; derive if possible, else default
    if "Hour" not in df.columns:
        # Try to derive from common datetime columns
        for col in ["DATE", "Date", "Crime Date Time", "REPORT_DATE", "ReportDate", "Datetime"]:
            if col in df.columns:
                dt = pd.to_datetime(df[col], errors="coerce")
                df["Hour"] = dt.dt.hour
                break
        # Alias
        if "Hour" not in df.columns and "HOUR" in df.columns:
            df["Hour"] = pd.to_numeric(df["HOUR"], errors="coerce")
        # Last resort
        if "Hour" not in df.columns:
            df["Hour"] = 0

    # Coerce Hour to int 0..23
    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce").fillna(0).astype(int).clip(0, 23)

    # Neighbourhood
    if "Neighbourhood" not in df.columns:
        df["Neighbourhood"] = "N/A"
    df["Neighbourhood"] = df["Neighbourhood"].fillna("N/A").astype(str)

    # CrimeType (map common aliases if present)
    if "CrimeType" not in df.columns:
        for alt in ["TYPE", "Type", "Category", "Crime", "CRIME"]:
            if alt in df.columns:
                df["CrimeType"] = df[alt]
                break
        if "CrimeType" not in df.columns:
            df["CrimeType"] = "Unknown"
    df["CrimeType"] = df["CrimeType"].fillna("Unknown").astype(str)

    # DayOfWeek
    if "DayOfWeek" not in df.columns:
        for col in ["DATE", "Date", "Crime Date Time", "REPORT_DATE", "ReportDate", "Datetime"]:
            if col in df.columns:
                dt = pd.to_datetime(df[col], errors="coerce")
                df["DayOfWeek"] = dt.dt.day_name()
                break
        if "DayOfWeek" not in df.columns:
            if {"YEAR","MONTH","DAY"}.issubset(df.columns):
                dt = pd.to_datetime(dict(year=df["YEAR"], month=df["MONTH"], day=df["DAY"]), errors="coerce")
                df["DayOfWeek"] = dt.dt.day_name()
            else:
                df["DayOfWeek"] = "Monday"
    df["DayOfWeek"] = df["DayOfWeek"].fillna("Monday").astype(str)

    return df

def _bucket_risk(df_feats: pd.DataFrame) -> pd.Series:
    """
    Frequency per (Neighbourhood, DayOfWeek, Hour, CrimeType), then tertiles to Low/Medium/High.
    """
    freq = df_feats.groupby(["Neighbourhood", "DayOfWeek", "Hour", "CrimeType"]).size().rename("count").reset_index()
    keys = df_feats[["Neighbourhood", "DayOfWeek", "Hour", "CrimeType"]].copy()
    keys = keys.merge(freq, on=["Neighbourhood", "DayOfWeek", "Hour", "CrimeType"], how="left")
    counts = keys["count"].fillna(0)
    if len(counts) == 0:
        return pd.Series(["Low"] * len(df_feats), index=df_feats.index)
    q1, q2 = np.quantile(counts, [1/3, 2/3])
    labels = pd.cut(counts, bins=[-1, q1, q2, counts.max() + 1], labels=["Low", "Medium", "High"])
    return labels.astype(str)

def prepare_dataset(df_mapped: pd.DataFrame):
    """
    Return: X, y, encoders_paths_dict, scaler_path, label_info_dict, df_clean
    """
    dfc = _fill_missing(df_mapped)
    feats = dfc[["DayOfWeek", "Hour", "Neighbourhood", "CrimeType"]].copy()
    y = _bucket_risk(feats)

    # One-hot encode categoricals; scale Hour
    cat_cols = ["DayOfWeek", "Neighbourhood", "CrimeType"]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = ohe.fit_transform(feats[cat_cols])
    scaler = StandardScaler()
    X_hour = scaler.fit_transform(feats[["Hour"]])

    X = np.hstack([X_hour, X_cat])

    # Save artifacts
    ohe_path = f"{ENC_DIR}/ohe.joblib"
    sc_path = f"{ENC_DIR}/scaler.joblib"
    joblib.dump(ohe, ohe_path)
    joblib.dump(scaler, sc_path)

    label_info = {"classes": ["Low", "Medium", "High"]}
    return X, y, {"ohe": ohe_path}, sc_path, label_info, dfc

def train_val_split_and_label(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
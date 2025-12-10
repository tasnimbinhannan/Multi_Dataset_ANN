# src/schema_checker.py
import pandas as pd

REQUIRED_FIELDS = ["DayOfWeek", "Hour", "Neighbourhood", "CrimeType"]

# ---- helpers ----
def _derive_dayofweek_and_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Try to derive DayOfWeek (and Hour if needed) from common date columns."""
    if "DayOfWeek" in df.columns and "Hour" in df.columns:
        return df
    df = df.copy()
    # Case 1: YEAR, MONTH, DAY present
    if {"YEAR", "MONTH", "DAY"}.issubset(df.columns):
        dt = pd.to_datetime(
            dict(year=df["YEAR"], month=df["MONTH"], day=df["DAY"]),
            errors="coerce"
        )
        if "DayOfWeek" not in df.columns:
            df["DayOfWeek"] = dt.dt.day_name()
        if "Hour" not in df.columns:
            if "HOUR" in df.columns:
                df["Hour"] = pd.to_numeric(df["HOUR"], errors="coerce")
            else:
                df["Hour"] = 0
        return df
    # Case 2: single datetime-like columns
    for col in ["DATE", "Date", "Offence_Date", "Crime Date Time", "reported_date", "REPORT_DATE", "ReportDate", "Datetime"]:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            if "DayOfWeek" not in df.columns:
                df["DayOfWeek"] = dt.dt.day_name()
            if "Hour" not in df.columns:
                df["Hour"] = dt.dt.hour.fillna(0)
            return df
    return df

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common raw column names to canonical ones used by our app."""
    mapping = {
        "TYPE": "CrimeType",
        "Type": "CrimeType",
        "Category": "CrimeType",
        "Crime": "CrimeType",
        "CRIME": "CrimeType",
        "NEIGHBOURHOOD": "Neighbourhood",
        "Neighbourhood": "Neighbourhood",
        "Neighborhood": "Neighbourhood",
        "HOUR": "Hour",
        "hour": "Hour",
    }
    cols = [mapping.get(c, c) for c in df.columns]
    out = df.copy()
    out.columns = cols
    return out

# ---- main API ----
def validate_or_map_schema(df_raw: pd.DataFrame):
    """Return (ok: bool, df_mapped: DataFrame, issues: str)."""
    df = _standardize_columns(df_raw)
    df = _derive_dayofweek_and_hour(df)

    issues = []
    # Ensure Hour exists at least as numeric 0..23
    if "Hour" not in df.columns:
        df["Hour"] = 0

    missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
        return False, df, "; ".join(issues)

    # Coerce Hour numeric and clip to [0, 23]
    if not pd.api.types.is_numeric_dtype(df["Hour"]):
        df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
    df["Hour"] = df["Hour"].fillna(0).astype(int).clip(0, 23)

    return True, df, "OK"
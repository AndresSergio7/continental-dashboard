from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

REQUIRED_COLUMNS = [
    "Summary",
    "Reporter",
    "Status",
    "Tipo",
    "Resolution",
    "Created",
    "Description",
]

OPTIONAL_COLUMNS = [
    "Issue Type",
    "Agencia",
    "Urgencia",
    "Departamento",
    "Tipo de Error",
    "Last Updated",
    "Updated",
]

# Map Continental Dashboard (Jira) export column names to our standard names
COLUMN_ALIASES = {
    "Custom field (Agencia que reporta)": "Agencia",
    "Custom field (Tipo de solicitud)": "Tipo",
    "Custom field (Nivel de urgencia)": "Urgencia",
    "Custom field (Departamento)": "Departamento",
    "Tipo de Error": "Tipo de Error",
    "Last Viewed": "Last Updated",
}

CREATED_DATETIME_FORMAT = "%m/%d/%Y %H:%M"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map aliased column names to standard names."""
    renames = {old: new for old, new in COLUMN_ALIASES.items() if old in df.columns}
    if renames:
        df = df.rename(columns=renames)
    return df


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(
        series,
        format=CREATED_DATETIME_FORMAT,
        errors="coerce",
    )
    missing_mask = parsed.isna()
    if missing_mask.any():
        fallback = pd.to_datetime(series[missing_mask], errors="coerce")
        parsed.loc[missing_mask] = fallback
    return parsed


def load_jira_from_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and parse a Jira DataFrame (e.g. from file upload).
    Supports both legacy format and Continental Dashboard export.
    """
    df = _normalize_columns(df_raw.copy())

    # Ensure standard columns exist (create empty if missing)
    for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Resolution date: Continental uses "Resolved", legacy may use "Resolution"
    resolved_date_col = df.get("Resolved", pd.Series("", index=df.index)).astype(str).str.strip()
    resolution_raw = df["Resolution"].astype(str).str.strip()
    resolution_dt_raw = resolved_date_col.mask(resolved_date_col.eq(""), pd.NA).fillna(
        resolution_raw.mask(resolution_raw.eq(""), pd.NA)
    )

    last_updated_raw = df.get("Last Updated", pd.Series("", index=df.index)).astype(str).str.strip()
    updated_raw = df.get("Updated", pd.Series("", index=df.index)).astype(str).str.strip()
    merged_last_updated = last_updated_raw.mask(last_updated_raw.eq(""), pd.NA).fillna(
        updated_raw.mask(updated_raw.eq(""), pd.NA)
    )

    df["Created_dt"] = _parse_datetime_series(df["Created"])
    df["Resolution_dt"] = _parse_datetime_series(resolution_dt_raw)
    df["Last_Updated_dt"] = _parse_datetime_series(merged_last_updated)

    df = df.dropna(subset=["Created_dt"]).copy()

    df["Date"] = df["Created_dt"].dt.strftime("%Y-%m-%d")
    df["ResolvedDate"] = df["Resolution_dt"].dt.strftime("%Y-%m-%d")
    df["DayOfWeek"] = df["Created_dt"].dt.day_name().str[:3]

    return df


def load_jira_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load Jira CSV/Excel and parse date fields. Supports legacy and Continental Dashboard formats.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df_raw = pd.read_excel(path)
    else:
        df_raw = pd.read_csv(path)

    return load_jira_from_dataframe(df_raw)

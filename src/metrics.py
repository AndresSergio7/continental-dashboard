from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def apply_filters(
    df: pd.DataFrame,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    statuses: Iterable[str] | None = None,
    tipos: Iterable[str] | None = None,
    reporters: Iterable[str] | None = None,
    agencias: Iterable[str] | None = None,
) -> pd.DataFrame:
    filtered = df.copy()

    if date_range and "Created_dt" in filtered.columns:
        start_dt, end_dt = date_range
        filtered = filtered[
            (filtered["Created_dt"] >= pd.to_datetime(start_dt))
            & (filtered["Created_dt"] <= pd.to_datetime(end_dt) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        ]

    filter_map = {
        "Status": statuses,
        "Tipo": tipos,
        "Reporter": reporters,
        "Agencia": agencias,
    }
    for col, selected_values in filter_map.items():
        if selected_values:
            selected_values = list(selected_values)
            filtered = filtered[filtered[col].isin(selected_values)]

    return filtered


def compute_kpis(df: pd.DataFrame) -> dict[str, str | int | float]:
    total = int(len(df))
    if total == 0:
        return {
            "total_tickets": 0,
            "open_tickets": 0,
            "pct_done": 0.0,
            "avg_tickets_day": 0.0,
            "top_tipo": "N/A",
            "top_agencia": "N/A",
        }

    open_tickets = int((df["Status"] != "Done").sum())
    done_tickets = int((df["Status"] == "Done").sum())
    pct_done = (done_tickets / total) * 100

    unique_days = max(df["Date"].nunique(), 1)
    avg_tickets_day = total / unique_days

    top_tipo = (
        df["Tipo"].value_counts(dropna=False).index[0]
        if not df["Tipo"].empty
        else "N/A"
    )
    top_agencia = (
        df["Agencia"].value_counts(dropna=False).index[0]
        if "Agencia" in df.columns and not df["Agencia"].empty
        else "N/A"
    )

    return {
        "total_tickets": total,
        "open_tickets": open_tickets,
        "pct_done": round(pct_done, 2),
        "avg_tickets_day": round(avg_tickets_day, 2),
        "top_tipo": str(top_tipo),
        "top_agencia": str(top_agencia),
    }


def tickets_per_day(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Date", "tickets", "rolling_7d"])

    daily = (
        df.groupby("Date", as_index=False)
        .size()
        .rename(columns={"size": "tickets"})
        .sort_values("Date")
    )
    daily["Date"] = pd.to_datetime(daily["Date"], format="%Y-%m-%d", errors="coerce")
    daily["rolling_7d"] = daily["tickets"].rolling(window=7, min_periods=1).mean()
    return daily


def count_by_column(df: pd.DataFrame, column: str, top_n: int | None = None) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])

    out = (
        df[column]
        .fillna("Unknown")
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name="count")
    )
    if top_n is not None:
        out = out.head(top_n)
    return out


def add_operational_time_metrics(df: pd.DataFrame, now_ts: pd.Timestamp | None = None) -> pd.DataFrame:
    out = df.copy()
    now_ref = pd.Timestamp.now() if now_ts is None else pd.to_datetime(now_ts)

    created = out["Created_dt"] if "Created_dt" in out.columns else pd.Series(pd.NaT, index=out.index)
    resolved = out["Resolution_dt"] if "Resolution_dt" in out.columns else pd.Series(pd.NaT, index=out.index)
    updated = out["Last_Updated_dt"] if "Last_Updated_dt" in out.columns else pd.Series(pd.NaT, index=out.index)
    status_series = out.get("Status", pd.Series("", index=out.index)).astype(str).str.strip().str.lower()
    done_mask = status_series.eq("done")

    # Effective finish datetime:
    # 1) Resolution_dt when present
    # 2) Last_Updated_dt for Done tickets with empty Resolution_dt
    out["resolution_effective_dt"] = resolved
    fallback_done_mask = out["resolution_effective_dt"].isna() & done_mask & updated.notna()
    out.loc[fallback_done_mask, "resolution_effective_dt"] = updated.loc[fallback_done_mask]

    out["resolution_time_hours"] = (
        (out["resolution_effective_dt"] - created).dt.total_seconds() / 3600.0
        if "Created_dt" in out.columns
        else np.nan
    )
    out.loc[out["resolution_time_hours"] < 0, "resolution_time_hours"] = np.nan
    out["ResolvedDate"] = out["resolution_effective_dt"].dt.strftime("%Y-%m-%d")

    open_mask = ~done_mask
    out["aging_hours"] = np.where(
        open_mask,
        (now_ref - created).dt.total_seconds() / 3600.0,
        np.nan,
    )
    out.loc[out["aging_hours"] < 0, "aging_hours"] = np.nan

    out["touch_time_hours"] = (
        (updated - created).dt.total_seconds() / 3600.0
        if "Last_Updated_dt" in out.columns
        else np.nan
    )
    out.loc[out["touch_time_hours"] < 0, "touch_time_hours"] = np.nan

    return out


def avg_resolution_time_over_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "resolution_time_hours" not in df.columns:
        return pd.DataFrame(columns=["ResolvedDate", "avg_resolution_time_hours"])
    resolved = df[df["resolution_time_hours"].notna() & df["ResolvedDate"].notna()].copy()
    if resolved.empty:
        return pd.DataFrame(columns=["ResolvedDate", "avg_resolution_time_hours"])
    out = (
        resolved.groupby("ResolvedDate", as_index=False)["resolution_time_hours"]
        .mean()
        .rename(columns={"resolution_time_hours": "avg_resolution_time_hours"})
    )
    out["ResolvedDate"] = pd.to_datetime(out["ResolvedDate"], errors="coerce")
    return out.sort_values("ResolvedDate")


def resolution_time_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "resolution_time_hours" not in df.columns:
        return pd.DataFrame(columns=["resolution_time_hours"])
    out = df[df["resolution_time_hours"].notna()][["resolution_time_hours"]].copy()
    return out


def resolution_time_by_tipo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "resolution_time_hours" not in df.columns:
        return pd.DataFrame(columns=["Tipo", "avg_resolution_time_hours"])
    resolved = df[df["resolution_time_hours"].notna()].copy()
    if resolved.empty:
        return pd.DataFrame(columns=["Tipo", "avg_resolution_time_hours"])
    out = (
        resolved.groupby("Tipo", as_index=False)["resolution_time_hours"]
        .mean()
        .rename(columns={"resolution_time_hours": "avg_resolution_time_hours"})
        .sort_values("avg_resolution_time_hours", ascending=False)
    )
    return out


def aging_buckets_open_tickets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "aging_hours" not in df.columns:
        return pd.DataFrame(columns=["aging_bucket", "count"])
    open_df = df[df["Status"].astype(str).ne("Done") & df["aging_hours"].notna()].copy()
    if open_df.empty:
        return pd.DataFrame(columns=["aging_bucket", "count"])
    bins = [0, 24, 48, 72, 168, np.inf]
    labels = ["0-24h", "24-48h", "48-72h", "72h-7d", "7d+"]
    open_df["aging_bucket"] = pd.cut(
        open_df["aging_hours"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    out = (
        open_df["aging_bucket"]
        .value_counts(sort=False, dropna=False)
        .rename_axis("aging_bucket")
        .reset_index(name="count")
    )
    out["aging_bucket"] = out["aging_bucket"].astype(str)
    return out


def created_vs_resolved_per_day(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Date", "created", "resolved"])
    created = (
        df.groupby("Date", as_index=False)
        .size()
        .rename(columns={"size": "created"})
    )
    resolved = (
        df[df.get("resolution_effective_dt", pd.Series(pd.NaT, index=df.index)).notna()]
        .groupby("ResolvedDate", as_index=False)
        .size()
        .rename(columns={"ResolvedDate": "Date", "size": "resolved"})
    )
    merged = created.merge(resolved, on="Date", how="outer").fillna(0)
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged["created"] = merged["created"].astype(int)
    merged["resolved"] = merged["resolved"].astype(int)
    return merged.sort_values("Date")


def client_pending_aging(df: pd.DataFrame) -> pd.DataFrame:
    """Tickets con status PENDIENTE CLIENTE y su tiempo esperando respuesta (aging en horas)."""
    if df.empty or "aging_hours" not in df.columns or "Status" not in df.columns:
        return pd.DataFrame(columns=["aging_hours", "aging_bucket"])
    status_lower = df["Status"].astype(str).str.strip().str.lower()
    # Incluye "PENDIENTE CLIENTE" y "PENDEINTE CLIENTE" (typo común)
    pendiente_mask = status_lower.str.contains("pende.*cliente", regex=True, na=False)
    pending = df[pendiente_mask & df["aging_hours"].notna()][["aging_hours"]].copy()
    if pending.empty:
        return pd.DataFrame(columns=["aging_hours", "aging_bucket"])
    bins = [0, 24, 48, 72, 168, np.inf]
    labels = ["0-24h", "24-48h", "48-72h", "72h-7d", "7d+"]
    pending["aging_bucket"] = pd.cut(
        pending["aging_hours"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    return pending


def client_pending_aging_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Conteo por rango de horas para tickets pendientes de cliente."""
    pending = client_pending_aging(df)
    if pending.empty or "aging_bucket" not in pending.columns:
        return pd.DataFrame(columns=["aging_bucket", "count"])
    out = (
        pending["aging_bucket"]
        .value_counts(sort=False, dropna=False)
        .rename_axis("aging_bucket")
        .reset_index(name="count")
    )
    out["aging_bucket"] = out["aging_bucket"].astype(str)
    # Ordenar por rango lógico
    order = ["0-24h", "24-48h", "48-72h", "72h-7d", "7d+"]
    out["aging_bucket"] = pd.Categorical(out["aging_bucket"], categories=order, ordered=True)
    return out.sort_values("aging_bucket").reset_index(drop=True)


def sla_compliance(df: pd.DataFrame, threshold_hours: float = 48.0) -> dict[str, float | int]:
    if df.empty or "resolution_time_hours" not in df.columns:
        return {"sla_pct": 0.0, "resolved_count": 0, "within_sla_count": 0}
    resolved = df[df["resolution_time_hours"].notna()].copy()
    if resolved.empty:
        return {"sla_pct": 0.0, "resolved_count": 0, "within_sla_count": 0}
    within = int((resolved["resolution_time_hours"] <= float(threshold_hours)).sum())
    total = int(len(resolved))
    pct = (within / total) * 100 if total else 0.0
    return {
        "sla_pct": round(pct, 2),
        "resolved_count": total,
        "within_sla_count": within,
    }

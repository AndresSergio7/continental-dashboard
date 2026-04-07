from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re
from typing import Iterable

import pandas as pd


MONTH_NAMES_ES = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}

DONE_STATUS = {"done", "cerrado", "closed", "resuelto", "resolved"}


@dataclass(frozen=True)
class MonthContext:
    current_period: pd.Period | None
    previous_period: pd.Period | None
    current_df: pd.DataFrame
    previous_df: pd.DataFrame


@dataclass(frozen=True)
class InsightItem:
    title: str
    detail: str


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return "Sin dato"
    text = str(value).strip()
    return text if text else "Sin dato"


def ensure_report_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rename_map = {
        "Custom field (Nivel de urgencia)": "Nivel de urgencia",
        "Custom field (Departamento)": "Departamento",
        "Tipo de Error": "Tipo de Error",
        "Custom field (Agencia que reporta)": "Agencia",
    }
    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})

    defaults = {
        "Nivel de urgencia": "Sin dato",
        "Departamento": "Sin dato",
        "Tipo de Error": "Sin dato",
        "Agencia": "Sin dato",
        "Status": "Sin dato",
    }
    for col, fallback in defaults.items():
        if col not in out.columns:
            out[col] = fallback
        out[col] = out[col].map(_normalize_text).fillna(fallback)

    if "Created_dt" in out.columns:
        out["ReportMonth"] = out["Created_dt"].dt.to_period("M")
        out["ReportMonthLabel"] = out["Created_dt"].dt.strftime("%Y-%m")

    return out


def get_month_context(df: pd.DataFrame) -> MonthContext:
    if df.empty or "ReportMonth" not in df.columns:
        return MonthContext(None, None, df.iloc[0:0].copy(), df.iloc[0:0].copy())

    months = sorted([m for m in df["ReportMonth"].dropna().unique()])
    if not months:
        return MonthContext(None, None, df.iloc[0:0].copy(), df.iloc[0:0].copy())

    current_period = months[-1]
    previous_period = months[-2] if len(months) >= 2 else None

    current_df = df[df["ReportMonth"] == current_period].copy()
    previous_df = df[df["ReportMonth"] == previous_period].copy() if previous_period is not None else df.iloc[0:0].copy()

    return MonthContext(current_period, previous_period, current_df, previous_df)


def period_to_spanish_label(period: pd.Period | None) -> str:
    if period is None:
        return "Sin periodo"
    return f"{MONTH_NAMES_ES.get(period.month, period.strftime('%B'))} {period.year}"


def _safe_pct_change(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return ((current - previous) / previous) * 100.0


def _top_value(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return "Sin dato"
    series = df[column].fillna("Sin dato").astype(str).str.strip()
    series = series.mask(series.eq(""), "Sin dato")
    counts = series.value_counts(dropna=False)
    return str(counts.index[0]) if not counts.empty else "Sin dato"


def _closed_mask(df: pd.DataFrame) -> pd.Series:
    if "Status" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["Status"].astype(str).str.strip().str.lower().isin(DONE_STATUS)


def compute_monthly_summary(month_ctx: MonthContext) -> dict[str, object]:
    current = month_ctx.current_df
    previous = month_ctx.previous_df

    current_total = int(len(current))
    previous_total = int(len(previous))
    delta_abs = current_total - previous_total
    delta_pct = _safe_pct_change(current_total, previous_total)

    current_closed = int(_closed_mask(current).sum()) if not current.empty else 0
    closure_rate = round((current_closed / current_total) * 100, 2) if current_total else 0.0

    return {
        "current_label": period_to_spanish_label(month_ctx.current_period),
        "previous_label": period_to_spanish_label(month_ctx.previous_period) if month_ctx.previous_period else None,
        "current_total": current_total,
        "previous_total": previous_total,
        "delta_abs": delta_abs,
        "delta_pct": None if delta_pct is None else round(delta_pct, 2),
        "current_closed": current_closed,
        "closure_rate": closure_rate,
        "agencies_count": int(current["Agencia"].nunique()) if "Agencia" in current.columns else 0,
        "departments_count": int(current["Departamento"].nunique()) if "Departamento" in current.columns else 0,
        "top_urgency": _top_value(current, "Nivel de urgencia"),
        "top_agencia": _top_value(current, "Agencia"),
        "top_error": _top_value(current, "Tipo de Error"),
    }


def compare_counts_by_column(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame,
    column: str,
    top_n: int | None = None,
) -> pd.DataFrame:
    current_counts = (
        current_df[column].fillna("Sin dato").astype(str).str.strip().replace("", "Sin dato").value_counts().rename("Mes actual")
        if not current_df.empty and column in current_df.columns
        else pd.Series(dtype="int64", name="Mes actual")
    )
    previous_counts = (
        previous_df[column].fillna("Sin dato").astype(str).str.strip().replace("", "Sin dato").value_counts().rename("Mes anterior")
        if not previous_df.empty and column in previous_df.columns
        else pd.Series(dtype="int64", name="Mes anterior")
    )

    merged = pd.concat([current_counts, previous_counts], axis=1).fillna(0).reset_index()
    merged = merged.rename(columns={"index": column})
    merged["Mes actual"] = merged["Mes actual"].astype(int)
    merged["Mes anterior"] = merged["Mes anterior"].astype(int)
    merged["Variación"] = merged["Mes actual"] - merged["Mes anterior"]
    merged["Total"] = merged["Mes actual"] + merged["Mes anterior"]
    merged = merged.sort_values(["Mes actual", "Mes anterior", column], ascending=[False, False, True])
    if top_n is not None:
        merged = merged.head(top_n)
    return merged.reset_index(drop=True)


def current_month_counts(df: pd.DataFrame, column: str, top_n: int | None = None) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])
    out = (
        df[column].fillna("Sin dato").astype(str).str.strip().replace("", "Sin dato").value_counts().rename_axis(column).reset_index(name="count")
    )
    if top_n is not None:
        out = out.head(top_n)
    return out.reset_index(drop=True)


def build_change_insights(month_ctx: MonthContext) -> list[InsightItem]:
    if month_ctx.previous_period is None:
        return []

    insights: list[InsightItem] = []
    for column, title in [
        ("Agencia", "Mayor aumento por agencia"),
        ("Departamento", "Mayor aumento por departamento"),
        ("Tipo de Error", "Mayor aumento por tipo de error"),
    ]:
        comparison = compare_counts_by_column(month_ctx.current_df, month_ctx.previous_df, column)
        if comparison.empty:
            continue
        max_row = comparison.sort_values(["Variación", "Mes actual"], ascending=[False, False]).iloc[0]
        min_row = comparison.sort_values(["Variación", "Mes actual"], ascending=[True, False]).iloc[0]
        if int(max_row["Variación"]) > 0:
            insights.append(
                InsightItem(
                    title=title,
                    detail=f"{max_row[column]} registró {int(max_row['Variación'])} tickets más que el mes anterior.",
                )
            )
        if int(min_row["Variación"]) < 0:
            insights.append(
                InsightItem(
                    title=title.replace("aumento", "disminución"),
                    detail=f"{min_row[column]} registró {abs(int(min_row['Variación']))} tickets menos que el mes anterior.",
                )
            )
    return insights[:4]


def build_narrative_insights(summary: dict[str, object], month_ctx: MonthContext) -> list[str]:
    current_label = str(summary.get("current_label", "Mes actual"))
    previous_label = summary.get("previous_label")
    current_total = int(summary.get("current_total", 0))
    previous_total = int(summary.get("previous_total", 0))
    delta_abs = int(summary.get("delta_abs", 0))
    delta_pct = summary.get("delta_pct")
    closure_rate = float(summary.get("closure_rate", 0.0))

    bullets: list[str] = []

    if previous_label:
        if delta_abs > 0:
            pct_text = f" ({delta_pct}%)" if delta_pct is not None else ""
            bullets.append(f"En {current_label} se registró un aumento de {delta_abs} tickets frente a {previous_label}{pct_text}.")
        elif delta_abs < 0:
            pct_text = f" ({abs(delta_pct)}%)" if delta_pct is not None else ""
            bullets.append(f"En {current_label} se observó una disminución de {abs(delta_abs)} tickets respecto a {previous_label}{pct_text}.")
        else:
            bullets.append(f"El volumen total de tickets se mantuvo estable entre {previous_label} y {current_label}.")
    else:
        bullets.append(f"En {current_label} se registraron {current_total} tickets en total.")

    bullets.append(
        f"La operación del mes se concentró principalmente en la urgencia '{summary.get('top_urgency', 'Sin dato')}', la agencia '{summary.get('top_agencia', 'Sin dato')}' y el tipo de error '{summary.get('top_error', 'Sin dato')}'."
    )
    bullets.append(f"La tasa de cierre del mes fue de {closure_rate}%, con {int(summary.get('current_closed', 0))} tickets cerrados de {current_total} totales.")

    change_insights = build_change_insights(month_ctx)
    for item in change_insights[:2]:
        bullets.append(item.detail)

    return bullets[:5]


def build_executive_summary_text(summary: dict[str, object]) -> str:
    current_label = str(summary.get("current_label", "Mes actual"))
    previous_label = summary.get("previous_label")
    current_total = int(summary.get("current_total", 0))
    top_agencia = summary.get("top_agencia", "Sin dato")
    top_error = summary.get("top_error", "Sin dato")
    delta_abs = int(summary.get("delta_abs", 0))
    delta_pct = summary.get("delta_pct")

    if previous_label:
        if delta_abs < 0:
            change_text = f"una disminución de {abs(delta_abs)} tickets"
            if delta_pct is not None:
                change_text += f" ({abs(delta_pct)}%)"
        elif delta_abs > 0:
            change_text = f"un aumento de {delta_abs} tickets"
            if delta_pct is not None:
                change_text += f" ({delta_pct}%)"
        else:
            change_text = "un volumen estable de tickets"
        return (
            f"En {current_label} se registraron {current_total} tickets, mostrando {change_text} frente a {previous_label}. "
            f"La mayor concentración se observó en {top_agencia} y predominó el tipo de error {top_error}."
        )
    return (
        f"En {current_label} se registraron {current_total} tickets. "
        f"La mayor concentración se observó en {top_agencia} y predominó el tipo de error {top_error}."
    )


def format_delta(delta_abs: int, delta_pct: float | None) -> str:
    sign = "+" if delta_abs > 0 else ""
    pct_text = "s/d" if delta_pct is None else f"{delta_pct}%"
    return f"{sign}{delta_abs} tickets · {pct_text}"

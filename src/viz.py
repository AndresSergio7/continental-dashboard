from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def line_tickets_per_day(daily_df: pd.DataFrame, include_rolling: bool = True) -> go.Figure:
    fig = go.Figure()

    if daily_df.empty:
        fig.update_layout(title="Tickets creados por día")
        return fig

    fig.add_trace(
        go.Scatter(
            x=daily_df["Date"],
            y=daily_df["tickets"],
            mode="lines+markers",
            name="Tickets por día",
        )
    )

    if include_rolling and "rolling_7d" in daily_df.columns:
        fig.add_trace(
            go.Scatter(
                x=daily_df["Date"],
                y=daily_df["rolling_7d"],
                mode="lines",
                name="Promedio 7 días",
                line={"dash": "dash"},
            )
        )

    fig.update_layout(
        title="Tickets creados por día",
        xaxis_title="Fecha",
        yaxis_title="Cantidad",
        legend_title="Serie",
    )
    return fig


def bar_counts(
    count_df: pd.DataFrame,
    x_col: str,
    y_col: str = "count",
    title: str = "",
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
) -> go.Figure:
    if count_df.empty:
        return go.Figure(layout={"title": title or "Sin datos"})
    fig = px.bar(count_df, x=x_col, y=y_col, title=title)
    fig.update_layout(
        xaxis_title=xaxis_title if xaxis_title is not None else x_col,
        yaxis_title=yaxis_title if yaxis_title is not None else y_col.replace("_", " ").title(),
    )
    return fig


def line_avg_resolution_time(avg_df: pd.DataFrame) -> go.Figure:
    if avg_df.empty:
        return go.Figure(layout={"title": "Tiempo promedio de resolución por fecha"})
    fig = px.line(
        avg_df,
        x="ResolvedDate",
        y="avg_resolution_time_hours",
        markers=True,
        title="Tiempo promedio de resolución por fecha de cierre",
    )
    fig.update_layout(xaxis_title="Fecha de cierre", yaxis_title="Horas")
    return fig


def histogram_resolution_time(dist_df: pd.DataFrame) -> go.Figure:
    if dist_df.empty:
        return go.Figure(layout={"title": "Distribución del tiempo de resolución"})
    fig = px.histogram(
        dist_df,
        x="resolution_time_hours",
        nbins=30,
        title="Distribución del tiempo de resolución (horas)",
    )
    fig.update_layout(xaxis_title="Horas para cerrar", yaxis_title="Cantidad de tickets")
    return fig


def bar_client_pending_aging(bucket_df: pd.DataFrame) -> go.Figure:
    """Gráfica de tiempo esperando respuesta del cliente (tickets PENDIENTE CLIENTE)."""
    if bucket_df.empty:
        return go.Figure(
            layout={
                "title": "Tiempo esperando respuesta del cliente (PENDIENTE CLIENTE)",
                "annotations": [
                    {"text": "No hay tickets pendientes de cliente con datos", "xref": "paper", "yref": "paper", "showarrow": False}
                ],
            }
        )
    fig = px.bar(
        bucket_df,
        x="aging_bucket",
        y="count",
        title="Tiempo esperando respuesta del cliente (PENDIENTE CLIENTE)",
    )
    fig.update_layout(
        xaxis_title="Horas esperando",
        yaxis_title="Cantidad de tickets",
    )
    return fig


def dual_line_created_vs_resolved(daily_df: pd.DataFrame) -> go.Figure:
    # Azul oscuro y celeste oscuro para diferenciar las dos líneas
    colors = ["#1e3a5f", "#0d9488"]
    fig = go.Figure()
    if daily_df.empty:
        fig.update_layout(title="Tickets creados vs cerrados por día")
        return fig
    fig.add_trace(
        go.Scatter(
            x=daily_df["Date"],
            y=daily_df["created"],
            mode="lines+markers",
            name="Creados",
            line=dict(color=colors[0]),
            marker=dict(color=colors[0]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily_df["Date"],
            y=daily_df["resolved"],
            mode="lines+markers",
            name="Cerrados",
            line=dict(color=colors[1]),
            marker=dict(color=colors[1]),
        )
    )
    fig.update_layout(
        title="Tickets creados vs cerrados por día",
        xaxis_title="Fecha",
        yaxis_title="Cantidad",
    )
    return fig


def grouped_month_comparison(
    comparison_df: pd.DataFrame,
    category_col: str,
    title: str,
) -> go.Figure:
    if comparison_df.empty:
        return go.Figure(layout={"title": title or "Sin datos para comparar"})
    fig = px.bar(
        comparison_df,
        x=category_col,
        y="count",
        color="Mes",
        barmode="group",
        title=title,
        color_discrete_map={"Mes actual": "#1e3a5f", "Mes anterior": "#0d9488"},
    )
    fig.update_layout(xaxis_title=category_col, yaxis_title="Tickets", legend_title="Periodo")
    return fig


def horizontal_distribution(
    count_df: pd.DataFrame,
    category_col: str,
    title: str,
) -> go.Figure:
    if count_df.empty:
        return go.Figure(layout={"title": title or "Sin datos"})
    fig = px.bar(
        count_df.sort_values("count", ascending=True),
        x="count",
        y=category_col,
        orientation="h",
        title=title,
        color_discrete_sequence=["#2c5282"],
    )
    fig.update_layout(xaxis_title="Tickets", yaxis_title="")
    return fig

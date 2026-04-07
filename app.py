from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.cleaning import clean_jira_dataframe
from src.io import load_jira_csv, load_jira_from_dataframe
from src.metrics import (
    add_operational_time_metrics,
    aging_buckets_open_tickets,
    apply_filters,
    avg_resolution_time_over_time,
    client_pending_aging_buckets,
    compute_kpis,
    compare_category_months,
    count_by_column,
    created_vs_resolved_per_day,
    detect_report_months,
    filter_by_month,
    generate_monthly_insights,
    monthly_kpis,
    month_label,
    composition_change_summary,
    resolution_time_by_tipo,
    resolution_time_distribution,
    sla_compliance,
    tickets_per_day,
)
from src.nlp import ThemeResult, build_description_themes, generate_theme_example_sentence
from src.viz import (
    bar_client_pending_aging,
    bar_counts,
    dual_line_created_vs_resolved,
    grouped_month_comparison,
    horizontal_distribution,
    histogram_resolution_time,
    line_avg_resolution_time,
    line_tickets_per_day,
)

DATA_PATH = Path("data") / "Jira (3).csv"
OUTPUTS_DIR = Path("outputs")
DATA_CACHE_VERSION = 5
NLP_CACHE_VERSION = 6


@st.cache_data
def load_and_prepare_data(path: str, cache_version: int = DATA_CACHE_VERSION) -> pd.DataFrame:
    _ = cache_version
    df = load_jira_csv(path)
    df = clean_jira_dataframe(df)
    df = add_operational_time_metrics(df)
    return df


@st.cache_data
def compute_nlp_themes(
    df: pd.DataFrame,
    keywords_per_theme: int = 12,
    examples_per_theme: int = 3,
    cache_version: int = NLP_CACHE_VERSION,
) -> ThemeResult:
    _ = cache_version
    label_terms = max(3, min(8, keywords_per_theme // 2))
    return build_description_themes(
        df,
        top_keywords=keywords_per_theme,
        label_terms=label_terms,
        top_representatives=examples_per_theme,
    )


def render_kpis(df: pd.DataFrame) -> None:
    kpis = compute_kpis(df)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total de tickets", kpis["total_tickets"])
    c2.metric("Tickets abiertos", kpis["open_tickets"])
    c3.metric("% Cerrados", f"{kpis['pct_done']}%")
    c4.metric("Promedio tickets por día", kpis["avg_tickets_day"])
    c5.metric("Tipo más común", kpis["top_tipo"])
    c6.metric("Agencia más común", kpis["top_agencia"])


def build_monthly_report_context(df: pd.DataFrame) -> dict:
    current_month, previous_month, _ = detect_report_months(df)
    df_current = filter_by_month(df, current_month)
    df_previous = filter_by_month(df, previous_month) if previous_month is not None else None
    kpis = monthly_kpis(df_current, df_previous)
    insights = generate_monthly_insights(kpis, df_current, df_previous)

    comparison_figs: list[tuple[str, object]] = []
    if df_previous is not None and not df_previous.empty:
        totals_df = pd.DataFrame(
            {
                "Métrica": ["Tickets creados", "Tickets creados"],
                "Mes": ["Mes actual", "Mes anterior"],
                "count": [len(df_current), len(df_previous)],
            }
        )
        comparison_figs.append(("Total tickets: actual vs anterior", grouped_month_comparison(totals_df, "Métrica", "Total tickets: comparación mensual")))
        comparison_figs.append(
            ("Tickets por urgencia (comparativo)", grouped_month_comparison(compare_category_months(df_current, df_previous, "Urgencia"), "Urgencia", "Tickets por urgencia"))
        )
        comparison_figs.append(
            ("Tickets por departamento (comparativo)", grouped_month_comparison(compare_category_months(df_current, df_previous, "Departamento"), "Departamento", "Tickets por departamento"))
        )
        comparison_figs.append(
            ("Top 5 agencias (comparativo)", grouped_month_comparison(compare_category_months(df_current, df_previous, "Agencia", top_n=5), "Agencia", "Top 5 agencias por volumen"))
        )
        comparison_figs.append(
            ("Top 5 tipos de error (comparativo)", grouped_month_comparison(compare_category_months(df_current, df_previous, "Tipo de Error", top_n=5), "Tipo de Error", "Top 5 tipos de error"))
        )

    current_figs = [
        ("Distribución por urgencia", horizontal_distribution(count_by_column(df_current, "Urgencia"), "Urgencia", "Distribución de tickets por urgencia")),
        ("Distribución por departamento", horizontal_distribution(count_by_column(df_current, "Departamento"), "Departamento", "Distribución de tickets por departamento")),
        ("Distribución por tipo de error", horizontal_distribution(count_by_column(df_current, "Tipo de Error"), "Tipo de Error", "Distribución de tickets por tipo de error")),
        ("Ranking de agencias (mes actual)", horizontal_distribution(count_by_column(df_current, "Agencia", top_n=10), "Agencia", "Ranking de agencias (mes actual)")),
        ("Distribución por estado", horizontal_distribution(count_by_column(df_current, "Status"), "Status", "Distribución de tickets por estado")),
    ]

    composition_rows = []
    if df_previous is not None and not df_previous.empty:
        for col, label in [("Agencia", "Agencias"), ("Departamento", "Departamentos"), ("Tipo de Error", "Tipos de error")]:
            item = composition_change_summary(df_current, df_previous, col)
            composition_rows.append(
                {
                    "Categoría": label,
                    "Mayor incremento": f"{item['mayor_alza']} ({item['valor_alza']:+d})",
                    "Mayor disminución": f"{item['mayor_baja']} ({item['valor_baja']:+d})",
                }
            )

    return {
        "current_month": current_month,
        "previous_month": previous_month,
        "df_current": df_current,
        "df_previous": df_previous,
        "kpis": kpis,
        "insights": insights,
        "comparison_figs": comparison_figs,
        "current_figs": current_figs,
        "composition_rows": composition_rows,
    }


def build_html_report(df: pd.DataFrame, sla_threshold_hours: float, embed_plotly: bool = False) -> str:
    """Genera un reporte HTML mensual ejecutivo."""
    _ = sla_threshold_hours
    ctx = build_monthly_report_context(df)
    kpis = ctx["kpis"]
    current_label = month_label(ctx["current_month"]) if ctx["current_month"] is not None else "sin datos"
    previous_label = month_label(ctx["previous_month"]) if ctx["previous_month"] is not None else None
    delta_pct = kpis["delta_pct"]
    delta_text = "Sin comparación disponible" if delta_pct is None else f"{kpis['delta_abs']:+d} ({delta_pct:+.1f}%)"
    subtitle = f"Comparación contra {previous_label}" if previous_label else "No existe mes previo para comparar"
    resumen = (
        f"En {current_label} se registraron {kpis['tickets_actual']} tickets. "
        + (f"Variación de {delta_pct:+.1f}% respecto a {previous_label}. " if delta_pct is not None else "")
        + f"Se concentraron principalmente en {kpis['top_agencia']} y en incidencias de tipo {kpis['top_error']}."
    )

    kpi_html = f"""
    <div class="kpis">
        <div class="kpi"><span class="label">Tickets creados (mes actual)</span><span class="value">{kpis["tickets_actual"]}</span></div>
        <div class="kpi"><span class="label">Tickets creados (mes anterior)</span><span class="value">{kpis["tickets_prev"]}</span></div>
        <div class="kpi"><span class="label">Variación mensual</span><span class="value">{delta_text}</span></div>
        <div class="kpi"><span class="label">Tickets resueltos</span><span class="value">{kpis["resueltos"]}</span></div>
        <div class="kpi"><span class="label">Tasa de resolución</span><span class="value">{kpis["tasa_resolucion"] if kpis["tasa_resolucion"] is not None else "N/A"}%</span></div>
        <div class="kpi"><span class="label">Agencias activas</span><span class="value">{kpis["agencias_activas"]}</span></div>
        <div class="kpi"><span class="label">Departamentos activos</span><span class="value">{kpis["departamentos_activos"]}</span></div>
    </div>
    """

    all_figs = ctx["comparison_figs"] + ctx["current_figs"]
    charts = []
    for i, (_, fig) in enumerate(all_figs):
        inc_plotly = (True if i == 0 else False) if embed_plotly else False
        charts.append(fig.to_html(full_html=False, include_plotlyjs=inc_plotly, div_id=f"chart_{i}"))
    charts_html = "".join(f'<div class="chart-wrap">{c}</div>' for c in charts)
    comp_rows = "".join(
        f"<tr><td>{row['Categoría']}</td><td>{row['Mayor incremento']}</td><td>{row['Mayor disminución']}</td></tr>"
        for row in ctx["composition_rows"]
    )
    insights_html = "".join(f"<li>{line}</li>" for line in ctx["insights"])
    plotly_script = "" if embed_plotly else '    <script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js" crossorigin="anonymous"></script>\n'

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte mensual de tickets - Grupo Continental</title>
    <meta name="color-scheme" content="light">
    <style>
        html {{ color-scheme: light; }}
        * {{ box-sizing: border-box; }}
        body {{ font-family: system-ui, -apple-system, sans-serif; margin: 24px; background: #f8f9fa; }}
        h1 {{ color: #1e3a5f; margin-bottom: 8px; }}
        .subtitle {{ color: #666; margin-bottom: 8px; font-size: 14px; }}
        .summary {{ background: #eef4ff; padding: 14px; border-radius: 8px; margin-bottom: 18px; color: #1e3a5f; }}
        .kpis {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; }}
        .kpi {{ background: white; padding: 16px 24px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 140px; }}
        .kpi .label {{ display: block; font-size: 12px; color: #666; margin-bottom: 4px; }}
        .kpi .value {{ font-size: 24px; font-weight: 600; color: #1e3a5f; }}
        .op-metrics {{ margin-top: 8px; }}
        .chart-wrap {{ background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }}
        .chart-wrap .plotly {{ width: 100% !important; }}
        @media print {{ body {{ background: white; }} .chart-wrap {{ break-inside: avoid; }} }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; margin-bottom: 24px; }}
        th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
        th {{ background: #f1f5f9; color: #1e3a5f; }}
    </style>
{plotly_script}
</head>
<body>
    <h1>Reporte mensual de tickets · {current_label.title()}</h1>
    <p class="subtitle">{subtitle} · generado el {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    <div class="summary">{resumen}</div>
    {kpi_html}
    <h2>Tendencias de composición</h2>
    <table><thead><tr><th>Categoría</th><th>Mayor incremento</th><th>Mayor disminución</th></tr></thead><tbody>{comp_rows}</tbody></table>
    <h2>Visualizaciones</h2>
    {charts_html}
    <h2>Hallazgos clave</h2>
    <ul>{insights_html}</ul>
</body>
</html>"""
    return html


def render_dashboard_tab(df: pd.DataFrame, sla_threshold_hours: float) -> None:
    if df.empty:
        st.info("No data for current filters")
        return

    render_kpis(df)

    daily = tickets_per_day(df)
    st.plotly_chart(line_tickets_per_day(daily, include_rolling=True), use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            bar_counts(count_by_column(df, "Tipo"), x_col="Tipo", title="Tickets por tipo de solicitud"),
            use_container_width=True,
        )
        st.plotly_chart(
            bar_counts(count_by_column(df, "Reporter", top_n=15), x_col="Reporter", title="Tickets por reportero (top 15)"),
            use_container_width=True,
        )
    with col_right:
        st.plotly_chart(
            bar_counts(count_by_column(df, "Status"), x_col="Status", title="Tickets por estado"),
            use_container_width=True,
        )
        st.plotly_chart(
            bar_counts(count_by_column(df, "Agencia", top_n=20), x_col="Agencia", title="Tickets por agencia"),
            use_container_width=True,
        )

    st.markdown("### Tiempo esperando respuesta del cliente")
    st.caption("Tickets en estado PENDIENTE CLIENTE: cuánto tiempo llevan esperando que el cliente responda.")
    st.plotly_chart(
        bar_client_pending_aging(client_pending_aging_buckets(df)),
        use_container_width=True,
    )

    st.markdown("### Desempeño operativo")
    sla_stats = sla_compliance(df, threshold_hours=sla_threshold_hours)
    op_c1, op_c2, op_c3 = st.columns(3)
    op_c1.metric(
        f"% resueltos en {sla_threshold_hours:.0f}h o menos",
        f"{sla_stats['sla_pct']}%",
    )
    op_c2.metric("Tickets resueltos (total)", sla_stats["resolved_count"])
    op_c3.metric("Dentro del plazo", sla_stats["within_sla_count"])

    st.plotly_chart(
        line_avg_resolution_time(avg_resolution_time_over_time(df)),
        use_container_width=True,
    )

    op_left, op_right = st.columns(2)
    with op_left:
        st.plotly_chart(
            histogram_resolution_time(resolution_time_distribution(df)),
            use_container_width=True,
        )
        st.plotly_chart(
            bar_counts(
                resolution_time_by_tipo(df),
                x_col="Tipo",
                y_col="avg_resolution_time_hours",
                title="Tiempo promedio de resolución por tipo",
                yaxis_title="Horas",
            ),
            use_container_width=True,
        )
    with op_right:
        st.plotly_chart(
            bar_counts(
                aging_buckets_open_tickets(df),
                x_col="aging_bucket",
                y_col="count",
                title="Tickets abiertos por antigüedad",
                xaxis_title="Rango de horas",
                yaxis_title="Cantidad",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            dual_line_created_vs_resolved(created_vs_resolved_per_day(df)),
            use_container_width=True,
        )


def render_monthly_report_tab(df: pd.DataFrame) -> None:
    ctx = build_monthly_report_context(df)
    kpis = ctx["kpis"]
    current_month = ctx["current_month"]
    previous_month = ctx["previous_month"]
    current_label = month_label(current_month).title() if current_month is not None else "Sin datos"
    previous_label = month_label(previous_month).title() if previous_month is not None else None

    st.markdown(f"## Reporte mensual ejecutivo · {current_label}")
    st.caption(
        f"Comparación contra {previous_label}" if previous_label else "No existe un mes anterior disponible para comparación."
    )

    delta_pct = kpis["delta_pct"]
    delta_desc = (
        "sin base comparativa"
        if delta_pct is None
        else ("incremento" if float(delta_pct) >= 0 else "disminución")
    )
    st.info(
        f"En {current_label} se observaron {kpis['tickets_actual']} tickets, con {delta_desc} "
        + (f"de {abs(float(delta_pct)):.1f}% frente a {previous_label}. " if delta_pct is not None else ". ")
        + f"Se concentraron principalmente en {kpis['top_agencia']} y en incidencias de tipo {kpis['top_error']}."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickets creados (actual)", kpis["tickets_actual"])
    c2.metric("Tickets creados (anterior)", kpis["tickets_prev"])
    c3.metric("Cambio mensual", "N/A" if delta_pct is None else f"{delta_pct:+.1f}%")
    c4.metric("Tickets resueltos", kpis["resueltos"])
    c5, c6, c7 = st.columns(3)
    c5.metric("Tasa de resolución", "N/A" if kpis["tasa_resolucion"] is None else f"{kpis['tasa_resolucion']:.1f}%")
    c6.metric("Agencias con tickets", kpis["agencias_activas"])
    c7.metric("Departamentos con tickets", kpis["departamentos_activos"])

    if ctx["comparison_figs"]:
        st.markdown("### Comparativos clave (mes actual vs mes anterior)")
        for name, fig in ctx["comparison_figs"]:
            st.plotly_chart(fig, use_container_width=True, key=f"cmp_{name}")

    st.markdown("### Distribución del mes actual")
    dist_left, dist_right = st.columns(2)
    for idx, (name, fig) in enumerate(ctx["current_figs"]):
        target = dist_left if idx % 2 == 0 else dist_right
        with target:
            st.plotly_chart(fig, use_container_width=True, key=f"cur_{name}")

    if ctx["composition_rows"]:
        st.markdown("### ¿Dónde cambiaron más los tickets?")
        st.dataframe(pd.DataFrame(ctx["composition_rows"]), use_container_width=True, hide_index=True)

    st.markdown("### Hallazgos clave")
    for insight in ctx["insights"]:
        st.markdown(f"- {insight}")


def render_nlp_tab(df: pd.DataFrame, keywords_per_theme: int, examples_per_theme: int) -> None:
    if df.empty:
        st.info("No hay datos con los filtros actuales")
        return

    result = compute_nlp_themes(
        df,
        keywords_per_theme=keywords_per_theme,
        examples_per_theme=examples_per_theme,
    )
    if result.themes.empty:
        st.info("No hay suficientes descripciones para generar temas (mín. 20 caracteres por ticket).")
        return

    # Ordenar por cantidad (mayor primero) — el #1 es el más común
    themes = result.themes.sort_values("count", ascending=False).reset_index(drop=True)

    st.markdown("### Casos más comunes")
    st.caption("Ordenados por cantidad de tickets. Cada ejemplo es una oración generada a partir de los tickets del grupo.")

    with st.spinner("Generando ejemplos en oraciones…"):
        items = []
        for rank, (_, row) in enumerate(themes.iterrows(), start=1):
            count = int(row.get("count", 0))
            sentence = generate_theme_example_sentence(
                int(row["theme_id"]), result.themes, result.ticket_themes
            )
            items.append((rank, count, sentence))

    for rank, count, sentence in items:
        st.markdown(f"**#{rank} · {count} tickets**")
        st.write(sentence)
        st.divider()


def main() -> None:
    st.set_page_config(page_title="Dashboards Tickets Grupo Continental", layout="wide")
    st.title("Dashboards Tickets Grupo Continental")

    st.sidebar.header("Data source")
    uploaded_file = st.sidebar.file_uploader(
        "Sube un archivo Jira (CSV o Excel) para actualizar el dashboard",
        type=["csv", "xlsx", "xls"],
        help="Formato Continental Dashboard o export Jira clásico",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
                df_raw = pd.read_excel(uploaded_file)
            else:
                df_raw = pd.read_csv(uploaded_file)
            df = load_jira_from_dataframe(df_raw)
            df = clean_jira_dataframe(df)
            df = add_operational_time_metrics(df)
            st.sidebar.success(f"✅ {uploaded_file.name} cargado ({len(df)} registros)")
        except Exception as e:
            st.sidebar.error(f"Error al procesar el archivo: {e}")
            if DATA_PATH.exists():
                df = load_and_prepare_data(str(DATA_PATH))
                st.sidebar.info("Usando archivo por defecto de data/")
            else:
                st.warning("Sube un archivo válido o coloca Jira (3).csv en data/")
                st.stop()
    elif DATA_PATH.exists():
        df = load_and_prepare_data(str(DATA_PATH))
        st.sidebar.caption(f"Usando: {DATA_PATH.name}")
    else:
        st.warning("Sube un archivo o coloca Jira (3).csv en data/")
        st.stop()

    st.sidebar.header("Filters")
    min_date = df["Created_dt"].min().date()
    max_date = df["Created_dt"].max().date()
    selected_date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
    else:
        start_date = min_date
        end_date = max_date

    selected_status = st.sidebar.multiselect("Status", sorted(df["Status"].dropna().unique().tolist()))
    selected_tipo = st.sidebar.multiselect("Tipo", sorted(df["Tipo"].dropna().unique().tolist()))
    selected_reporter = st.sidebar.multiselect(
        "Reporter", sorted(df["Reporter"].dropna().unique().tolist())
    )
    selected_agencia = st.sidebar.multiselect(
        "Agencia", sorted(df["Agencia"].dropna().unique().tolist())
    )
    st.sidebar.subheader("NLP settings")
    keywords_per_theme = st.sidebar.slider(
        "Words/phrases per theme",
        min_value=7,
        max_value=20,
        value=12,
        step=1,
    )
    st.sidebar.caption("Affects both theme keywords and label detail.")
    examples_per_theme = max(3, min(10, keywords_per_theme // 2))
    st.sidebar.caption(f"Representative examples per theme: up to {examples_per_theme}")
    if st.sidebar.button("Clear cached NLP results"):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.subheader("Operational settings")
    sla_threshold_hours = st.sidebar.slider(
        "Plazo máximo (horas) para considerar 'a tiempo'",
        min_value=4,
        max_value=240,
        value=48,
        step=4,
    )

    filtered_df = apply_filters(
        df=df,
        date_range=(pd.Timestamp(start_date), pd.Timestamp(end_date)),
        statuses=selected_status,
        tipos=selected_tipo,
        reporters=selected_reporter,
        agencias=selected_agencia,
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    st.sidebar.subheader("Exportar reporte")
    report_ctx = build_monthly_report_context(filtered_df)
    current_report_month = report_ctx["current_month"]
    month_name = month_label(current_report_month).replace(" ", "_") if current_report_month is not None else "sin_datos"
    report_filename = f"Reporte_Tickets_GC_{month_name}.html"
    with st.spinner("Preparando reporte…"):
        report_html = build_html_report(filtered_df, float(sla_threshold_hours), embed_plotly=False)
    st.sidebar.download_button(
        "Descargar reporte (rápido)",
        data=report_html.encode("utf-8"),
        file_name=report_filename,
        mime="text/html",
    )
    st.sidebar.caption("Archivo pequeño (~100 KB). Necesita internet para ver las gráficas.")
    if st.sidebar.button("Generar versión para redes restrictivas"):
        with st.spinner("Generando versión completa (puede tardar 30–60 seg)…"):
            report_full = build_html_report(filtered_df, float(sla_threshold_hours), embed_plotly=True)
        st.session_state.report_full = report_full
    if "report_full" in st.session_state:
        st.sidebar.download_button(
            "Descargar versión completa (~4 MB)",
            data=st.session_state.report_full.encode("utf-8"),
            file_name=report_filename.replace(".html", "_completo.html"),
            mime="text/html",
            key="dl_full",
        )
        st.sidebar.caption("Funciona sin internet. Úsala si las gráficas no se ven al compartir.")
    export_col1, export_col2 = st.columns(2)
    if export_col1.button("Export cleaned dataset"):
        cleaned_path = OUTPUTS_DIR / "cleaned_jira.parquet"
        df.to_parquet(cleaned_path, index=False)
        st.success(f"Saved: {cleaned_path}")
    if export_col2.button("Export NLP themes"):
        themes_path = OUTPUTS_DIR / "nlp_themes.csv"
        nlp_result = compute_nlp_themes(
            filtered_df,
            keywords_per_theme=keywords_per_theme,
            examples_per_theme=examples_per_theme,
        )
        nlp_result.themes.to_csv(themes_path, index=False)
        st.success(f"Saved: {themes_path}")

    tab_reporte, tab_dashboard, tab_nlp = st.tabs(["Reporte mensual", "Dashboard", "Casos más comunes"])
    with tab_reporte:
        render_monthly_report_tab(filtered_df)
    with tab_dashboard:
        render_dashboard_tab(filtered_df, sla_threshold_hours=float(sla_threshold_hours))
    with tab_nlp:
        render_nlp_tab(
            filtered_df,
            keywords_per_theme=keywords_per_theme,
            examples_per_theme=examples_per_theme,
        )


if __name__ == "__main__":
    main()

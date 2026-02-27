from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from src.cleaning import truncate_text

SPANISH_TICKET_STOPWORDS = {
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o", "u",
    "a", "ante", "con", "contra", "desde", "en", "entre", "hacia", "hasta", "para",
    "por", "sin", "sobre", "tras", "que", "se", "su", "sus", "al", "del", "lo",
    "le", "les", "como", "ya", "si", "no", "mas", "muy", "favor", "gracias",
    "buen", "buenos", "buenas", "dia", "dias", "tarde", "tardes", "noche", "noches",
    "estimado", "estimados", "adjunto", "adjunta", "quedo", "pendiente", "cordial",
    "saludo", "saludos", "hola", "ticket", "jira",
}
NOISE_KEYWORD_RE = re.compile(
    r"(?:^|[\s_-])(image|imagen|img|png|jpg|jpeg|gif|webp|svg|bmp|heic|alt|width|height)(?:$|[\s_-])",
    re.IGNORECASE,
)


@dataclass
class ThemeResult:
    themes: pd.DataFrame
    ticket_themes: pd.DataFrame
    selected_k: int
    used_silhouette: bool
    total_descriptions: int
    eligible_descriptions: int
    excluded_short_descriptions: int


def _choose_k(
    matrix,
    k_min: int = 8,
    k_max: int = 25,
    random_state: int = 42,
) -> tuple[int, bool]:
    n_samples = matrix.shape[0]
    candidate_ks = [k for k in range(k_min, k_max + 1) if 2 <= k < n_samples]

    if not candidate_ks:
        # Fallback requested by spec when silhouette cannot be evaluated.
        fallback_k = min(10, max(1, n_samples))
        return fallback_k, False

    score_by_k: dict[int, float] = {}
    for k in candidate_ks:
        try:
            model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            labels = model.fit_predict(matrix)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(matrix, labels)
            score_by_k[k] = float(score)
        except Exception:
            continue

    if not score_by_k:
        fallback_k = min(10, max(1, n_samples))
        return fallback_k, False

    best_score = max(score_by_k.values())
    tolerance = 0.02
    eligible_ks = [k for k, score in score_by_k.items() if score >= (best_score - tolerance)]
    selected_k = min(eligible_ks) if eligible_ks else max(score_by_k, key=score_by_k.get)

    return selected_k, True


def _semantic_projection(tfidf_matrix, random_state: int = 42):
    """
    Build dense semantic vectors from TF-IDF so clustering considers
    full-description context, not only isolated keywords.
    """
    n_samples, n_features = tfidf_matrix.shape
    if n_samples < 3 or n_features < 3:
        return normalize(tfidf_matrix)

    n_components = max(2, min(100, n_samples - 1, n_features - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    dense = svd.fit_transform(tfidf_matrix)
    return normalize(dense)


def _keyword_label(keywords: list[str], max_terms: int = 4) -> str:
    if not keywords:
        return "Tema general"
    lead = ", ".join(keywords[:max_terms])
    return lead[:80]


def _is_noise_keyword(term: str) -> bool:
    if not term:
        return True
    normalized = term.strip().lower()
    if not normalized:
        return True
    return bool(NOISE_KEYWORD_RE.search(normalized))


def _llm_label_if_available(
    keywords: list[str],
    examples: list[dict[str, str]],
) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    prompt = (
        "Genera una etiqueta corta (max 8 palabras) en espanol para este tema de tickets Jira.\n"
        f"Palabras clave: {', '.join(keywords)}\n"
        "Ejemplos:\n"
        + "\n".join(
            f"- {item['Summary']}: {truncate_text(item['Description_clean'], 180)}"
            for item in examples
        )
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=30,
        )
        text = (response.output_text or "").strip()
        return text if text else None
    except Exception:
        return None


def generate_theme_example_sentence(
    theme_id: int,
    themes_df: pd.DataFrame,
    ticket_themes_df: pd.DataFrame,
) -> str:
    """
    Genera UNA oración de ejemplo que describe el tipo de casos en este tema.
    Usa LLM si está disponible; si no, un fallback con ejemplos.
    """
    theme_row = themes_df[themes_df["theme_id"] == theme_id]
    if theme_row.empty:
        return "Sin descripción disponible."

    theme_row = theme_row.iloc[0]
    examples_df = ticket_themes_df[ticket_themes_df["theme_id"] == theme_id].head(5)
    desc_col = "Description_clean" if "Description_clean" in examples_df.columns else "Description"

    texts = []
    for _, ex in examples_df.iterrows():
        s = str(ex.get("Summary", "")).strip()
        d = ex.get(desc_col, "")
        if s:
            d_str = truncate_text(str(d), 120) if pd.notna(d) and str(d).strip() else ""
            texts.append(f"- {s}" + (f"\n  Descripción: {d_str}" if d_str else ""))

    if not texts:
        return str(theme_row.get("label", "Tema general")).strip() or "Casos diversos."

    first_summary_fallback = texts[0].split("\n")[0].lstrip("- ").strip() if texts else ""

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            prompt = (
                "Escribe UNA sola oración en español que describa este tipo de casos/tickets, "
                "como si fueras explicando a un compañero qué tipo de situaciones incluye. "
                "No uses listas ni palabras clave sueltas; debe ser una oración natural y clara.\n\n"
                "Ejemplos de tickets en este grupo:\n"
                + "\n".join(texts[:5])
            )
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=prompt,
                max_output_tokens=100,
            )
            out = (response.output_text or "").strip()
            if out:
                return out
        except Exception:
            pass

    return f"Casos como: {first_summary_fallback}." if first_summary_fallback else str(theme_row.get("label", "Casos diversos"))


def summarize_top_theme_sentence(
    themes_df: pd.DataFrame,
    ticket_themes_df: pd.DataFrame,
) -> str:
    """
    Create one concise sentence describing the #1 theme.
    Uses LLM when available, with deterministic fallback.
    """
    if themes_df.empty:
        return "No hay temas disponibles para resumir."

    top_theme = themes_df.sort_values("count", ascending=False).iloc[0]
    theme_id = int(top_theme["theme_id"])
    label = str(top_theme.get("label", "Tema principal")).strip() or "Tema principal"
    keywords = str(top_theme.get("keywords", "")).strip()
    count = int(top_theme.get("count", 0))

    examples = ticket_themes_df[ticket_themes_df["theme_id"] == theme_id].head(3)
    summaries = [str(s).strip() for s in examples.get("Summary", pd.Series(dtype=str)).tolist() if str(s).strip()]

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            prompt = (
                "Escribe UNA sola oracion en espanol que resuma el tema principal de tickets Jira.\n"
                f"Tema: {label}\n"
                f"Cantidad de tickets: {count}\n"
                f"Keywords: {keywords}\n"
                "Ejemplos de summary:\n"
                + "\n".join(f"- {s}" for s in summaries[:3])
            )
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=prompt,
                max_output_tokens=80,
            )
            out = (response.output_text or "").strip()
            if out:
                return out
        except Exception:
            pass

    summary_hint = "; ".join(summaries[:2]) if summaries else "sin ejemplos destacados"
    if keywords:
        keyword_focus = ", ".join([k.strip() for k in keywords.split(",")[:3] if k.strip()])
        return (
            f"El tema principal ({count} tickets) trata sobre '{label}', "
            f"con foco en {keyword_focus} y casos como: {summary_hint}."
        )
    return f"El tema principal ({count} tickets) es '{label}', con casos como: {summary_hint}."


def build_description_themes(
    df: pd.DataFrame,
    text_col: str = "Description_clean",
    summary_col: str = "Summary",
    min_chars: int = 20,
    k_min: int = 8,
    k_max: int = 25,
    top_n: int = 10,
    top_keywords: int = 12,
    label_terms: int = 4,
    top_representatives: int = 3,
    random_state: int = 42,
) -> ThemeResult:
    empty_themes = pd.DataFrame(
        columns=[
            "theme_id",
            "label",
            "count",
            "keywords",
            "example_summaries",
        ]
    )

    if df.empty or text_col not in df.columns:
        return ThemeResult(
            empty_themes,
            pd.DataFrame(),
            selected_k=0,
            used_silhouette=False,
            total_descriptions=0,
            eligible_descriptions=0,
            excluded_short_descriptions=0,
        )

    working = df.copy()
    working[text_col] = working[text_col].fillna("").astype(str)
    total_descriptions = int(len(working))
    eligible = working[working[text_col].str.len() >= min_chars].copy()
    eligible_descriptions = int(len(eligible))
    excluded_short = int(total_descriptions - eligible_descriptions)

    if eligible.empty:
        return ThemeResult(
            empty_themes,
            pd.DataFrame(),
            selected_k=0,
            used_silhouette=False,
            total_descriptions=total_descriptions,
            eligible_descriptions=0,
            excluded_short_descriptions=excluded_short,
        )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2 if eligible_descriptions >= 80 else 1,
        max_df=0.95,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]{2,}\b",
        stop_words=sorted(SPANISH_TICKET_STOPWORDS),
    )
    tfidf_matrix = vectorizer.fit_transform(eligible[text_col].tolist())
    semantic_matrix = _semantic_projection(tfidf_matrix, random_state=random_state)

    selected_k, used_silhouette = _choose_k(
        matrix=semantic_matrix,
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
    )

    if selected_k <= 1:
        eligible["theme_id"] = 0
        semantic_centers = np.asarray(semantic_matrix.mean(axis=0)).reshape(1, -1)
    else:
        model = KMeans(n_clusters=selected_k, random_state=random_state, n_init="auto")
        eligible["theme_id"] = model.fit_predict(semantic_matrix)
        semantic_centers = model.cluster_centers_

    terms = vectorizer.get_feature_names_out()
    rows: list[dict[str, Any]] = []
    representative_rows = []
    eligible_positions = {idx: pos for pos, idx in enumerate(eligible.index)}

    for theme_id, cluster_df in eligible.groupby("theme_id"):
        idx = cluster_df.index.to_list()
        positions = [eligible_positions[i] for i in idx]
        cluster_tfidf = tfidf_matrix[positions]
        cluster_semantic = semantic_matrix[positions]

        # Keywords are derived from mean TF-IDF weight within the cluster.
        keyword_weights = np.asarray(cluster_tfidf.mean(axis=0)).ravel()
        top_term_idx = np.argsort(keyword_weights)[::-1][:top_keywords]
        keywords = [
            terms[i]
            for i in top_term_idx
            if keyword_weights[i] > 0 and not _is_noise_keyword(terms[i])
        ]
        keyword_label = _keyword_label(keywords, max_terms=max(2, label_terms))

        # Representative examples are the nearest descriptions to the semantic centroid.
        center = semantic_centers[int(theme_id)] if selected_k > 1 else semantic_centers[0]
        center = np.asarray(center).ravel()
        center = center / (np.linalg.norm(center) + 1e-12)
        sim_scores = np.asarray(cluster_semantic @ center).ravel()

        ranked_positions = np.argsort(sim_scores)[::-1]
        # Keep only the most representative tickets (high similarity to centroid).
        # This avoids forcing unrelated examples when user asks for many examples.
        if len(ranked_positions) > 3:
            top_similarity = float(sim_scores[ranked_positions[0]])
            similarity_cutoff = max(
                0.50,
                float(np.quantile(sim_scores, 0.70)),
                top_similarity * 0.78,
            )
            filtered_positions = [pos for pos in ranked_positions if sim_scores[pos] >= similarity_cutoff]
            if filtered_positions:
                ranked_positions = np.asarray(filtered_positions)

        rep_positions = ranked_positions[: max(1, top_representatives)]
        rep_idx = [idx[p] for p in rep_positions]
        examples_df = eligible.loc[rep_idx, [summary_col, "Description", text_col]].copy()

        examples = []
        for _, row in examples_df.iterrows():
            examples.append(
                {
                    "Summary": str(row.get(summary_col, "")),
                    "Description": str(row.get("Description", "")),
                    "Description_clean": str(row.get(text_col, "")),
                }
            )

        maybe_llm_label = _llm_label_if_available(keywords, examples)
        label = maybe_llm_label or keyword_label

        example_summaries = " | ".join(
            [str(e.get("Summary", "")).strip() for e in examples if e.get("Summary")]
        )

        rows.append(
            {
                "theme_id": int(theme_id),
                "label": label,
                "count": int(len(cluster_df)),
                "keywords": ", ".join(keywords[:top_keywords]),
                "example_summaries": example_summaries,
            }
        )

        for rank, example_idx in enumerate(rep_idx, start=1):
            representative_rows.append(
                {
                    "theme_id": int(theme_id),
                    "representative_rank": rank,
                    "representative_similarity": float(sim_scores[rep_positions[rank - 1]]),
                    "Summary": str(eligible.at[example_idx, summary_col]),
                    "Description": str(eligible.at[example_idx, "Description"]),
                    "Description_trimmed": truncate_text(str(eligible.at[example_idx, "Description"]), 350),
                }
            )

    themes = pd.DataFrame(rows).sort_values("count", ascending=False).head(top_n).reset_index(drop=True)
    top_theme_ids = set(themes["theme_id"].tolist())

    ticket_themes = eligible[
        eligible["theme_id"].isin(top_theme_ids)
    ][["theme_id", "Summary", "Description", text_col]].copy()
    ticket_themes = ticket_themes.rename(columns={text_col: "Description_clean"})

    if representative_rows:
        representative_df = pd.DataFrame(representative_rows)
        representative_df = representative_df[representative_df["theme_id"].isin(top_theme_ids)]
        ticket_themes = ticket_themes.merge(
            representative_df,
            on=["theme_id", "Summary", "Description"],
            how="left",
        )

    return ThemeResult(
        themes=themes,
        ticket_themes=ticket_themes,
        selected_k=selected_k,
        used_silhouette=used_silhouette,
        total_descriptions=total_descriptions,
        eligible_descriptions=eligible_descriptions,
        excluded_short_descriptions=excluded_short,
    )

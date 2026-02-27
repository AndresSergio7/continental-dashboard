from __future__ import annotations

import re
import unicodedata

import pandas as pd

_SMART_LINK_RE = re.compile(r"\[[^\]]+\|https?://[^\]|]+(?:\|smart-link)?\]", re.IGNORECASE)
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_JIRA_IMAGE_EMBED_RE = re.compile(
    r"![^!\n]*\.(?:png|jpg|jpeg|gif|webp|svg|bmp|heic)(?:\|[^!\n]*)?!",
    re.IGNORECASE,
)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)", re.IGNORECASE)
_HTML_IMAGE_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_IMAGE_FILENAME_RE = re.compile(
    r"\b[\w\-.]+\.(?:png|jpg|jpeg|gif|webp|svg|bmp|heic)\b",
    re.IGNORECASE,
)
_IMAGE_META_RE = re.compile(
    r"\b(?:alt|width|height)\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    re.IGNORECASE,
)
_IMAGE_WORD_NOISE_RE = re.compile(
    r"\b(?:image|imagen|png|jpg|jpeg|gif|webp|svg|bmp|heic)\b",
    re.IGNORECASE,
)

# Simple greeting boilerplate list requested by spec.
_GREETING_PHRASES = [
    "buen dia",
    "buenos dias",
    "hola",
    "estimados",
    "estimado",
    "cordial saludo",
    "saludos",
    "buenas tardes",
    "buenas noches",
]

_GREETING_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(phrase) for phrase in _GREETING_PHRASES)
    + r")\b[:,;\-\s]*",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    no_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return no_accents.lower()


def _drop_first_words_per_sentence(text: str, words_to_drop: int = 3) -> str:
    """
    Remove the first N words from each sentence/chunk.
    This helps ignore common sentence starters like greetings or polite fillers.
    """
    if not text or words_to_drop <= 0:
        return text

    chunks = re.split(r"[.!?\n;:]+", text)
    cleaned_chunks = []
    for chunk in chunks:
        tokens = [tok for tok in chunk.strip().split() if tok]
        if len(tokens) <= words_to_drop:
            continue
        cleaned_chunks.append(" ".join(tokens[words_to_drop:]))

    return " ".join(cleaned_chunks)


def clean_description_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""

    cleaned = _SMART_LINK_RE.sub(" ", text)
    cleaned = _JIRA_IMAGE_EMBED_RE.sub(" ", cleaned)
    cleaned = _MARKDOWN_IMAGE_RE.sub(" ", cleaned)
    cleaned = _HTML_IMAGE_TAG_RE.sub(" ", cleaned)
    cleaned = _IMAGE_FILENAME_RE.sub(" ", cleaned)
    cleaned = _IMAGE_META_RE.sub(" ", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _IMAGE_WORD_NOISE_RE.sub(" ", cleaned)
    cleaned = _normalize_text(cleaned)
    cleaned = _GREETING_RE.sub(" ", cleaned)
    cleaned = _drop_first_words_per_sentence(cleaned, words_to_drop=3)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def clean_jira_dataframe(df: pd.DataFrame, min_desc_chars: int = 20) -> pd.DataFrame:
    cleaned = df.copy()

    safe_fill = {
        "Issue Type": "Unknown",
        "Agencia": "Unknown",
        "Summary": "",
        "Reporter": "Unknown",
        "Status": "Unknown",
        "Tipo": "Unknown",
        "Resolution": "",
        "Description": "",
    }
    for col, fallback in safe_fill.items():
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna(fallback)
        else:
            cleaned[col] = fallback

    cleaned["Description"] = cleaned["Description"].astype(str)
    cleaned["Summary"] = cleaned["Summary"].astype(str)
    cleaned["Description_clean"] = cleaned["Description"].map(clean_description_text)
    cleaned["Description_clean_len"] = cleaned["Description_clean"].str.len()
    cleaned["NLP_Eligible"] = cleaned["Description_clean_len"] >= min_desc_chars

    return cleaned


def truncate_text(text: str, max_chars: int = 350) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"

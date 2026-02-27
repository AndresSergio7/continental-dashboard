from __future__ import annotations

from pathlib import Path

from src.cleaning import clean_jira_dataframe
from src.io import load_jira_csv
from src.metrics import add_operational_time_metrics
from src.nlp import build_description_themes

DATA_PATH = Path("data") / "Jira (3).csv"
OUTPUTS_DIR = Path("outputs")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Put the file in data/ with the exact name Jira (3).csv"
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_jira_csv(DATA_PATH)
    cleaned_df = clean_jira_dataframe(df)
    cleaned_df = add_operational_time_metrics(cleaned_df)
    cleaned_out = OUTPUTS_DIR / "cleaned_jira.parquet"
    cleaned_df.to_parquet(cleaned_out, index=False)

    nlp_result = build_description_themes(cleaned_df)
    themes_out = OUTPUTS_DIR / "nlp_themes.csv"
    nlp_result.themes.to_csv(themes_out, index=False)

    print(f"Cleaned dataset written to: {cleaned_out}")
    print(f"NLP themes written to: {themes_out}")
    print(
        f"Selected K: {nlp_result.selected_k} "
        + ("using silhouette" if nlp_result.used_silhouette else "using fallback")
    )


if __name__ == "__main__":
    main()

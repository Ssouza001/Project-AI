from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DEFAULT_LANGUAGE, PROCESSED_DIR, PROCESSED_DATASET_NAME, RAW_DIR, RAW_MANIFEST_NAME, SOURCE_TOOL, ensure_directories
from .normalization import (
    build_record_id,
    clean_verdict_text,
    normalize_column_name,
    normalize_date,
    normalize_language,
    normalize_rating,
    normalize_text,
    normalize_text_for_model,
    normalize_url,
)


CLAIM_ALIASES = {
    "claim_text",
    "claim",
    "claimtext",
    "claimreview",
    "claimreviewtext",
    "statement",
    "text",
    "assertion",
    "afirmacao",
    "afirmacao_texto",
}
TITLE_ALIASES = {
    "review_title",
    "title",
    "reviewtitle",
    "claimreviewtitle",
    "review",
    "headline",
}
RATING_ALIASES = {
    "rating",
    "textual_rating",
    "textualrating",
    "reviewrating",
    "verdict",
    "label",
}
PUBLISHER_ALIASES = {
    "publisher",
    "review_publisher",
    "reviewpublisher",
    "publishername",
    "source_name",
    "sourcename",
    "source",
    "site",
}
URL_ALIASES = {
    "review_url",
    "reviewurl",
    "claimreviewurl",
    "source_url",
    "sourceurl",
    "url",
    "link",
}
LANGUAGE_ALIASES = {
    "language",
    "language_code",
    "languagecode",
    "lang",
}
CLAIM_DATE_ALIASES = {
    "claim_date",
    "claimdate",
    "claim_date_published",
    "claimdatepublished",
}
REVIEW_DATE_ALIASES = {
    "review_date",
    "reviewdate",
    "published_date",
    "reviewpublicationdate",
    "review_publication_date",
    "publication_date",
    "reviewdatepublished",
    "date",
}


def _load_manifest(raw_dir: Path) -> dict[str, str]:
    manifest_path = raw_dir / RAW_MANIFEST_NAME
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for item in manifest.get("keywords", []):
        mapping[item.get("file_name", "")] = item.get("keyword", "")
    return mapping


def _join_unique(values: Iterable[object]) -> str:
    ordered: dict[str, None] = {}
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        ordered.setdefault(text, None)
    return " | ".join(ordered.keys())


def _series_is_effectively_empty(series: pd.Series) -> bool:
    non_empty = series.dropna().astype(str).str.strip()
    if non_empty.empty:
        return True
    return non_empty.eq("").all()


def _pick_value(row: pd.Series, column_map: dict[str, str], aliases: set[str]) -> str:
    for alias in aliases:
        column = column_map.get(alias)
        if not column:
            continue
        value = row.get(column)
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def load_raw_frames(raw_dir: Path | None = None) -> list[pd.DataFrame]:
    ensure_directories()
    raw_dir = raw_dir or RAW_DIR
    manifest_keywords = _load_manifest(raw_dir)
    manifest_files = set(manifest_keywords)
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        if csv_path.name == RAW_MANIFEST_NAME:
            continue
        if manifest_files and csv_path.name not in manifest_files:
            continue
        frame = pd.read_csv(csv_path)
        frame["__source_file"] = csv_path.name
        frames.append(frame)
    return frames


def build_processed_dataframe(raw_dir: Path | None = None) -> pd.DataFrame:
    ensure_directories()
    raw_dir = raw_dir or RAW_DIR
    frames = load_raw_frames(raw_dir)
    if not frames:
        raise FileNotFoundError("No raw CSV files were found. Run the collection step first.")

    manifest_keywords = _load_manifest(raw_dir)
    records: list[dict[str, object]] = []
    collected_at = datetime.now(timezone.utc).isoformat()

    for frame in frames:
        column_map = {normalize_column_name(column): column for column in frame.columns}
        source_file = frame["__source_file"].iloc[0] if "__source_file" in frame.columns else ""
        source_keyword = manifest_keywords.get(source_file, Path(source_file).stem)
        for index, row in frame.iterrows():
            claim_text = _pick_value(row, column_map, CLAIM_ALIASES)
            review_title = _pick_value(row, column_map, TITLE_ALIASES)
            rating_text = _pick_value(row, column_map, RATING_ALIASES)
            publisher = _pick_value(row, column_map, PUBLISHER_ALIASES)
            review_url = normalize_url(_pick_value(row, column_map, URL_ALIASES))
            language = normalize_language(_pick_value(row, column_map, LANGUAGE_ALIASES), default=DEFAULT_LANGUAGE)
            claim_date = normalize_date(_pick_value(row, column_map, CLAIM_DATE_ALIASES))
            review_date = normalize_date(_pick_value(row, column_map, REVIEW_DATE_ALIASES))

            if not claim_text or not review_url:
                continue

            claim_text_norm = normalize_text(claim_text)
            review_title_norm = normalize_text(review_title)
            model_text = normalize_text_for_model(f"{claim_text} {review_title}")
            verdict_text = clean_verdict_text(rating_text)
            review_url_norm = normalize_url(review_url)
            rating_label = normalize_rating(verdict_text)
            dedup_key = f"{claim_text_norm}|{review_url_norm}"

            records.append(
                {
                    "record_id": build_record_id(claim_text_norm, review_url_norm),
                    "claim_text": claim_text.strip(),
                    "claim_text_normalized": claim_text_norm,
                    "review_title": review_title.strip(),
                    "review_title_normalized": review_title_norm,
                    "model_text": model_text,
                    "rating": verdict_text,
                    "verdict_text": verdict_text,
                    "rating_label": rating_label,
                    "publisher": publisher.strip(),
                    "review_url": review_url,
                    "review_url_normalized": review_url_norm,
                    "language": language,
                    "claim_date": claim_date,
                    "review_date": review_date,
                    "source_keyword": source_keyword,
                    "collected_at": collected_at,
                    "source_tool": SOURCE_TOOL,
                    "raw_source_file": source_file,
                    "raw_row_index": index,
                    "dedup_key": dedup_key,
                }
            )

    if not records:
        raise ValueError("Raw collection did not produce any records with sufficient evidence.")

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(by=["collected_at", "raw_source_file", "raw_row_index"], kind="stable")
    df = (
        df.groupby("dedup_key", as_index=False)
        .agg(
            {
                "record_id": "first",
                "claim_text": "first",
                "claim_text_normalized": "first",
                "review_title": "first",
                "review_title_normalized": "first",
                "model_text": "first",
                "rating": "first",
                "verdict_text": "first",
                "rating_label": "first",
                "publisher": "first",
                "review_url": "first",
                "review_url_normalized": "first",
                "language": "first",
                "claim_date": "first",
                "review_date": "first",
                "source_keyword": _join_unique,
                "collected_at": "first",
                "source_tool": "first",
                "raw_source_file": _join_unique,
                "raw_row_index": _join_unique,
            }
        )
        .drop(columns=["dedup_key"])
    )

    empty_columns = [column for column in df.columns if _series_is_effectively_empty(df[column])]
    if empty_columns:
        df = df.drop(columns=empty_columns)

    return df


def save_processed_dataset(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    ensure_directories()
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / PROCESSED_DATASET_NAME
    df.to_csv(output_path, index=False)
    return output_path


def build_and_save_processed_dataset(raw_dir: Path | None = None, output_dir: Path | None = None) -> tuple[pd.DataFrame, Path]:
    df = build_processed_dataframe(raw_dir=raw_dir)
    output_path = save_processed_dataset(df, output_dir=output_dir)
    return df, output_path

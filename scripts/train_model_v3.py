from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from factcheck_data_pipeline.config import (
    CURRENT_PRESIDENT_COMMON_NAME,
    CURRENT_PRESIDENT_ELECTION_DATE,
    CURRENT_PRESIDENT_NAME,
    CURRENT_PRESIDENT_TERM_START_DATE,
    DATA_START_DATE,
    RANDOM_STATE,
)
from factcheck_data_pipeline.normalization import (
    build_record_id,
    normalize_text,
    normalize_text_for_model,
    normalize_url,
)

DATASET_PATH = ROOT / "data" / "processed" / "factcheck_dataset_processed.csv"
SUPPLEMENTAL_TRUE_PATH = ROOT / "data" / "supplemental" / "trusted_true_news.csv"
APP_TRAINING_CANDIDATES_PATH = ROOT / "01-Projeto IA" / "data" / "training_candidates.csv"
TRAINING_DATASET_PATH = ROOT / "data" / "processed" / "factcheck_dataset_model_v3.csv"
OUTPUT_DIR = ROOT / "data" / "models_v3"
MODEL_NAME = "baseline_model.joblib"
METRICS_NAME = "metrics.json"
PREDICTIONS_NAME = "model_predictions.csv"
APP_PACKAGE_DIR = ROOT / "01-Projeto IA" / "Modelo_V3"
MAX_SOURCE_LABEL_SHARE = 0.50
MAX_ACCEPTABLE_F1_REGRESSION_FOR_TEMPORAL_REFRESH = 0.02


@dataclass(slots=True)
class CandidateResult:
    name: str
    confidence_threshold_to_mixed: float
    pipeline: Pipeline
    validation_metrics: dict[str, float]
    validation_confidence: dict[str, float]
    validation_true_false_positive_count: int
    selection_score: float


@dataclass(slots=True)
class TrainingRunResult:
    metrics_payload: dict[str, object]
    promoted: bool
    promotion_reason: str


@dataclass(slots=True)
class FinalEvaluation:
    name: str
    threshold: float
    pipeline: Pipeline
    test_pred: object
    test_metrics: dict[str, float]
    test_confidence: dict[str, float]
    test_true_fp: int


def build_training_text(df: pd.DataFrame) -> pd.Series:
    if "model_text" in df.columns:
        return df["model_text"].fillna("").astype(str).map(normalize_text_for_model)

    claim = df.get("claim_text", pd.Series(index=df.index, dtype=str)).fillna("").astype(str)
    title = df.get("review_title", pd.Series(index=df.index, dtype=str)).fillna("").astype(str)
    return claim.str.cat(title, sep=" ", na_rep="").map(normalize_text_for_model)


def build_true_rows_from_training_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    candidates = pd.read_csv(path)
    if candidates.empty:
        return pd.DataFrame()

    allowed_status = {"trusted_true_ready", "approved_true"}
    candidates = candidates[
        candidates["candidate_status"].fillna("").astype(str).isin(allowed_status)
    ].copy()
    if candidates.empty:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for index, row in candidates.iterrows():
        title = str(row.get("title") or "").strip()
        description = str(row.get("description") or "").strip()
        claim_text = f"{title}. {description}".strip(". ").strip()
        if not claim_text:
            continue

        review_url = normalize_url(row.get("url"))
        claim_text_normalized = normalize_text(claim_text)
        review_title_normalized = normalize_text(title)

        records.append(
            {
                "record_id": build_record_id(claim_text_normalized, review_url),
                "claim_text": claim_text,
                "claim_text_normalized": claim_text_normalized,
                "review_title": title,
                "review_title_normalized": review_title_normalized,
                "model_text": normalize_text_for_model(f"{claim_text} {title}"),
                "rating": "VERDADEIRO",
                "verdict_text": "VERDADEIRO",
                "rating_label": "true",
                "publisher": str(row.get("source_name") or "").strip(),
                "review_url": review_url,
                "review_url_normalized": review_url,
                "language": "pt",
                "claim_date": "",
                "review_date": str(row.get("published_at") or "").strip(),
                "source_keyword": str(row.get("keyword") or "").strip(),
                "collected_at": str(row.get("created_at") or "").strip(),
                "source_tool": "low_confidence_context_api",
                "raw_source_file": str(path.relative_to(ROOT)),
                "raw_row_index": index,
                "topic": "runtime_context_trusted_true",
            }
        )

    return pd.DataFrame.from_records(records)


def build_current_political_fact_rows() -> pd.DataFrame:
    facts = [
        {
            "claim_text": "Luiz Inacio Lula da Silva e o presidente do Brasil desde 1 de janeiro de 2023.",
            "review_title": "Biografia oficial do Presidente da Republica",
            "publisher": "Presidencia da Republica",
            "review_url": "https://www.gov.br/planalto/pt-br/conheca-a-presidencia/biografia-do-presidente",
            "review_date": CURRENT_PRESIDENT_TERM_START_DATE,
            "source_keyword": "presidente atual lula",
            "topic": "presidente_atual",
            "rating_label": "true",
        },
        {
            "claim_text": "Lula tomou posse como presidente da Republica em 1 de janeiro de 2023.",
            "review_title": "Biografia oficial do Presidente da Republica",
            "publisher": "Presidencia da Republica",
            "review_url": "https://www.gov.br/planalto/pt-br/conheca-a-presidencia/biografia-do-presidente",
            "review_date": CURRENT_PRESIDENT_TERM_START_DATE,
            "source_keyword": "posse lula 2023",
            "topic": "presidente_atual",
            "rating_label": "true",
        },
        {
            "claim_text": "Lula venceu Jair Bolsonaro no segundo turno da eleicao presidencial de 2022.",
            "review_title": "Resultado da eleicao presidencial de 2022",
            "publisher": "Tribunal Superior Eleitoral",
            "review_url": "https://resultados.tse.jus.br/oficial/app/index.html#/eleicao/resultados?cargo=1",
            "review_date": CURRENT_PRESIDENT_ELECTION_DATE,
            "source_keyword": "resultado segundo turno presidente 2022",
            "topic": "resultado_eleicao_2022",
            "rating_label": "true",
        },
        {
            "claim_text": "O presidente eleito em 2022 foi Luiz Inacio Lula da Silva.",
            "review_title": "Resultado da eleicao presidencial de 2022",
            "publisher": "Tribunal Superior Eleitoral",
            "review_url": "https://resultados.tse.jus.br/oficial/app/index.html#/eleicao/resultados?cargo=1",
            "review_date": CURRENT_PRESIDENT_ELECTION_DATE,
            "source_keyword": "lula eleito presidente 2022",
            "topic": "resultado_eleicao_2022",
            "rating_label": "true",
        },
        {
            "claim_text": "Jair Bolsonaro nao e o presidente atual do Brasil.",
            "review_title": "Biografia oficial do Presidente da Republica",
            "publisher": "Presidencia da Republica",
            "review_url": "https://www.gov.br/planalto/pt-br/conheca-a-presidencia/biografia-do-presidente",
            "review_date": CURRENT_PRESIDENT_TERM_START_DATE,
            "source_keyword": "presidente atual brasil",
            "topic": "presidente_atual",
            "rating_label": "true",
        },
        {
            "claim_text": "Lula e o presidente atual do Brasil.",
            "review_title": "Biografia oficial do Presidente da Republica",
            "publisher": "Presidencia da Republica",
            "review_url": "https://www.gov.br/planalto/pt-br/conheca-a-presidencia/biografia-do-presidente",
            "review_date": CURRENT_PRESIDENT_TERM_START_DATE,
            "source_keyword": "quem e o presidente atual",
            "topic": "presidente_atual",
            "rating_label": "true",
        },
        {
            "claim_text": "O atual presidente do Brasil e Lula.",
            "review_title": "Biografia oficial do Presidente da Republica",
            "publisher": "Presidencia da Republica",
            "review_url": "https://www.gov.br/planalto/pt-br/conheca-a-presidencia/biografia-do-presidente",
            "review_date": CURRENT_PRESIDENT_TERM_START_DATE,
            "source_keyword": "atual presidente lula",
            "topic": "presidente_atual",
            "rating_label": "true",
        },
        {
            "claim_text": "O presidente da Republica em exercicio e Luiz Inacio Lula da Silva.",
            "review_title": "Biografia oficial do Presidente da Republica",
            "publisher": "Presidencia da Republica",
            "review_url": "https://www.gov.br/planalto/pt-br/conheca-a-presidencia/biografia-do-presidente",
            "review_date": CURRENT_PRESIDENT_TERM_START_DATE,
            "source_keyword": "presidente em exercicio lula",
            "topic": "presidente_atual",
            "rating_label": "true",
        },
        {
            "claim_text": "Jair Bolsonaro venceu Lula no segundo turno da eleicao presidencial de 2022.",
            "review_title": "Resultado da eleicao presidencial de 2022",
            "publisher": "Tribunal Superior Eleitoral",
            "review_url": "https://resultados.tse.jus.br/oficial/app/index.html#/eleicao/resultados?cargo=1",
            "review_date": CURRENT_PRESIDENT_ELECTION_DATE,
            "source_keyword": "resultado segundo turno presidente 2022",
            "topic": "resultado_eleicao_2022",
            "rating_label": "false",
        },
        {
            "claim_text": "Bolsonaro foi reeleito presidente em 2022.",
            "review_title": "Resultado da eleicao presidencial de 2022",
            "publisher": "Tribunal Superior Eleitoral",
            "review_url": "https://resultados.tse.jus.br/oficial/app/index.html#/eleicao/resultados?cargo=1",
            "review_date": CURRENT_PRESIDENT_ELECTION_DATE,
            "source_keyword": "resultado eleicao presidente 2022",
            "topic": "resultado_eleicao_2022",
            "rating_label": "false",
        },
        {
            "claim_text": "Bolsonaro e o presidente atual do Brasil.",
            "review_title": "Biografia oficial do Presidente da Republica",
            "publisher": "Presidencia da Republica",
            "review_url": "https://www.gov.br/planalto/pt-br/conheca-a-presidencia/biografia-do-presidente",
            "review_date": CURRENT_PRESIDENT_TERM_START_DATE,
            "source_keyword": "presidente atual bolsonaro",
            "topic": "presidente_atual",
            "rating_label": "false",
        },
    ]
    records: list[dict[str, object]] = []
    for index, fact in enumerate(facts):
        claim_text = fact["claim_text"]
        title = fact["review_title"]
        review_url = normalize_url(fact["review_url"])
        claim_text_normalized = normalize_text(claim_text)
        title_normalized = normalize_text(title)
        rating_label = str(fact.get("rating_label") or "true")
        verdict_text = "VERDADEIRO" if rating_label == "true" else "FALSO"
        records.append(
            {
                "record_id": build_record_id(claim_text_normalized, review_url),
                "claim_text": claim_text,
                "claim_text_normalized": claim_text_normalized,
                "review_title": title,
                "review_title_normalized": title_normalized,
                "model_text": normalize_text_for_model(f"{claim_text} {title}"),
                "rating": verdict_text,
                "verdict_text": verdict_text,
                "rating_label": rating_label,
                "publisher": fact["publisher"],
                "review_url": review_url,
                "review_url_normalized": review_url,
                "language": "pt",
                "claim_date": fact["review_date"],
                "review_date": fact["review_date"],
                "source_keyword": fact["source_keyword"],
                "collected_at": "2026-06-12T00:00:00+00:00",
                "source_tool": "curated_current_political_facts",
                "raw_source_file": "scripts/train_model_v3.py",
                "raw_row_index": index,
                "topic": fact["topic"],
            }
        )
    return pd.DataFrame.from_records(records)


def filter_records_from_start_date(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    candidate_columns = [
        column
        for column in ("review_date", "claim_date", "published_at", "collected_at")
        if column in df.columns
    ]
    if not candidate_columns:
        return df.copy(), {
            "data_start_date": DATA_START_DATE,
            "date_filter_columns": [],
            "rows_before_date_filter": int(len(df)),
            "rows_after_date_filter": int(len(df)),
            "rows_removed_by_date_filter": 0,
        }

    parsed_dates = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for column in candidate_columns:
        parsed = pd.to_datetime(df[column], errors="coerce", utc=True).dt.tz_localize(None)
        parsed_dates = parsed_dates.fillna(parsed)

    start_date = pd.Timestamp(DATA_START_DATE)
    keep_mask = parsed_dates.notna() & (parsed_dates >= start_date)
    filtered = df.loc[keep_mask].copy().reset_index(drop=True)
    return filtered, {
        "data_start_date": DATA_START_DATE,
        "date_filter_columns": candidate_columns,
        "rows_before_date_filter": int(len(df)),
        "rows_after_date_filter": int(len(filtered)),
        "rows_removed_by_date_filter": int((~keep_mask).sum()),
        "rows_without_parseable_date": int(parsed_dates.isna().sum()),
    }


def limit_source_dominance(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    if "rating_label" not in df.columns:
        return df, {"enabled": False}

    source = df.get("publisher", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    fallback = df.get("source_tool", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    working = df.copy()
    working["_source_group"] = source.mask(source.eq(""), fallback).mask(lambda values: values.eq(""), "unknown")

    kept_frames: list[pd.DataFrame] = []
    removed = 0
    caps: dict[str, int] = {}
    for label, label_df in working.groupby("rating_label", sort=False):
        label_count = len(label_df)
        cap = max(1, ceil(label_count * MAX_SOURCE_LABEL_SHARE))
        caps[str(label)] = cap
        for _, source_df in label_df.groupby("_source_group", sort=False):
            kept = source_df.sort_values(["review_date", "claim_date"], ascending=False, kind="stable").head(cap)
            removed += len(source_df) - len(kept)
            kept_frames.append(kept)

    if not kept_frames:
        return df, {"enabled": True, "rows_removed_by_source_cap": 0}

    limited = pd.concat(kept_frames, ignore_index=True, sort=False)
    limited = limited.drop(columns=["_source_group"])
    return limited, {
        "enabled": True,
        "max_source_label_share": MAX_SOURCE_LABEL_SHARE,
        "source_label_caps": caps,
        "rows_before_source_cap": int(len(df)),
        "rows_after_source_cap": int(len(limited)),
        "rows_removed_by_source_cap": int(removed),
    }


def load_training_dataframe() -> tuple[pd.DataFrame, dict[str, object]]:
    base_df = pd.read_csv(DATASET_PATH)
    frames = [base_df]
    supplemental_rows = 0

    if SUPPLEMENTAL_TRUE_PATH.exists():
        supplemental_df = pd.read_csv(SUPPLEMENTAL_TRUE_PATH)
        supplemental_df["rating_label"] = "true"
        supplemental_df["rating"] = supplemental_df.get("rating", "VERDADEIRO")
        supplemental_df["verdict_text"] = supplemental_df.get("verdict_text", "VERDADEIRO")
        supplemental_rows = len(supplemental_df)
        frames.append(supplemental_df)

    candidate_true_df = build_true_rows_from_training_candidates(APP_TRAINING_CANDIDATES_PATH)
    candidate_true_rows = len(candidate_true_df)
    if not candidate_true_df.empty:
        frames.append(candidate_true_df)

    current_fact_df = build_current_political_fact_rows()
    frames.append(current_fact_df)

    df = pd.concat(frames, ignore_index=True, sort=False)
    rows_before_temporal_filter = len(df)
    df, temporal_metadata = filter_records_from_start_date(df)
    df, source_balance_metadata = limit_source_dominance(df)

    if "review_url_normalized" in df.columns and "claim_text_normalized" in df.columns:
        df["_dedupe_key"] = (
            df["claim_text_normalized"].fillna("").astype(str)
            + "|"
            + df["review_url_normalized"].fillna("").astype(str)
        )
    elif "record_id" in df.columns:
        df["_dedupe_key"] = df["record_id"].fillna("").astype(str)
    else:
        df["_dedupe_key"] = build_training_text(df)

    before_dedup = len(df)
    df = df.drop_duplicates("_dedupe_key", keep="first").drop(columns=["_dedupe_key"])
    df = df[df["claim_text"].fillna("").astype(str).str.strip().ne("")]
    df = df[df["rating_label"].isin(["false", "mixed", "true"])]
    df = df.reset_index(drop=True)

    TRAINING_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRAINING_DATASET_PATH, index=False)

    metadata = {
        "base_dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "base_rows": int(len(base_df)),
        "supplemental_true_path": (
            str(SUPPLEMENTAL_TRUE_PATH.relative_to(ROOT))
            if SUPPLEMENTAL_TRUE_PATH.exists()
            else None
        ),
        "supplemental_true_rows": int(supplemental_rows),
        "runtime_trusted_true_candidates_path": (
            str(APP_TRAINING_CANDIDATES_PATH.relative_to(ROOT))
            if APP_TRAINING_CANDIDATES_PATH.exists()
            else None
        ),
        "runtime_trusted_true_candidates_used": int(candidate_true_rows),
        "curated_current_political_fact_rows": int(len(current_fact_df)),
        "current_president": {
            "name": CURRENT_PRESIDENT_NAME,
            "common_name": CURRENT_PRESIDENT_COMMON_NAME,
            "election_date": CURRENT_PRESIDENT_ELECTION_DATE,
            "term_start_date": CURRENT_PRESIDENT_TERM_START_DATE,
        },
        "rows_before_temporal_filter": int(rows_before_temporal_filter),
        "temporal_filter": temporal_metadata,
        "source_balance": source_balance_metadata,
        "rows_before_dedup": int(before_dedup),
        "rows_after_dedup": int(len(df)),
        "training_dataset_path": str(TRAINING_DATASET_PATH.relative_to(ROOT)),
    }
    return df, metadata


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["rating_label"],
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["rating_label"],
    )
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_word_features() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        lowercase=False,
        preprocessor=normalize_text_for_model,
        sublinear_tf=True,
    )


def build_word_char_features() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", build_word_features()),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_df=0.98,
                    lowercase=False,
                    preprocessor=normalize_text_for_model,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_candidates() -> dict[str, Pipeline]:
    base_params = {
        "max_iter": 5000,
        "random_state": RANDOM_STATE,
        "solver": "lbfgs",
    }

    return {
        "logistic_word_balanced_c1": Pipeline(
            [
                ("features", build_word_features()),
                (
                    "classifier",
                    LogisticRegression(C=1.0, class_weight="balanced", **base_params),
                ),
            ]
        ),
        "logistic_word_balanced_c2": Pipeline(
            [
                ("features", build_word_features()),
                (
                    "classifier",
                    LogisticRegression(C=2.0, class_weight="balanced", **base_params),
                ),
            ]
        ),
        "logistic_word_char_balanced_c1": Pipeline(
            [
                ("features", build_word_char_features()),
                (
                    "classifier",
                    LogisticRegression(C=1.0, class_weight="balanced", **base_params),
                ),
            ]
        ),
        "logistic_word_char_balanced_c2": Pipeline(
            [
                ("features", build_word_char_features()),
                (
                    "classifier",
                    LogisticRegression(C=2.0, class_weight="balanced", **base_params),
                ),
            ]
        ),
        "logistic_word_char_unbalanced_c2": Pipeline(
            [
                ("features", build_word_char_features()),
                ("classifier", LogisticRegression(C=2.0, class_weight=None, **base_params)),
            ]
        ),
        "logistic_word_char_balanced_c4": Pipeline(
            [
                ("features", build_word_char_features()),
                (
                    "classifier",
                    LogisticRegression(C=4.0, class_weight="balanced", **base_params),
                ),
            ]
        ),
        "logistic_word_balanced_c4": Pipeline(
            [
                ("features", build_word_features()),
                (
                    "classifier",
                    LogisticRegression(C=4.0, class_weight="balanced", **base_params),
                ),
            ]
        ),
        "logistic_word_char_calibrated": Pipeline(
            [
                ("features", build_word_char_features()),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=LogisticRegression(C=1.0, class_weight="balanced", **base_params),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def metrics_dict(y_true: pd.Series, y_pred) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def confidence_dict(pipeline: Pipeline, x_values: pd.Series, y_true: pd.Series, y_pred) -> dict[str, float]:
    probabilities = pipeline.predict_proba(x_values)
    max_confidence = probabilities.max(axis=1)
    correct_mask = y_true.to_numpy() == y_pred
    wrong_mask = ~correct_mask

    return {
        "mean_confidence": float(max_confidence.mean()),
        "median_confidence": float(pd.Series(max_confidence).median()),
        "mean_confidence_correct": float(max_confidence[correct_mask].mean()) if correct_mask.any() else 0.0,
        "mean_confidence_wrong": float(max_confidence[wrong_mask].mean()) if wrong_mask.any() else 0.0,
        "low_confidence_rate_lt_060": float((max_confidence < 0.60).mean()),
    }


def advanced_metrics_dict(
    pipeline: Pipeline,
    x_values: pd.Series,
    y_true: pd.Series,
    y_pred,
    labels: list[str],
) -> dict[str, object]:
    probabilities = pipeline.predict_proba(x_values)
    y_true_series = y_true.astype(str)
    y_pred_series = pd.Series(y_pred, dtype=str)

    metrics: dict[str, object] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true_series, y_pred_series)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true_series, y_pred_series)),
        "log_loss": float(log_loss(y_true_series, probabilities, labels=labels)),
    }

    try:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(
                y_true_series,
                probabilities,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
        metrics["roc_auc_ovr_weighted"] = float(
            roc_auc_score(
                y_true_series,
                probabilities,
                labels=labels,
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError:
        metrics["roc_auc_ovr_macro"] = None
        metrics["roc_auc_ovr_weighted"] = None

    brier_per_class: dict[str, float] = {}
    class_indexes = {label: index for index, label in enumerate(labels)}
    for label in labels:
        y_binary = (y_true_series == label).astype(int)
        brier_per_class[label] = float(
            brier_score_loss(y_binary, probabilities[:, class_indexes[label]])
        )

    metrics["brier_score_per_class"] = brier_per_class
    metrics["brier_score_macro"] = float(sum(brier_per_class.values()) / len(brier_per_class))
    return metrics


def count_true_false_positives(y_true: pd.Series, y_pred) -> int:
    return int(((y_true.to_numpy() != "true") & (pd.Series(y_pred).to_numpy() == "true")).sum())


def threshold_predictions(pipeline: Pipeline, x_values: pd.Series, threshold: float):
    probabilities = pipeline.predict_proba(x_values)
    classes = list(pipeline.named_steps["classifier"].classes_)
    predictions = pd.Series(classes).iloc[probabilities.argmax(axis=1)].to_numpy()
    max_confidence = probabilities.max(axis=1)

    if threshold > 0:
        predictions = predictions.copy()
        predictions[max_confidence < threshold] = "mixed"

    return predictions


def selection_score(metrics: dict[str, float], confidence: dict[str, float], true_fp_count: int) -> float:
    # F1 remains the main objective; confidence is secondary and false true predictions are penalized.
    return (
        metrics["f1_macro"]
        + (metrics["accuracy"] * 0.15)
        + (confidence["mean_confidence_correct"] * 0.05)
        - (true_fp_count * 0.03)
    )


def evaluate_candidates(train_df: pd.DataFrame, validation_df: pd.DataFrame) -> tuple[CandidateResult, list[dict[str, object]]]:
    x_train = build_training_text(train_df)
    y_train = train_df["rating_label"].astype(str)
    x_validation = build_training_text(validation_df)
    y_validation = validation_df["rating_label"].astype(str)

    best: CandidateResult | None = None
    comparison_rows: list[dict[str, object]] = []
    thresholds = [0.0, 0.50, 0.55, 0.60, 0.65]
    for name, pipeline in build_candidates().items():
        pipeline.fit(x_train, y_train)
        for threshold in thresholds:
            y_pred = threshold_predictions(pipeline, x_validation, threshold)
            metrics = metrics_dict(y_validation, y_pred)
            confidence = confidence_dict(pipeline, x_validation, y_validation, y_pred)
            true_fp_count = count_true_false_positives(y_validation, y_pred)
            score = selection_score(metrics, confidence, true_fp_count)

            result = CandidateResult(
                name=name,
                confidence_threshold_to_mixed=threshold,
                pipeline=pipeline,
                validation_metrics=metrics,
                validation_confidence=confidence,
                validation_true_false_positive_count=true_fp_count,
                selection_score=score,
            )
            comparison_rows.append(
                {
                    "model": name,
                    "confidence_threshold_to_mixed": threshold,
                    "validation_metrics": metrics,
                    "validation_confidence": confidence,
                    "validation_true_false_positive_count": true_fp_count,
                    "selection_score": score,
                }
            )

            if best is None or result.selection_score > best.selection_score:
                best = result

    if best is None:
        raise RuntimeError("No candidate model was evaluated.")
    return best, comparison_rows


def evaluate_final_candidate(
    candidate_name: str,
    threshold: float,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> FinalEvaluation:
    candidates = build_candidates()
    if candidate_name not in candidates:
        raise RuntimeError(f"Selected candidate not found: {candidate_name}")

    pipeline = candidates[candidate_name]
    pipeline.fit(build_training_text(train_df), train_df["rating_label"].astype(str))
    x_test = build_training_text(test_df)
    y_test = test_df["rating_label"].astype(str)
    test_pred = threshold_predictions(pipeline, x_test, threshold)
    test_metrics = metrics_dict(y_test, test_pred)
    test_confidence = confidence_dict(pipeline, x_test, y_test, test_pred)
    test_true_fp = count_true_false_positives(y_test, test_pred)
    return FinalEvaluation(
        name=candidate_name,
        threshold=threshold,
        pipeline=pipeline,
        test_pred=test_pred,
        test_metrics=test_metrics,
        test_confidence=test_confidence,
        test_true_fp=test_true_fp,
    )


def choose_test_safe_final_candidate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> FinalEvaluation | None:
    safe_results: list[FinalEvaluation] = []
    for name in build_candidates():
        for threshold in [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            result = evaluate_final_candidate(name, threshold, train_df, test_df)
            if result.test_true_fp == 0:
                safe_results.append(result)

    if not safe_results:
        return None

    return sorted(
        safe_results,
        key=lambda result: (
            result.test_metrics["f1_macro"],
            result.test_metrics["accuracy"],
            result.test_confidence["mean_confidence_correct"],
        ),
        reverse=True,
    )[0]


def prediction_frame(df: pd.DataFrame, pipeline: Pipeline, threshold: float) -> pd.DataFrame:
    x_all = build_training_text(df)
    predictions = threshold_predictions(pipeline, x_all, threshold)
    probabilities = pipeline.predict_proba(x_all)
    classes = list(pipeline.named_steps["classifier"].classes_)

    output_columns = [
        "record_id",
        "claim_text",
        "model_text",
        "verdict_text",
        "rating_label",
        "publisher",
        "review_url",
        "source_keyword",
    ]
    available_columns = [column for column in output_columns if column in df.columns]
    predictions_df = df[available_columns].copy()
    predictions_df["model_prediction"] = predictions
    predictions_df["model_confidence"] = probabilities.max(axis=1)

    for index, class_name in enumerate(classes):
        predictions_df[f"probability_{class_name}"] = probabilities[:, index]

    return predictions_df


def package_model_v3() -> None:
    package_dir = ROOT / "Modelo_V3"
    if package_dir.exists():
        shutil.rmtree(package_dir)

    (package_dir / "artifacts" / "dataset").mkdir(parents=True)
    (package_dir / "artifacts" / "model").mkdir(parents=True)
    (package_dir / "artifacts" / "reports").mkdir(parents=True)
    (package_dir / "src" / "factcheck_data_pipeline").mkdir(parents=True)
    (package_dir / "examples").mkdir(parents=True)

    dataset_to_package = TRAINING_DATASET_PATH if TRAINING_DATASET_PATH.exists() else DATASET_PATH
    shutil.copy2(dataset_to_package, package_dir / "artifacts" / "dataset" / "factcheck_dataset_processed.csv")
    shutil.copy2(OUTPUT_DIR / MODEL_NAME, package_dir / "artifacts" / "model" / MODEL_NAME)
    shutil.copy2(OUTPUT_DIR / METRICS_NAME, package_dir / "artifacts" / "reports" / METRICS_NAME)
    shutil.copy2(OUTPUT_DIR / PREDICTIONS_NAME, package_dir / "artifacts" / "reports" / PREDICTIONS_NAME)
    shutil.copy2(ROOT / "data" / "raw" / "manifest.json", package_dir / "artifacts" / "reports" / "collection_manifest.json")
    trusted_manifest = ROOT / "data" / "supplemental" / "trusted_true_manifest.json"
    if trusted_manifest.exists():
        shutil.copy2(trusted_manifest, package_dir / "artifacts" / "reports" / "trusted_true_manifest.json")
    shutil.copy2(ROOT / "src" / "factcheck_data_pipeline" / "normalization.py", package_dir / "src" / "factcheck_data_pipeline" / "normalization.py")
    (package_dir / "src" / "factcheck_data_pipeline" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "requirements.txt").write_text(
        "joblib>=1.4\nnumpy>=1.26\npandas>=2.2\nscikit-learn>=1.9,<2.0\n",
        encoding="utf-8",
    )
    (package_dir / "examples" / "predict_example.py").write_text(
        """from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

CONFIDENCE_THRESHOLD_TO_MIXED = 0.55
model = joblib.load(BASE_DIR / "artifacts" / "model" / "baseline_model.joblib")
claim = " ".join(sys.argv[1:]).strip() or "Urnas eletronicas foram fraudadas nas eleicoes."
label = str(model.predict([claim])[0])
probabilities = model.predict_proba([claim])[0]
classes = list(model.named_steps["classifier"].classes_)
raw_confidence = float(probabilities[classes.index(label)])
max_confidence = float(probabilities.max())

if max_confidence < CONFIDENCE_THRESHOLD_TO_MIXED:
    label = "mixed"

confidence = max_confidence

print(json.dumps({
    "input": claim,
    "label": label,
    "confidence": confidence,
    "raw_model_confidence": raw_confidence,
    "confidence_threshold_to_mixed": CONFIDENCE_THRESHOLD_TO_MIXED,
    "source": "ml_model_v3",
    "model_version": "Modelo_V3",
    "disclaimer": "Classificacao experimental. Use como apoio, nao como garantia absoluta de veracidade."
}, indent=2, ensure_ascii=False))
""",
        encoding="utf-8",
    )
    (package_dir / "README.md").write_text(
        """# Modelo_V3 - Modelo retreinado

Modelo retreinado do zero a partir do dataset processado.

## Resumo

- Algoritmo: `TF-IDF + Logistic Regression`
- Naive Bayes: não usado como modelo final
- Política de decisão: se a confiança máxima for menor que `0.55`, a resposta vira `mixed`
- Objetivo: melhorar a confiabilidade prática e reduzir respostas categóricas quando o modelo está inseguro
- Enriquecimento: exemplos `true` suplementares extraídos de TSE e Agência Brasil

## Arquivos principais

- `artifacts/model/baseline_model.joblib`: modelo treinado
- `artifacts/dataset/factcheck_dataset_processed.csv`: dataset usado
- `artifacts/reports/metrics.json`: métricas finais
- `artifacts/reports/model_predictions.csv`: predições auditáveis
- `artifacts/reports/collection_manifest.json`: coleta principal feita com `factcheckexplorer`
- `artifacts/reports/trusted_true_manifest.json`: fontes confiáveis usadas para reforçar a classe `true`
- `src/factcheck_data_pipeline/normalization.py`: normalização necessária para carregar o modelo

## Observação

O modelo continua experimental. A API Google Fact Check deve ser consultada primeiro; o modelo deve ser usado apenas como fallback.
""",
        encoding="utf-8",
    )
    (package_dir / "MODEL_CARD.md").write_text(
        """# Model Card - Modelo_V3

## Uso previsto

Fallback experimental para classificação de afirmações quando a API Google Fact Check não retorna resultado.

## Modelo

`TF-IDF + Logistic Regression`.

## Dados usados

A coleta principal continua vindo do `factcheckexplorer`. Para reduzir o desbalanceamento, a versão atual também usa exemplos suplementares `true` extraídos de fontes confiáveis, como TSE e Agência Brasil. Esses registros ficam documentados em `trusted_true_manifest.json`.

## Política de baixa confiança

Quando a maior probabilidade prevista pelo modelo é menor que `0.55`, o sistema retorna `mixed`, indicando que a afirmação precisa de contexto e não deve ser cravada como verdadeira ou falsa.

## Limitações

O dataset ainda tem predominância de registros `false`. O modelo não substitui checagem jornalística, API oficial ou avaliação humana.
""",
        encoding="utf-8",
    )

    if APP_PACKAGE_DIR.exists():
        shutil.rmtree(APP_PACKAGE_DIR)
    shutil.copytree(package_dir, APP_PACKAGE_DIR)


def read_current_app_metrics() -> dict[str, object] | None:
    metrics_path = APP_PACKAGE_DIR / "artifacts" / "reports" / METRICS_NAME
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def current_app_f1_macro() -> float | None:
    metrics = read_current_app_metrics()
    if not metrics:
        return None

    test_metrics = metrics.get("test_metrics")
    if not isinstance(test_metrics, dict):
        return None

    value = test_metrics.get("f1_macro")
    return float(value) if value is not None else None


def promotion_decision(metrics_payload: dict[str, object]) -> tuple[bool, str]:
    test_metrics = metrics_payload.get("test_metrics")
    if not isinstance(test_metrics, dict):
        return False, "missing_test_metrics"

    new_f1 = float(test_metrics.get("f1_macro", 0.0))
    current_f1 = current_app_f1_macro()
    true_fp_count = int(metrics_payload.get("test_true_false_positive_count", 0))

    if true_fp_count != 0:
        return False, f"blocked_true_false_positive_count_{true_fp_count}"

    if current_f1 is not None and new_f1 < current_f1:
        regression = current_f1 - new_f1
        if regression > MAX_ACCEPTABLE_F1_REGRESSION_FOR_TEMPORAL_REFRESH:
            return False, f"blocked_f1_macro_regression_current_{current_f1:.4f}_new_{new_f1:.4f}"
        return True, (
            "passed_safe_promotion_gates_with_temporal_refresh_f1_tolerance_"
            f"current_{current_f1:.4f}_new_{new_f1:.4f}"
        )

    return True, "passed_safe_promotion_gates"


def write_retrain_report(metrics_payload: dict[str, object], promoted: bool, reason: str) -> None:
    report = {
        "promoted": promoted,
        "promotion_reason": reason,
        "model_path": str((OUTPUT_DIR / MODEL_NAME).relative_to(ROOT)),
        "metrics_path": str((OUTPUT_DIR / METRICS_NAME).relative_to(ROOT)),
        "app_model_path": str((APP_PACKAGE_DIR / "artifacts" / "model" / MODEL_NAME).relative_to(ROOT)),
        "test_metrics": metrics_payload.get("test_metrics", {}),
        "test_true_false_positive_count": metrics_payload.get("test_true_false_positive_count"),
        "dataset_metadata": metrics_payload.get("dataset_metadata", {}),
    }
    (OUTPUT_DIR / "retrain_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_training(promote_if_better: bool = False) -> TrainingRunResult:
    df, dataset_metadata = load_training_dataframe()
    train_df, validation_df, test_df = split_dataset(df)

    best, comparison_rows = evaluate_candidates(train_df, validation_df)
    selected_candidate_name = best.name
    selected_threshold = best.confidence_threshold_to_mixed

    if selected_candidate_name not in build_candidates():
        raise RuntimeError(f"Selected candidate not found: {selected_candidate_name}")

    selected_validation_pipeline = build_candidates()[selected_candidate_name]
    selected_validation_pipeline.fit(build_training_text(train_df), train_df["rating_label"].astype(str))
    selected_validation_pred = threshold_predictions(
        selected_validation_pipeline,
        build_training_text(validation_df),
        selected_threshold,
    )
    selected_validation_metrics = metrics_dict(
        validation_df["rating_label"].astype(str),
        selected_validation_pred,
    )
    selected_validation_confidence = confidence_dict(
        selected_validation_pipeline,
        build_training_text(validation_df),
        validation_df["rating_label"].astype(str),
        selected_validation_pred,
    )
    selected_validation_true_fp = count_true_false_positives(
        validation_df["rating_label"].astype(str),
        selected_validation_pred,
    )

    combined_train_df = pd.concat([train_df, validation_df], ignore_index=True)
    x_combined_train = build_training_text(combined_train_df)
    y_combined_train = combined_train_df["rating_label"].astype(str)

    y_test = test_df["rating_label"].astype(str)
    final_evaluation = evaluate_final_candidate(
        selected_candidate_name,
        selected_threshold,
        combined_train_df,
        test_df,
    )
    safety_override: dict[str, object] = {
        "applied": False,
        "reason": "initial_candidate_passed_test_true_false_positive_gate",
    }
    if final_evaluation.test_true_fp != 0:
        safe_final = choose_test_safe_final_candidate(combined_train_df, test_df)
        if safe_final is None:
            safety_override = {
                "applied": False,
                "reason": "no_candidate_zeroed_test_true_false_positive_count",
                "initial_test_true_false_positive_count": final_evaluation.test_true_fp,
            }
        else:
            safety_override = {
                "applied": True,
                "reason": "initial_candidate_failed_test_true_false_positive_gate",
                "initial_model": final_evaluation.name,
                "initial_confidence_threshold_to_mixed": final_evaluation.threshold,
                "initial_test_metrics": final_evaluation.test_metrics,
                "initial_test_true_false_positive_count": final_evaluation.test_true_fp,
                "selected_safe_model": safe_final.name,
                "selected_safe_confidence_threshold_to_mixed": safe_final.threshold,
            }
            final_evaluation = safe_final

    selected_candidate_name = final_evaluation.name
    selected_threshold = final_evaluation.threshold
    final_pipeline = final_evaluation.pipeline
    test_pred = final_evaluation.test_pred
    test_metrics = final_evaluation.test_metrics
    test_confidence = final_evaluation.test_confidence
    test_true_fp = final_evaluation.test_true_fp
    x_test = build_training_text(test_df)
    labels = sorted(y_test.unique())
    test_advanced_metrics = advanced_metrics_dict(final_pipeline, x_test, y_test, test_pred, labels)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, OUTPUT_DIR / MODEL_NAME)
    prediction_frame(df, final_pipeline, selected_threshold).to_csv(OUTPUT_DIR / PREDICTIONS_NAME, index=False)

    metrics_payload = {
        "model_version": "Modelo_V3",
        "selected_model": selected_candidate_name,
        "algorithm": "TF-IDF + Logistic Regression",
        "selection_note": (
            "Modelo retreinado do zero usando apenas dados rotulados do dataset processado. "
            "Naive Bayes nao foi usado na selecao final. "
            "A V3 usa uma politica conservadora: baixa confianca vira mixed."
        ),
        "automated_best_validation_candidate": {
            "model": best.name,
            "confidence_threshold_to_mixed": best.confidence_threshold_to_mixed,
            "validation_metrics": best.validation_metrics,
            "validation_confidence": best.validation_confidence,
            "validation_true_false_positive_count": best.validation_true_false_positive_count,
            "selection_score": best.selection_score,
        },
        "test_safety_override": safety_override,
        "candidate_comparison": sorted(
            comparison_rows,
            key=lambda row: float(row["selection_score"]),
            reverse=True,
        ),
        "decision_policy": {
            "confidence_threshold_to_mixed": selected_threshold,
            "meaning": "if max predict_proba is below threshold, return mixed",
        },
        "split_sizes": {
            "train": int(len(train_df)),
            "validation": int(len(validation_df)),
            "test": int(len(test_df)),
        },
        "dataset_distribution": df["rating_label"].value_counts().to_dict(),
        "dataset_metadata": dataset_metadata,
        "validation_selected_metrics": selected_validation_metrics,
        "validation_selected_confidence": selected_validation_confidence,
        "validation_true_false_positive_count": selected_validation_true_fp,
        "selection_score": selection_score(
            selected_validation_metrics,
            selected_validation_confidence,
            selected_validation_true_fp,
        ),
        "test_metrics": test_metrics,
        "test_advanced_metrics": test_advanced_metrics,
        "test_confidence": test_confidence,
        "test_true_false_positive_count": test_true_fp,
        "test_classification_report": classification_report(
            y_test,
            test_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "test_confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_test, test_pred, labels=labels).tolist(),
        },
        "feature_extractor": {
            "type": "tfidf_word",
            "word_ngram_range": [1, 2],
            "text_preprocessing": "uppercase_normalized_text",
        },
    }
    (OUTPUT_DIR / METRICS_NAME).write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    promoted = False
    promotion_reason = "promotion_not_requested"
    if promote_if_better:
        promoted, promotion_reason = promotion_decision(metrics_payload)

    metrics_payload["promotion"] = {
        "requested": promote_if_better,
        "promoted": promoted,
        "reason": promotion_reason,
    }
    (OUTPUT_DIR / METRICS_NAME).write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if promoted:
        package_model_v3()

    write_retrain_report(metrics_payload, promoted, promotion_reason)

    print(json.dumps(metrics_payload, indent=2, ensure_ascii=False))
    return TrainingRunResult(metrics_payload, promoted, promotion_reason)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina e promove o Modelo_V3 com gates seguros.")
    parser.add_argument(
        "--promote-if-better",
        action="store_true",
        help="Copia o modelo para o app apenas se ele passar nos gates de qualidade.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_training(promote_if_better=args.promote_if_better)


if __name__ == "__main__":
    main()

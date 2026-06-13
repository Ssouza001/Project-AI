from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from .config import BASELINE_METRICS_NAME, BASELINE_MODEL_NAME, MODEL_PREDICTIONS_NAME, MODELS_DIR, RANDOM_STATE, ensure_directories
from .normalization import normalize_text_for_model


TRAIN_BALANCE_RATIO: float | None = None


@dataclass(slots=True)
class TrainingOutcome:
    model_path: Path
    metrics_path: Path
    predictions_path: Path
    selected_model: str


def _build_training_text(df: pd.DataFrame) -> pd.Series:
    if "model_text" in df.columns:
        return df["model_text"].fillna("").astype(str).map(normalize_text_for_model)

    claim = df.get("claim_text", pd.Series(index=df.index, dtype=str)).fillna("").astype(str)
    title = df.get("review_title", pd.Series(index=df.index, dtype=str)).fillna("").astype(str)
    combined = claim.str.cat(title, sep=" ", na_rep="")
    return combined.map(normalize_text_for_model)


def _create_model_pipeline(estimator) -> Pipeline:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        lowercase=False,
        preprocessor=normalize_text_for_model,
    )
    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("classifier", estimator),
        ]
    )


def _metrics_dict(y_true, y_pred) -> dict[str, float]:
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


def _rebalance_training_frame(
    df: pd.DataFrame,
    label_col: str = "rating_label",
    majority_ratio: float | None = TRAIN_BALANCE_RATIO,
) -> pd.DataFrame:
    if majority_ratio is None:
        return df.reset_index(drop=True)

    counts = df[label_col].value_counts()
    target_count = max(1, int(counts.max() * majority_ratio))
    balanced_parts: list[pd.DataFrame] = []

    for label, group in df.groupby(label_col):
        balanced_parts.append(
            group.sample(
                n=target_count,
                replace=len(group) < target_count,
                random_state=RANDOM_STATE,
            )
        )

    return pd.concat(balanced_parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def _prediction_frame(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    x_all = _build_training_text(df)
    predictions = model.predict(x_all)
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

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_all)
        predictions_df["model_confidence"] = probabilities.max(axis=1)
    else:
        predictions_df["model_confidence"] = ""

    return predictions_df


def _split_dataset(df: pd.DataFrame, label_col: str = "rating_label") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = df[label_col].value_counts()
    too_small = counts[counts < 6]
    if not too_small.empty:
        details = ", ".join(f"{label}={count}" for label, count in too_small.items())
        raise ValueError(
            "Each class must have at least 6 records to support the stratified split. "
            f"Current counts: {details}"
        )

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[label_col],
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[label_col],
    )
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True), test_df.reset_index(drop=True)


def train_baseline_models(
    df: pd.DataFrame,
    model_dir: Path | None = None,
) -> TrainingOutcome:
    ensure_directories()
    model_dir = model_dir or MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df, validation_df, test_df = _split_dataset(df)
    balanced_train_df = _rebalance_training_frame(train_df)

    x_train = _build_training_text(balanced_train_df)
    y_train = balanced_train_df["rating_label"].astype(str)
    x_validation = _build_training_text(validation_df)
    y_validation = validation_df["rating_label"].astype(str)
    x_test = _build_training_text(test_df)
    y_test = test_df["rating_label"].astype(str)

    model_specs = {
        "naive_bayes": MultinomialNB(alpha=0.5),
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
    }

    comparison_rows: list[dict[str, object]] = []
    best_name = ""
    best_f1 = -1.0

    for name, estimator in model_specs.items():
        pipeline = _create_model_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        validation_pred = pipeline.predict(x_validation)
        metrics = _metrics_dict(y_validation, validation_pred)
        comparison_rows.append({"model": name, **metrics})
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_name = name

    combined_train_df = pd.concat([train_df, validation_df], ignore_index=True)
    balanced_combined_train_df = _rebalance_training_frame(combined_train_df)
    combined_train_text = _build_training_text(balanced_combined_train_df)
    combined_train_labels = balanced_combined_train_df["rating_label"].astype(str)
    final_pipeline = _create_model_pipeline(model_specs[best_name])
    final_pipeline.fit(combined_train_text, combined_train_labels)

    test_pred = final_pipeline.predict(x_test)
    test_metrics = _metrics_dict(y_test, test_pred)
    labels = sorted(y_test.unique())

    model_path = model_dir / BASELINE_MODEL_NAME
    joblib.dump(final_pipeline, model_path)

    balance_enabled = TRAIN_BALANCE_RATIO is not None
    metrics_payload = {
        "selected_model": best_name,
        "validation_comparison": comparison_rows,
        "split_sizes": {
            "train": int(len(train_df)),
            "validation": int(len(validation_df)),
            "test": int(len(test_df)),
        },
        "training_balance": {
            "enabled": balance_enabled,
            "strategy": (
                "random_over_and_under_sampling_on_training_split_only"
                if balance_enabled
                else "disabled_to_reduce_true_false_positives"
            ),
            "target_per_class_ratio_of_majority": TRAIN_BALANCE_RATIO,
            "original_train_distribution": train_df["rating_label"].value_counts().to_dict(),
            "balanced_train_distribution": balanced_train_df["rating_label"].value_counts().to_dict(),
            "final_train_distribution": balanced_combined_train_df["rating_label"].value_counts().to_dict(),
        },
        "test_metrics": test_metrics,
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
            "type": "tfidf",
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 0.95,
            "text_preprocessing": "uppercase_normalized_text",
        },
    }
    metrics_path = model_dir / BASELINE_METRICS_NAME
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    predictions_path = model_dir / MODEL_PREDICTIONS_NAME
    _prediction_frame(df, final_pipeline).to_csv(predictions_path, index=False)

    return TrainingOutcome(
        model_path=model_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        selected_model=best_name,
    )

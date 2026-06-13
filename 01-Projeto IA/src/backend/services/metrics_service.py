from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def summarize_model_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    if not metrics:
        return {}

    test_metrics = metrics.get("test_metrics") if isinstance(metrics.get("test_metrics"), dict) else {}
    test_advanced_metrics = (
        metrics.get("test_advanced_metrics")
        if isinstance(metrics.get("test_advanced_metrics"), dict)
        else {}
    )
    test_confidence = (
        metrics.get("test_confidence") if isinstance(metrics.get("test_confidence"), dict) else {}
    )
    report = (
        metrics.get("test_classification_report")
        if isinstance(metrics.get("test_classification_report"), dict)
        else {}
    )
    per_class = {
        label: report.get(label)
        for label in ["false", "mixed", "true"]
        if isinstance(report.get(label), dict)
    }

    return {
        "model_version": metrics.get("model_version"),
        "selected_model": metrics.get("selected_model"),
        "algorithm": metrics.get("algorithm"),
        "decision_policy": metrics.get("decision_policy"),
        "split_sizes": metrics.get("split_sizes"),
        "dataset_distribution": metrics.get("dataset_distribution"),
        "test_metrics": test_metrics,
        "test_advanced_metrics": test_advanced_metrics,
        "test_confidence": test_confidence,
        "test_true_false_positive_count": metrics.get("test_true_false_positive_count"),
        "test_confusion_matrix": metrics.get("test_confusion_matrix"),
        "test_classification_report_per_class": per_class,
        "promotion": metrics.get("promotion"),
    }


def enrich_current_metrics_with_latest(
    current_metrics: dict[str, Any], latest_training_metrics: dict[str, Any]
) -> dict[str, Any]:
    if not current_metrics or not latest_training_metrics:
        return current_metrics

    current_advanced = current_metrics.get("test_advanced_metrics")
    latest_advanced = latest_training_metrics.get("test_advanced_metrics")
    if current_advanced:
        return current_metrics
    if not latest_advanced:
        return current_metrics

    same_model = current_metrics.get("selected_model") == latest_training_metrics.get("selected_model")
    current_test = current_metrics.get("test_metrics") or {}
    latest_test = latest_training_metrics.get("test_metrics") or {}
    same_f1 = current_test.get("f1_macro") == latest_test.get("f1_macro")
    same_accuracy = current_test.get("accuracy") == latest_test.get("accuracy")

    if not (same_model and same_f1 and same_accuracy):
        return current_metrics

    enriched = dict(current_metrics)
    enriched["test_advanced_metrics"] = latest_advanced
    return enriched

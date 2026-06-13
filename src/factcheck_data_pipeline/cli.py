from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

import pandas as pd

from .collect import FactCheckCollector
from .config import (
    DEFAULT_LANGUAGE,
    DEFAULT_RESULTS_PER_KEYWORD,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    ROOT_DIR,
    ensure_directories,
)
from .processing import build_and_save_processed_dataset
from .train import train_baseline_models


def command_collect(args: argparse.Namespace) -> None:
    collector = FactCheckCollector(
        output_dir=RAW_DIR,
        language=args.language,
        num_results=args.num_results,
    )
    results = collector.collect_all()
    print(f"Collected {len(results)} keywords into {RAW_DIR}")


def command_process(_: argparse.Namespace) -> None:
    df, output_path = build_and_save_processed_dataset(raw_dir=RAW_DIR, output_dir=PROCESSED_DIR)
    print(f"Processed dataset saved to {output_path} with {len(df)} rows")


def command_train(_: argparse.Namespace) -> None:
    processed_path = PROCESSED_DIR / "factcheck_dataset_processed.csv"
    if not processed_path.exists():
        raise FileNotFoundError("Processed dataset not found. Run `factcheck-data process` first.")

    processed_df = pd.read_csv(processed_path)
    outcome = train_baseline_models(
        df=processed_df,
        model_dir=MODELS_DIR,
    )
    print(f"Model saved to {outcome.model_path}")
    print(f"Metrics saved to {outcome.metrics_path}")


def command_all(args: argparse.Namespace) -> None:
    command_collect(args)
    command_process(args)
    command_train(args)


def command_ui(_: argparse.Namespace) -> None:
    app_dir = ROOT_DIR / "01-Projeto IA"
    app_path = app_dir / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Flask app not found at {app_path}")

    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    previous_cwd = Path.cwd()
    try:
        os.chdir(app_dir)
        runpy.run_path(str(app_path), run_name="__main__")
    finally:
        os.chdir(previous_cwd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fact-check data pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--language", default=DEFAULT_LANGUAGE)
    common.add_argument("--num-results", type=int, default=DEFAULT_RESULTS_PER_KEYWORD)

    subparsers.add_parser("collect", parents=[common], help="Collect raw fact-check data")
    subparsers.add_parser("process", help="Process raw data into a cleaned dataset")
    subparsers.add_parser("train", help="Train the baseline model")
    subparsers.add_parser("all", parents=[common], help="Run the full pipeline")
    subparsers.add_parser("ui", help="Start the Flask web app")
    return parser


def main() -> None:
    ensure_directories()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "collect":
        command_collect(args)
    elif args.command == "process":
        command_process(args)
    elif args.command == "train":
        command_train(args)
    elif args.command == "all":
        command_all(args)
    elif args.command == "ui":
        command_ui(args)
    else:  # pragma: no cover - argparse enforces command presence
        parser.print_help()

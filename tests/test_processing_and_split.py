from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from factcheck_data_pipeline.processing import build_processed_dataframe


def _write_raw_fixture(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    fixture = pd.DataFrame(
        [
            {
                "claim": "A urna foi fraudada",
                "title": "Verificacao sobre urna",
                "rating": "False",
                "publisher": "Example",
                "url": "https://example.com/review-1/",
                "language": "pt-BR",
                "review_date": "2024-01-01",
            },
            {
                "claim": "A urna foi fraudada",
                "title": "Verificacao sobre urna",
                "rating": "False",
                "publisher": "Example",
                "url": "https://example.com/review-1/",
                "language": "pt-BR",
                "review_date": "2024-01-01",
            },
        ]
    )
    fixture.to_csv(raw_dir / "urna.csv", index=False)
    manifest = {
        "keywords": [
            {
                "keyword": "urna",
                "file_name": "urna.csv",
                "row_count": 2,
            }
        ]
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_build_processed_dataframe_deduplicates_and_normalizes(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_raw_fixture(raw_dir)

    df = build_processed_dataframe(raw_dir=raw_dir)

    assert len(df) == 1
    assert df.iloc[0]["model_text"] == "A URNA FOI FRAUDADA VERIFICACAO SOBRE URNA"
    assert df.iloc[0]["rating_label"] == "false"
    assert df.iloc[0]["verdict_text"] == "FALSE"
    assert df.iloc[0]["source_keyword"] == "urna"
    assert df.iloc[0]["review_url"].startswith("https://example.com/review-1")

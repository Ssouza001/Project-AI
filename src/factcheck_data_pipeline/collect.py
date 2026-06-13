from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DEFAULT_LANGUAGE, DEFAULT_RESULTS_PER_KEYWORD, KEYWORDS, RAW_DIR, RAW_MANIFEST_NAME, SOURCE_TOOL, ensure_directories
from .normalization import safe_filename


try:
    from factcheckexplorer.factcheckexplorer import FactCheckLib
except ImportError:  # pragma: no cover - dependency is optional until install time
    FactCheckLib = None


@dataclass(slots=True)
class CollectionResult:
    keyword: str
    file_path: Path
    row_count: int


class FactCheckCollector:
    def __init__(
        self,
        output_dir: Path | None = None,
        language: str = DEFAULT_LANGUAGE,
        num_results: int = DEFAULT_RESULTS_PER_KEYWORD,
    ) -> None:
        ensure_directories()
        self.output_dir = output_dir or RAW_DIR
        self.language = language
        self.num_results = num_results
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_keyword(self, keyword: str) -> CollectionResult:
        if FactCheckLib is None:
            raise RuntimeError(
                "factcheckexplorer is not installed. Run: pip install -r requirements.txt"
            )

        slug = safe_filename(keyword)
        csv_path = self.output_dir / f"query_{slug}.csv"
        attempt_path = self.output_dir / f"query_{slug}.attempt.csv"
        legacy_csv_path = self.output_dir / f"{slug}.csv"
        fallback_limits = [limit for limit in (80, 50, 20) if limit < self.num_results]
        requested_limits = [self.num_results, *fallback_limits]
        last_error: Exception | None = None

        for num_results in requested_limits:
            if attempt_path.exists():
                attempt_path.unlink()
            collector = FactCheckLib(
                query=keyword,
                language=self.language,
                num_results=num_results,
                csv_filename=str(attempt_path),
            )
            try:
                collector.process()
                if attempt_path.exists():
                    attempt_path.replace(csv_path)
                break
            except Exception as exc:
                last_error = exc
                if attempt_path.exists():
                    attempt_path.replace(csv_path)
                    break
        else:
            if csv_path.exists():
                pass
            elif legacy_csv_path.exists():
                csv_path = legacy_csv_path
            elif last_error is not None:
                raise last_error

        if not csv_path.exists():
            candidates = sorted(self.output_dir.glob(f"{slug}*.csv"), key=lambda path: path.stat().st_mtime)
            if candidates:
                csv_path = candidates[-1]

        if not csv_path.exists():
            raise FileNotFoundError(f"Collection completed but no CSV was written for keyword '{keyword}'.")

        try:
            row_count = len(pd.read_csv(csv_path))
        except Exception:
            row_count = 0

        return CollectionResult(keyword=keyword, file_path=csv_path, row_count=row_count)

    def collect_all(self, keywords: Iterable[str] | None = None) -> list[CollectionResult]:
        results: list[CollectionResult] = []
        errors: list[dict[str, str]] = []
        for keyword in keywords or KEYWORDS:
            try:
                results.append(self.collect_keyword(keyword))
            except Exception as exc:
                errors.append({"keyword": keyword, "error": f"{type(exc).__name__}: {exc}"})
        if not results:
            raise RuntimeError("Collection did not produce any CSV files.")
        self._write_manifest(results, errors)
        return results

    def _write_manifest(self, results: list[CollectionResult], errors: list[dict[str, str]] | None = None) -> None:
        manifest = {
            "source_tool": SOURCE_TOOL,
            "language": self.language,
            "num_results_per_keyword": self.num_results,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "keywords": [
                {
                    "keyword": result.keyword,
                    "file_name": result.file_path.name,
                    "row_count": result.row_count,
                }
                for result in results
            ],
            "errors": errors or [],
        }
        manifest_path = self.output_dir / RAW_MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

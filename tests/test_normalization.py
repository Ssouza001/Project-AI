from __future__ import annotations

import pytest

from factcheck_data_pipeline.normalization import (
    clean_verdict_text,
    normalize_rating,
    normalize_text,
    normalize_text_for_model,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Verdadeiro", "true"),
        ("Certo", "true"),
        ("Correto", "true"),
        ("Falso", "false"),
        ("Errado", "false"),
        ("Nao e verdade", "false"),
        ("Meio verdade", "mixed"),
        ("Out of context", "mixed"),
        ("Nao e bem assim", "mixed"),
        ("Distorcido", "mixed"),
        ("Enganosa", "mixed"),
        ("Sao enganosas postagens sobre pesquisa eleitoral", "mixed"),
        ("Pants on fire", "false"),
        ("Verified", "true"),
        ("Falso: teste simples", "false"),
        ("Verdadeiro: teste confirmado", "true"),
        (None, "mixed"),
    ],
)
def test_normalize_rating(value, expected):
    assert normalize_rating(value) == expected


def test_clean_verdict_text_outputs_uppercase_text():
    assert clean_verdict_text("  Falso: teste simples  ") == "FALSO: TESTE SIMPLES"


def test_normalize_text_strips_accents_and_punctuation():
    assert normalize_text("Eleição! Bolsonaro, Lula?") == "eleicao bolsonaro lula"


def test_normalize_text_for_model_outputs_uppercase_text():
    assert normalize_text_for_model("Eleição! Bolsonaro, Lula?") == "ELEICAO BOLSONARO LULA"

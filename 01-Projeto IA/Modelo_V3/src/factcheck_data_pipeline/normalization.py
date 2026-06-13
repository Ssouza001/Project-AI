from __future__ import annotations

import math
import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit


TRUE_PATTERNS = {
    "true",
    "verdadeiro",
    "verdade",
    "verified",
    "correct",
    "correto",
    "certo",
    "accurate",
    "genuine",
    "confiavel",
    "comprovado",
    "yes",
}

FALSE_PATTERNS = {
    "false",
    "falso",
    "incorrect",
    "wrong",
    "errado",
    "fake",
    "hoax",
    "pants on fire",
    "not true",
    "untrue",
    "nao e verdade",
}

MIXED_PATTERNS = {
    "mixed",
    "meio verdade",
    "meio falso",
    "partly true",
    "partly false",
    "mostly true",
    "mostly false",
    "out of context",
    "misleading",
    "enganoso",
    "enganosa",
    "enganosos",
    "enganosas",
    "enganador",
    "enganadora",
    "distorcido",
    "imprecise",
    "parcial",
    "fora de contexto",
    "needs context",
    "context",
    "nao e bem assim",
    "sem contexto",
    "falta contexto",
    "contextualizando",
    "confiavel mas",
    "confiavel, mas",
    "satira",
    "insustentavel",
    "exagerado",
}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(value: object) -> str:
    if _is_missing(value):
        return ""
    text = str(value)
    text = strip_accents(text)
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return normalize_whitespace(text)


def clean_verdict_text(value: object) -> str:
    if _is_missing(value):
        return ""
    return normalize_whitespace(str(value)).upper()


def normalize_text_for_model(value: object) -> str:
    return normalize_text(value).upper()


def normalize_column_name(name: object) -> str:
    return normalize_text(name).replace(" ", "_")


def safe_filename(value: str) -> str:
    text = normalize_text(value)
    text = text.replace(" ", "_")
    return re.sub(r"[^a-z0-9_]+", "", text).strip("_") or "dataset"


def normalize_url(value: object) -> str:
    if _is_missing(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if "://" not in text:
        text = f"https://{text}"
    parsed = urlsplit(text)
    netloc = parsed.netloc.lower().replace("www.", "")
    path = re.sub(r"/+$", "", parsed.path)
    return urlunsplit((parsed.scheme.lower(), netloc, path, "", ""))


def normalize_language(value: object, default: str = "pt") -> str:
    if _is_missing(value):
        return default
    text = str(value).strip().lower().replace("_", "-")
    if not text:
        return default
    if text.startswith("pt"):
        return "pt"
    if text.startswith("en"):
        return "en"
    if "-" in text:
        return text.split("-")[0]
    return text[:3]


def normalize_date(value: object) -> str:
    if _is_missing(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        import pandas as pd

        parsed = pd.to_datetime(text, errors="coerce", utc=False, dayfirst=True)
        if pd.isna(parsed):
            return ""
        return parsed.date().isoformat()
    except Exception:
        return ""


def normalize_rating(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return "mixed"
    if any(pattern in text for pattern in MIXED_PATTERNS):
        return "mixed"
    if any(pattern in text for pattern in FALSE_PATTERNS):
        return "false"
    if any(pattern in text for pattern in TRUE_PATTERNS):
        return "true"
    return "mixed"


def build_record_id(claim_text: str, review_url: str) -> str:
    import hashlib

    basis = f"{claim_text}|{review_url}".encode("utf-8")
    return hashlib.sha256(basis).hexdigest()[:16]

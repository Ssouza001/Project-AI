from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock, Thread
from time import sleep
from typing import Any
from urllib.parse import urlparse

import joblib
import requests
from flask import Flask, jsonify, render_template, request
from src.backend.services.metrics_service import (
    enrich_current_metrics_with_latest,
    read_json_file,
    summarize_model_metrics,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - keeps the app usable without python-dotenv.
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODEL_VERSION = "Modelo_V3"
MODEL_SOURCE = "ml_model_v3"
DEFAULT_MODEL_CONFIDENCE_THRESHOLD_TO_MIXED = 0.55
DEFAULT_FALSE_CONFIDENCE_THRESHOLD_TO_FALSE = 0.85
MODEL_CONTEXT_CONFIDENCE_THRESHOLD = 0.60
DATA_START_DATE = os.getenv("NEWS_DATA_START_DATE", "2024-01-01")
CURRENT_PRESIDENT_NAME = "Luiz Inacio Lula da Silva"
CURRENT_PRESIDENT_COMMON_NAME = "Lula"
CURRENT_PRESIDENT_TERM_START_DATE = "2023-01-01"
CURRENT_PRESIDENT_CONFIDENCE = 0.98
MODELO_DIR = BASE_DIR / MODEL_VERSION
MODEL_PATH = MODELO_DIR / "artifacts" / "model" / "baseline_model.joblib"
MODEL_METRICS_PATH = MODELO_DIR / "artifacts" / "reports" / "metrics.json"
LATEST_TRAINING_METRICS_PATH = ROOT_DIR / "data" / "models_v3" / "metrics.json"
MODEL_SRC = MODELO_DIR / "src"
DATA_DIR = BASE_DIR / "data"
TRUSTED_TRUE_NEWS_PATH = ROOT_DIR / "data" / "supplemental" / "trusted_true_news.csv"
RUNTIME_RESULTS_PATH = DATA_DIR / "runtime_results.csv"
BR_NEWS_PATH = DATA_DIR / "br_news.csv"
TRAINING_CANDIDATES_PATH = DATA_DIR / "training_candidates.csv"
AUTO_RETRAIN_STATE_PATH = DATA_DIR / "auto_retrain_state.json"
GOOGLE_FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
NEWSAPI_TOP_HEADLINES_URL = "https://newsapi.org/v2/top-headlines"
MEDIASTACK_NEWS_URL = "http://api.mediastack.com/v1/news"
DEFAULT_GOOGLE_FACTCHECK_LANGUAGE = "pt"

RUNTIME_DATASET_FIELDS = [
    "id",
    "created_at",
    "last_seen_at",
    "times_seen",
    "input_claim",
    "returned_claim",
    "source",
    "label",
    "confidence",
    "model_version",
    "publisher",
    "textual_rating",
    "review_title",
    "review_url",
    "warning",
    "fact_check_error",
]

runtime_dataset_lock = Lock()
br_news_lock = Lock()
training_candidates_lock = Lock()
retrain_lock = Lock()
auto_retrain_started = False
last_retrain_status: dict[str, Any] = {
    "status": "idle",
    "last_started_at": "",
    "last_finished_at": "",
    "last_error": "",
}

BR_NEWS_FIELDS = [
    "id",
    "collected_at",
    "last_seen_at",
    "times_seen",
    "source_api",
    "source_name",
    "title",
    "description",
    "url",
    "published_at",
    "language",
    "country",
    "keyword",
]

TRAINING_CANDIDATE_FIELDS = [
    "id",
    "created_at",
    "last_seen_at",
    "times_seen",
    "input_claim",
    "model_label",
    "model_confidence",
    "raw_model_label",
    "reason",
    "candidate_status",
    "suggested_label",
    "source_api",
    "source_name",
    "title",
    "description",
    "url",
    "published_at",
    "keyword",
    "notes",
]


def load_local_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")
else:
    load_local_env_file(BASE_DIR / ".env")

if str(MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_SRC))

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo nao encontrado em: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_model_metrics() -> dict[str, Any]:
    if not MODEL_METRICS_PATH.exists():
        return {}

    try:
        return json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def parse_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip())
    except ValueError:
        return default


def parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "sim", "on"}


def get_model_confidence_threshold_to_mixed() -> float:
    env_threshold = os.getenv("MODEL_CONFIDENCE_THRESHOLD_TO_MIXED")
    if env_threshold:
        return parse_float_env(
            "MODEL_CONFIDENCE_THRESHOLD_TO_MIXED",
            DEFAULT_MODEL_CONFIDENCE_THRESHOLD_TO_MIXED,
        )

    metrics = load_model_metrics()
    decision_policy = metrics.get("decision_policy")
    if isinstance(decision_policy, dict):
        value = decision_policy.get("confidence_threshold_to_mixed")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    return DEFAULT_MODEL_CONFIDENCE_THRESHOLD_TO_MIXED


def get_false_confidence_threshold_to_false() -> float:
    return parse_float_env(
        "MODEL_FALSE_CONFIDENCE_THRESHOLD_TO_FALSE",
        DEFAULT_FALSE_CONFIDENCE_THRESHOLD_TO_FALSE,
    )


def clear_model_runtime_caches() -> None:
    load_model.cache_clear()
    load_model_metrics.cache_clear()


def get_google_factcheck_api_key() -> str | None:
    """Keeps API keys out of the frontend and reads them only on the backend."""
    return (
        os.getenv("GOOGLE_FACTCHECK_API_KEY")
        or os.getenv("FACTCHECK_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


def get_news_api_key() -> str | None:
    return os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY")


def get_mediastack_api_key() -> str | None:
    return os.getenv("MEDIASTACK_API_KEY")


def get_admin_token() -> str | None:
    return os.getenv("ADMIN_TOKEN")


def request_admin_token() -> str:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", maxsplit=1)[1].strip()
    return request.headers.get("X-Admin-Token", "").strip()


def normalize_for_match(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_url(value: str | None) -> str:
    text = str(value or "").strip().lower()
    text = text.split("#", maxsplit=1)[0]
    text = text.rstrip("/")
    return text


def has_any_term(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


STOPWORDS_PT = {
    "a",
    "o",
    "as",
    "os",
    "um",
    "uma",
    "uns",
    "umas",
    "de",
    "da",
    "do",
    "das",
    "dos",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "para",
    "por",
    "com",
    "sem",
    "que",
    "e",
    "ou",
    "foi",
    "foram",
    "sera",
    "serao",
    "sobre",
    "apos",
    "mais",
    "menos",
    "como",
    "se",
    "ao",
    "aos",
    "pela",
    "pelo",
    "pelas",
    "pelos",
}

ELECTION_PRIORITY_TERMS = [
    "eleicao",
    "eleicoes",
    "eleitoral",
    "suplementar",
    "roraima",
    "tse",
    "tre",
    "urna",
    "urnas",
    "voto",
    "votacao",
    "apuracao",
    "campanha",
    "mesario",
    "titulo",
    "candidato",
    "partido",
]

GENERIC_CONTEXT_TERMS = {
    "eleicao",
    "eleicoes",
    "eleitoral",
    "tse",
    "tre",
    "voto",
    "votacao",
    "urna",
    "urnas",
    "campanha",
}

TRUSTED_TRAINING_DOMAINS = [
    "tse.jus.br",
    "tre-",
    "agenciabrasil.ebc.com.br",
    "gov.br",
    "planalto.gov.br",
]

BLOCKED_NEWS_DOMAINS = {
    domain.strip().lower()
    for domain in os.getenv("BLOCKED_NEWS_DOMAINS", "").split(",")
    if domain.strip()
}


def normalize_google_label(textual_rating: str | None) -> str:
    """Maps several API verdict texts to the project taxonomy."""
    text = normalize_for_match(textual_rating)
    if not text:
        return "mixed"

    mixed_terms = [
        "mixed",
        "mista",
        "misto",
        "meio",
        "parcial",
        "parcialmente",
        "distorcido",
        "distorcida",
        "enganoso",
        "enganosa",
        "fora de contexto",
        "sem contexto",
        "nao e bem assim",
        "nao eh bem assim",
        "impreciso",
        "imprecisa",
        "exagerado",
        "exagerada",
        "inconclusivo",
        "inconclusiva",
        "nao comprovado",
        "nao comprovada",
    ]
    false_terms = [
        "false",
        "falso",
        "falsa",
        "errado",
        "errada",
        "fake",
        "hoax",
        "mentira",
        "mentiroso",
        "mentirosa",
        "boato",
        "nao e verdade",
        "nao eh verdade",
        "sem provas",
        "inveridico",
        "inveridica",
        "nao procede",
        "nao confere",
    ]
    true_terms = [
        "true",
        "verdadeiro",
        "verdadeira",
        "correto",
        "correta",
        "certo",
        "certa",
        "comprovado",
        "comprovada",
        "verificado",
        "verificada",
        "e verdade",
        "confere",
    ]

    if has_any_term(text, mixed_terms):
        return "mixed"
    if has_any_term(text, false_terms):
        return "false"
    if has_any_term(text, true_terms):
        return "true"
    return "mixed"


def parse_iso_like_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        pass
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def configured_start_date() -> date:
    return parse_iso_like_date(DATA_START_DATE) or date(2024, 1, 1)


def is_on_or_after_start_date(value: Any) -> bool:
    parsed = parse_iso_like_date(value)
    return parsed is not None and parsed >= configured_start_date()


def article_domain(value: str | None) -> str:
    hostname = (urlparse(normalize_url(value)).hostname or "").lower()
    return hostname.removeprefix("www.")


def is_blocked_domain(value: str | None) -> bool:
    hostname = article_domain(value)
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in BLOCKED_NEWS_DOMAINS)


def is_article_in_configured_period(article: dict[str, Any]) -> bool:
    return is_on_or_after_start_date(article.get("published_at") or article.get("review_date"))


def meaningful_tokens(value: str | None) -> set[str]:
    return {
        token
        for token in normalize_for_match(value).split()
        if len(token) >= 4 and token not in STOPWORDS_PT
    }


def is_explicit_false_allegation(claim: str) -> bool:
    normalized = normalize_for_match(claim)
    election_context = [
        "urna",
        "urnas",
        "voto",
        "eleicao",
        "eleicoes",
        "eleitoral",
        "tse",
        "campanha",
        "presidente",
    ]
    false_markers = [
        "fraude",
        "fraudada",
        "fraudadas",
        "fraudado",
        "falso",
        "falsa",
        "fake",
        "mentira",
        "boato",
        "alterar votos",
        "manipular votos",
        "votos manipulados",
        "urnas hackeadas",
        "urna hackeada",
        "sem provas",
    ]
    return any(term in normalized for term in election_context) and any(
        marker in normalized for marker in false_markers
    )


@lru_cache(maxsize=1)
def load_trusted_true_news_rows() -> tuple[dict[str, Any], ...]:
    if not TRUSTED_TRUE_NEWS_PATH.exists():
        return tuple()

    rows: list[dict[str, Any]] = []
    try:
        with TRUSTED_TRUE_NEWS_PATH.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if str(row.get("rating_label") or "").strip().lower() != "true":
                    continue
                if not is_on_or_after_start_date(row.get("review_date") or row.get("claim_date")):
                    continue
                claim_text = str(row.get("claim_text") or "").strip()
                if not claim_text:
                    continue
                rows.append(
                    {
                        "claim_text": claim_text,
                        "review_title": str(row.get("review_title") or "").strip(),
                        "publisher": str(row.get("publisher") or "").strip(),
                        "review_url": str(row.get("review_url") or "").strip(),
                        "review_date": str(row.get("review_date") or "").strip(),
                        "tokens": meaningful_tokens(
                            f"{claim_text} {row.get('review_title') or ''} {row.get('source_keyword') or ''}"
                        ),
                    }
                )
    except OSError:
        return tuple()
    return tuple(rows)


def trusted_true_evidence_result(claim: str) -> dict[str, Any] | None:
    normalized_claim = normalize_for_match(claim)
    claim_tokens = meaningful_tokens(claim)
    if len(claim_tokens) < 3:
        return None

    best_row: dict[str, Any] | None = None
    best_score = 0.0
    for row in load_trusted_true_news_rows():
        row_text = normalize_for_match(f"{row['claim_text']} {row['review_title']}")
        if normalized_claim and (
            normalized_claim in row_text or row_text in normalized_claim
        ):
            best_row = row
            best_score = 1.0
            break

        row_tokens = set(row.get("tokens") or set())
        if not row_tokens:
            continue
        overlap = claim_tokens & row_tokens
        denominator = max(1, min(len(claim_tokens), len(row_tokens)))
        score = len(overlap) / denominator
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None or best_score < 0.55:
        return None

    return {
        "source": "trusted_true_2024plus_evidence",
        "label": "true",
        "confidence": float(min(0.97, 0.84 + (best_score * 0.13))),
        "warning": (
            "Encontramos evidência compatível na base confiável 2024+ usada no retreino."
        ),
        "claim": claim,
        "review_title": best_row.get("review_title") or best_row.get("claim_text"),
        "review_url": best_row.get("review_url"),
        "publisher": best_row.get("publisher"),
        "textual_rating": "VERDADEIRO",
        "model_version": MODEL_VERSION,
    }


def detect_current_president_fact(claim: str) -> dict[str, Any] | None:
    normalized = normalize_for_match(claim)
    if not normalized:
        return None

    asks_current_president = (
        "quem e o presidente" in normalized
        or "presidente atual" in normalized
        or "atual presidente" in normalized
        or "presidente do brasil" in normalized
    )
    mentions_lula = "lula" in normalized or "luiz inacio" in normalized
    mentions_bolsonaro = "bolsonaro" in normalized
    says_not = any(term in normalized for term in [" nao ", " nao e ", " deixou de ser "])

    if asks_current_president and (mentions_lula or not mentions_bolsonaro):
        return {
            "source": "current_political_fact_rules",
            "label": "true",
            "confidence": CURRENT_PRESIDENT_CONFIDENCE,
            "warning": (
                "Fato politico atual validado por regra de seguranca: "
                f"{CURRENT_PRESIDENT_COMMON_NAME} e o presidente em exercicio desde "
                f"{CURRENT_PRESIDENT_TERM_START_DATE}."
            ),
            "claim": claim,
            "review_title": "Presidente atual do Brasil",
            "review_url": "https://www.gov.br/planalto/pt-br/conheca-a-presidencia/biografia-do-presidente",
            "publisher": "Presidencia da Republica",
            "textual_rating": "VERDADEIRO",
            "model_version": MODEL_VERSION,
        }

    if asks_current_president and mentions_bolsonaro and not says_not:
        return {
            "source": "current_political_fact_rules",
            "label": "false",
            "confidence": CURRENT_PRESIDENT_CONFIDENCE,
            "warning": (
                "Fato politico atual validado por regra de seguranca: "
                f"o presidente em exercicio e {CURRENT_PRESIDENT_NAME}, nao Jair Bolsonaro."
            ),
            "claim": claim,
            "review_title": "Presidente atual do Brasil",
            "review_url": "https://www.gov.br/planalto/pt-br/conheca-a-presidencia/biografia-do-presidente",
            "publisher": "Presidencia da Republica",
            "textual_rating": "FALSO",
            "model_version": MODEL_VERSION,
        }

    return None


def detect_2022_presidential_result_fact(claim: str) -> dict[str, Any] | None:
    normalized = normalize_for_match(claim)
    if not normalized or "2022" not in normalized:
        return None

    election_terms = ["eleicao", "eleicoes", "segundo turno", "presidencial", "presidente eleito"]
    if not any(term in normalized for term in election_terms):
        return None

    mentions_lula = "lula" in normalized or "luiz inacio" in normalized
    mentions_bolsonaro = "bolsonaro" in normalized
    win_terms = ["venceu", "ganhou", "derrotou", "foi eleito", "eleito presidente"]
    has_win_term = any(term in normalized for term in win_terms)

    if mentions_lula and mentions_bolsonaro and has_win_term:
        lula_before_bolsonaro = normalized.find("lula") < normalized.find("bolsonaro")
        label = "true" if lula_before_bolsonaro else "false"
    elif mentions_lula and has_win_term:
        label = "true"
    elif mentions_bolsonaro and ("reeleito" in normalized or has_win_term):
        label = "false"
    else:
        return None

    return {
        "source": "current_political_fact_rules",
        "label": label,
        "confidence": CURRENT_PRESIDENT_CONFIDENCE,
        "warning": (
            "Resultado presidencial de 2022 validado por regra de seguranca: "
            "Lula venceu Jair Bolsonaro no segundo turno."
        ),
        "claim": claim,
        "review_title": "Resultado da eleicao presidencial de 2022",
        "review_url": "https://resultados.tse.jus.br/oficial/app/index.html#/eleicao/resultados?cargo=1",
        "publisher": "Tribunal Superior Eleitoral",
        "textual_rating": "VERDADEIRO" if label == "true" else "FALSO",
        "model_version": MODEL_VERSION,
    }


def detect_election_security_true_fact(claim: str) -> dict[str, Any] | None:
    normalized = normalize_for_match(claim)
    if not normalized:
        return None

    election_terms = [
        "urna",
        "urnas",
        "urna eletronica",
        "sistema eleitoral",
        "sistemas eleitorais",
        "voto",
        "votacao",
        "tse",
        "tribunal superior eleitoral",
        "justica eleitoral",
    ]
    positive_terms = [
        "seguranca",
        "seguro",
        "segura",
        "integridade",
        "sigilo",
        "transparencia",
        "auditoria",
        "teste publico",
        "teste de seguranca",
        "barreiras de seguranca",
        "reforcando as barreiras",
        "nao conseguiu comprometer",
        "nao conseguiram comprometer",
        "nenhuma das tentativas conseguiu comprometer",
        "sem comprometer",
        "confirmam a seguranca",
        "comprovou a seguranca",
        "melhoria continua",
        "melhorias nos sistemas eleitorais",
    ]
    direct_false_allegations = [
        "foram fraudadas",
        "foi fraudada",
        "fraudadas para alterar",
        "fraude nas urnas",
        "alterar votos",
        "manipular votos",
        "votos manipulados",
        "urna hackeada",
        "urnas hackeadas",
    ]

    has_election_context = any(term in normalized for term in election_terms)
    has_positive_security_context = any(term in normalized for term in positive_terms)
    has_direct_false_allegation = any(term in normalized for term in direct_false_allegations)

    if not has_election_context or not has_positive_security_context or has_direct_false_allegation:
        return None

    return {
        "source": "current_political_fact_rules",
        "label": "true",
        "confidence": 0.92,
        "warning": (
            "Afirmação sobre segurança ou integridade do processo eleitoral reconhecida "
            "como contexto verdadeiro por regra de segurança do domínio."
        ),
        "claim": claim,
        "review_title": "Segurança e integridade do processo eleitoral",
        "review_url": "https://www.tse.jus.br/comunicacao/noticias",
        "publisher": "Tribunal Superior Eleitoral",
        "textual_rating": "VERDADEIRO",
        "model_version": MODEL_VERSION,
    }


def search_google_fact_check(claim: str) -> dict[str, Any] | None:
    api_key = get_google_factcheck_api_key()
    if not api_key:
        return None

    params = {
        "key": api_key,
        "query": claim,
        "languageCode": os.getenv("GOOGLE_FACTCHECK_LANGUAGE", DEFAULT_GOOGLE_FACTCHECK_LANGUAGE),
        "pageSize": 1,
    }

    response = requests.get(GOOGLE_FACTCHECK_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    claims = payload.get("claims") or []
    if not claims:
        return None

    first_claim: dict[str, Any] | None = None
    review: dict[str, Any] | None = None
    for candidate_claim in claims:
        for candidate_review in candidate_claim.get("claimReview") or []:
            review_url = candidate_review.get("url")
            if is_blocked_domain(review_url):
                continue
            review_date = candidate_review.get("reviewDate") or candidate_review.get("publishDate")
            if review_date and not is_on_or_after_start_date(review_date):
                continue
            first_claim = candidate_claim
            review = candidate_review
            break
        if review is not None:
            break

    if first_claim is None or review is None:
        return None

    rating = review.get("textualRating") or ""
    publisher = review.get("publisher") or {}

    return {
        "source": "google_fact_check",
        "label": normalize_google_label(rating),
        "confidence": None,
        "warning": "Resultado retornado pela API Google Fact Check.",
        "claim": first_claim.get("text") or claim,
        "review_title": review.get("title"),
        "review_url": review.get("url"),
        "publisher": publisher.get("name"),
        "textual_rating": rating,
    }


def predict_with_local_model(claim: str) -> dict[str, Any]:
    current_fact_result = detect_current_president_fact(claim)
    if current_fact_result:
        current_fact_result["raw_model_label"] = current_fact_result["label"]
        current_fact_result["confidence_threshold_to_mixed"] = get_model_confidence_threshold_to_mixed()
        current_fact_result["low_confidence_adjusted"] = False
        return current_fact_result

    election_result = detect_2022_presidential_result_fact(claim)
    if election_result:
        election_result["raw_model_label"] = election_result["label"]
        election_result["confidence_threshold_to_mixed"] = get_model_confidence_threshold_to_mixed()
        election_result["low_confidence_adjusted"] = False
        return election_result

    election_security_result = detect_election_security_true_fact(claim)
    if election_security_result:
        election_security_result["raw_model_label"] = election_security_result["label"]
        election_security_result["confidence_threshold_to_mixed"] = get_model_confidence_threshold_to_mixed()
        election_security_result["false_confidence_threshold_to_false"] = get_false_confidence_threshold_to_false()
        election_security_result["low_confidence_adjusted"] = False
        return election_security_result

    trusted_true_result = trusted_true_evidence_result(claim)
    if trusted_true_result:
        trusted_true_result["raw_model_label"] = trusted_true_result["label"]
        trusted_true_result["confidence_threshold_to_mixed"] = get_model_confidence_threshold_to_mixed()
        trusted_true_result["false_confidence_threshold_to_false"] = get_false_confidence_threshold_to_false()
        trusted_true_result["low_confidence_adjusted"] = False
        return trusted_true_result

    model = load_model()
    raw_label = str(model.predict([claim])[0])
    label = raw_label
    confidence = None
    low_confidence_adjusted = False
    confidence_threshold_to_mixed = get_model_confidence_threshold_to_mixed()

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([claim])[0]
        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = model.named_steps["classifier"].classes_
        classes = list(classes)
        confidence = float(probabilities.max())

        if confidence < confidence_threshold_to_mixed:
            label = "mixed"
            low_confidence_adjusted = raw_label != label
        elif (
            raw_label == "false"
            and confidence < get_false_confidence_threshold_to_false()
            and not is_explicit_false_allegation(claim)
        ):
            label = "mixed"
            low_confidence_adjusted = True

    return {
        "source": MODEL_SOURCE,
        "label": label,
        "confidence": confidence,
        "raw_model_label": raw_label,
        "confidence_threshold_to_mixed": confidence_threshold_to_mixed,
        "false_confidence_threshold_to_false": get_false_confidence_threshold_to_false(),
        "low_confidence_adjusted": low_confidence_adjusted,
        "warning": (
            "Classificacao experimental do Modelo_V3. "
            "Use como apoio, nao como garantia absoluta de veracidade."
        ),
        "model_version": MODEL_VERSION,
    }


def build_news_id(input_claim: str, result: dict[str, Any]) -> str:
    """Creates a stable code so the same news item is not stored twice."""
    review_url = normalize_url(result.get("review_url"))
    if review_url:
        basis = f"url:{review_url}"
    else:
        returned_claim = result.get("claim") or result.get("returned_claim") or input_claim
        basis = f"claim:{normalize_for_match(returned_claim)}"

    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def parse_positive_int(value: Any, default: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def normalize_runtime_row(row: dict[str, Any]) -> dict[str, Any]:
    input_claim = str(row.get("input_claim") or "").strip()
    returned_claim = str(row.get("returned_claim") or input_claim).strip()
    created_at = str(row.get("created_at") or datetime.now(timezone.utc).isoformat())
    normalized = {field: row.get(field, "") for field in RUNTIME_DATASET_FIELDS}
    normalized["input_claim"] = input_claim
    normalized["returned_claim"] = returned_claim
    normalized["created_at"] = created_at
    normalized["last_seen_at"] = str(row.get("last_seen_at") or created_at)
    normalized["times_seen"] = str(parse_positive_int(row.get("times_seen"), 1))
    normalized["id"] = build_news_id(input_claim, normalized)
    return normalized


def merge_runtime_rows(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    existing["last_seen_at"] = incoming.get("last_seen_at") or existing.get("last_seen_at")
    existing["times_seen"] = str(
        parse_positive_int(existing.get("times_seen"), 1)
        + parse_positive_int(incoming.get("times_seen"), 1)
    )

    # Preserve richer API data if a repeated claim later gets a Google result.
    for field in [
        "source",
        "label",
        "confidence",
        "model_version",
        "publisher",
        "textual_rating",
        "review_title",
        "review_url",
        "warning",
        "fact_check_error",
    ]:
        if incoming.get(field):
            existing[field] = incoming[field]

    return existing


def read_runtime_dataset_rows() -> list[dict[str, Any]]:
    if not RUNTIME_RESULTS_PATH.exists():
        return []

    with RUNTIME_RESULTS_PATH.open("r", newline="", encoding="utf-8") as file:
        return [normalize_runtime_row(row) for row in csv.DictReader(file)]


def write_runtime_dataset_rows(rows: list[dict[str, Any]]) -> None:
    with RUNTIME_RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RUNTIME_DATASET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_br_news_id(article: dict[str, Any]) -> str:
    article_url = normalize_url(article.get("url"))
    if article_url:
        basis = f"url:{article_url}"
    else:
        title = normalize_for_match(article.get("title"))
        source_name = normalize_for_match(article.get("source_name"))
        basis = f"title:{source_name}:{title}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def normalize_br_article(article: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    title = str(article.get("title") or "").strip()
    description = str(article.get("description") or "").strip()
    normalized = {
        "id": "",
        "collected_at": str(article.get("collected_at") or now),
        "last_seen_at": str(article.get("last_seen_at") or now),
        "times_seen": str(parse_positive_int(article.get("times_seen"), 1)),
        "source_api": str(article.get("source_api") or ""),
        "source_name": str(article.get("source_name") or ""),
        "title": title,
        "description": description,
        "url": normalize_url(article.get("url")),
        "published_at": str(article.get("published_at") or ""),
        "language": str(article.get("language") or "pt"),
        "country": str(article.get("country") or "br"),
        "keyword": str(article.get("keyword") or ""),
    }
    normalized["id"] = build_br_news_id(normalized)
    return normalized


def read_br_news_rows() -> list[dict[str, Any]]:
    if not BR_NEWS_PATH.exists():
        return []

    with BR_NEWS_PATH.open("r", newline="", encoding="utf-8") as file:
        return [normalize_br_article(row) for row in csv.DictReader(file)]


def write_br_news_rows(rows: list[dict[str, Any]]) -> None:
    with BR_NEWS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BR_NEWS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def merge_br_news_rows(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    existing["last_seen_at"] = incoming.get("last_seen_at") or existing.get("last_seen_at")
    existing["times_seen"] = str(
        parse_positive_int(existing.get("times_seen"), 1)
        + parse_positive_int(incoming.get("times_seen"), 1)
    )

    for field in BR_NEWS_FIELDS:
        if field in {"id", "collected_at", "last_seen_at", "times_seen"}:
            continue
        if incoming.get(field):
            existing[field] = incoming[field]
    return existing


def store_br_news_articles(articles: list[dict[str, Any]]) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    normalized_articles = [
        normalize_br_article(article)
        for article in articles
        if article.get("title")
        and not is_blocked_domain(article.get("url"))
        and is_article_in_configured_period(article)
    ]

    with br_news_lock:
        rows_by_id: dict[str, dict[str, Any]] = {}
        for row in read_br_news_rows():
            rows_by_id[row["id"]] = row

        inserted = 0
        duplicates = 0
        stored_articles: list[dict[str, Any]] = []
        for article in normalized_articles:
            if article["id"] in rows_by_id:
                duplicates += 1
                rows_by_id[article["id"]] = merge_br_news_rows(rows_by_id[article["id"]], article)
            else:
                inserted += 1
                rows_by_id[article["id"]] = article
            stored_articles.append(rows_by_id[article["id"]])

        rows = list(rows_by_id.values())
        rows.sort(key=lambda item: str(item.get("collected_at") or ""))
        write_br_news_rows(rows)

    return {
        "articles": stored_articles,
        "inserted": inserted,
        "duplicates": duplicates,
        "dataset_path": "data/br_news.csv",
    }


def article_claim_text(article: dict[str, Any]) -> str:
    title = str(article.get("title") or "").strip()
    description = str(article.get("description") or "").strip()
    if description and description.lower() != title.lower():
        return f"{title}. {description}"
    return title


def tokens_for_match(value: str | None) -> list[str]:
    normalized = normalize_for_match(value)
    return [
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in STOPWORDS_PT
    ]


def build_context_keyword(claim: str) -> str:
    tokens = tokens_for_match(claim)
    if not tokens:
        return claim[:90]

    selected: list[str] = []
    for priority in ELECTION_PRIORITY_TERMS:
        if priority in tokens and priority not in selected:
            selected.append(priority)

    for token in tokens:
        if token not in selected:
            selected.append(token)
        if len(selected) >= 7:
            break

    return " ".join(selected[:7])


def article_relevance_score(claim: str, article: dict[str, Any]) -> int:
    claim_tokens = set(tokens_for_match(claim))
    article_tokens = set(tokens_for_match(article_claim_text(article)))
    if not claim_tokens or not article_tokens:
        return 0

    distinctive_tokens = claim_tokens - GENERIC_CONTEXT_TERMS
    if distinctive_tokens and not (distinctive_tokens & article_tokens):
        return 0

    score = len(claim_tokens & article_tokens) * 3
    article_text = normalize_for_match(article_claim_text(article))
    for term in ELECTION_PRIORITY_TERMS:
        if term in article_text:
            score += 1
    return score


def best_relevant_articles(claim: str, articles: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    scored = []
    for article in articles:
        score = article_relevance_score(claim, article)
        if score > 0:
            scored.append((score, article))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [article for _, article in scored[:limit]]


def local_context_articles(claim: str, limit: int = 3) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_br_news_rows()
        if not is_blocked_domain(row.get("url")) and is_article_in_configured_period(row)
    ]
    return best_relevant_articles(claim, rows, limit=limit)


def is_trusted_training_article(article: dict[str, Any]) -> bool:
    article_url = normalize_url(article.get("url"))
    source_name = normalize_for_match(article.get("source_name"))

    if "tribunal superior eleitoral" in source_name or source_name == "tse":
        return True
    if "agencia brasil" in source_name:
        return True

    parsed = urlparse(article_url)
    hostname = (parsed.hostname or "").lower()

    # Trust only official election justice domains and Agencia Brasil host.
    if hostname == "agenciabrasil.ebc.com.br":
        return True
    if hostname == "tse.jus.br" or hostname.endswith(".tse.jus.br"):
        return True
    if hostname.endswith(".jus.br") and hostname.startswith("tre-"):
        return True

    return False


def build_training_candidate_id(input_claim: str, article: dict[str, Any]) -> str:
    article_url = normalize_url(article.get("url"))
    if article_url:
        basis = f"{normalize_for_match(input_claim)}|{article_url}"
    else:
        basis = f"{normalize_for_match(input_claim)}|{normalize_for_match(article_claim_text(article))}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def normalize_training_candidate(row: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    normalized = {field: row.get(field, "") for field in TRAINING_CANDIDATE_FIELDS}
    normalized["created_at"] = str(row.get("created_at") or now)
    normalized["last_seen_at"] = str(row.get("last_seen_at") or normalized["created_at"])
    normalized["times_seen"] = str(parse_positive_int(row.get("times_seen"), 1))
    normalized["input_claim"] = str(row.get("input_claim") or "").strip()
    normalized["url"] = normalize_url(row.get("url"))
    normalized["id"] = row.get("id") or build_training_candidate_id(normalized["input_claim"], normalized)
    return normalized


def read_training_candidate_rows() -> list[dict[str, Any]]:
    if not TRAINING_CANDIDATES_PATH.exists():
        return []

    with TRAINING_CANDIDATES_PATH.open("r", newline="", encoding="utf-8") as file:
        return [normalize_training_candidate(row) for row in csv.DictReader(file)]


def write_training_candidate_rows(rows: list[dict[str, Any]]) -> None:
    with TRAINING_CANDIDATES_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAINING_CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def merge_training_candidate(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    existing["last_seen_at"] = incoming.get("last_seen_at") or existing.get("last_seen_at")
    existing["times_seen"] = str(
        parse_positive_int(existing.get("times_seen"), 1)
        + parse_positive_int(incoming.get("times_seen"), 1)
    )

    for field in TRAINING_CANDIDATE_FIELDS:
        if field in {"id", "created_at", "last_seen_at", "times_seen"}:
            continue
        if incoming.get(field):
            existing[field] = incoming[field]
    return existing


def store_training_candidates(
    input_claim: str,
    model_result: dict[str, Any],
    articles: list[dict[str, Any]],
    keyword: str,
) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    incoming_rows = []

    for article in articles:
        trusted = is_trusted_training_article(article)
        candidate_status = "trusted_true_ready" if trusted else "needs_review"
        suggested_label = "true" if trusted else ""
        normalized_article = normalize_br_article(article)
        incoming_rows.append(
            normalize_training_candidate(
                {
                    "created_at": now,
                    "last_seen_at": now,
                    "times_seen": "1",
                    "input_claim": input_claim,
                    "model_label": model_result.get("label"),
                    "model_confidence": model_result.get("confidence"),
                    "raw_model_label": model_result.get("raw_model_label"),
                    "reason": "low_model_confidence_context",
                    "candidate_status": candidate_status,
                    "suggested_label": suggested_label,
                    "source_api": normalized_article.get("source_api"),
                    "source_name": normalized_article.get("source_name"),
                    "title": normalized_article.get("title"),
                    "description": normalized_article.get("description"),
                    "url": normalized_article.get("url"),
                    "published_at": normalized_article.get("published_at"),
                    "keyword": keyword,
                    "notes": (
                        "Fonte confiavel: pode ser promovida para treino true."
                        if trusted
                        else "Revisar manualmente antes de usar no treino."
                    ),
                }
            )
        )

    with training_candidates_lock:
        rows_by_id: dict[str, dict[str, Any]] = {}
        for row in read_training_candidate_rows():
            rows_by_id[str(row.get("id") or "")] = row

        inserted = 0
        duplicates = 0
        trusted_ready = 0
        for row in incoming_rows:
            if row["candidate_status"] == "trusted_true_ready":
                trusted_ready += 1
            if row["id"] in rows_by_id:
                duplicates += 1
                rows_by_id[row["id"]] = merge_training_candidate(rows_by_id[row["id"]], row)
            else:
                inserted += 1
                rows_by_id[row["id"]] = row

        rows = list(rows_by_id.values())
        rows.sort(key=lambda item: str(item.get("created_at") or ""))
        write_training_candidate_rows(rows)

    return {
        "training_candidates_path": "data/training_candidates.csv",
        "training_candidates_saved": inserted,
        "training_candidates_duplicates": duplicates,
        "training_candidates_trusted_ready": trusted_ready,
    }


def should_fetch_context(model_result: dict[str, Any]) -> bool:
    if model_result.get("source") != MODEL_SOURCE:
        return False

    confidence = model_result.get("confidence")
    if confidence is None:
        return True

    try:
        return float(confidence) < MODEL_CONTEXT_CONFIDENCE_THRESHOLD
    except (TypeError, ValueError):
        return True


def build_low_confidence_context(input_claim: str, model_result: dict[str, Any]) -> dict[str, Any]:
    keyword = build_context_keyword(input_claim)
    errors: list[dict[str, str]] = []
    api_articles: list[dict[str, Any]] = []

    try:
        api_context = fetch_brazil_news(keyword=keyword, limit=8)
        api_articles = api_context.get("articles") or []
        errors.extend(api_context.get("errors") or [])
    except requests.RequestException as exc:
        errors.append({"source": "news_context", "error": str(exc)})

    selected_articles = best_relevant_articles(input_claim, api_articles, limit=3)
    context_source = "external_apis"

    if not selected_articles:
        selected_articles = local_context_articles(input_claim, limit=3)
        context_source = "local_news_dataset"

    storage_info = store_training_candidates(
        input_claim=input_claim,
        model_result=model_result,
        articles=selected_articles,
        keyword=keyword,
    ) if selected_articles else {
        "training_candidates_path": "data/training_candidates.csv",
        "training_candidates_saved": 0,
        "training_candidates_duplicates": 0,
        "training_candidates_trusted_ready": 0,
    }

    return {
        "context_used": bool(selected_articles),
        "context_reason": "model_confidence_below_threshold",
        "context_confidence_threshold": MODEL_CONTEXT_CONFIDENCE_THRESHOLD,
        "context_keyword": keyword,
        "context_source": context_source,
        "context_articles": [
            {
                "title": article.get("title"),
                "description": article.get("description"),
                "source_name": article.get("source_name"),
                "url": article.get("url"),
                "published_at": article.get("published_at"),
                "trusted_for_training": is_trusted_training_article(article),
            }
            for article in selected_articles
        ],
        "context_errors": errors,
        "training_note": (
            "Noticias de contexto foram salvas como candidatas. "
            "So devem virar treino automaticamente quando tiverem rotulo confiavel "
            "ou vierem de fonte marcada como trusted_true."
        ),
        **storage_info,
    }


def fetch_newsapi_br_news(keyword: str, limit: int) -> list[dict[str, Any]]:
    api_key = get_news_api_key()
    if not api_key:
        return []

    params = {
        "apiKey": api_key,
        "country": "br",
        "pageSize": min(max(limit, 1), 20),
    }
    if keyword:
        params["q"] = keyword

    response = requests.get(NEWSAPI_TOP_HEADLINES_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    articles = payload.get("articles") or []

    output = []
    for article in articles:
        source = article.get("source") or {}
        output.append(
            {
                "source_api": "newsapi",
                "source_name": source.get("name"),
                "title": article.get("title"),
                "description": article.get("description"),
                "url": article.get("url"),
                "published_at": article.get("publishedAt"),
                "language": "pt",
                "country": "br",
                "keyword": keyword,
            }
        )
    return output


def fetch_mediastack_br_news(keyword: str, limit: int) -> list[dict[str, Any]]:
    api_key = get_mediastack_api_key()
    if not api_key:
        return []

    params = {
        "access_key": api_key,
        "countries": "br",
        "languages": "pt",
        "limit": min(max(limit, 1), 20),
        "sort": "published_desc",
        "date": f"{DATA_START_DATE},{datetime.now(timezone.utc).date().isoformat()}",
    }
    if keyword:
        params["keywords"] = keyword

    response = requests.get(MEDIASTACK_NEWS_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    articles = payload.get("data") or []

    output = []
    for article in articles:
        output.append(
            {
                "source_api": "mediastack",
                "source_name": article.get("source"),
                "title": article.get("title"),
                "description": article.get("description"),
                "url": article.get("url"),
                "published_at": article.get("published_at"),
                "language": article.get("language") or "pt",
                "country": article.get("country") or "br",
                "keyword": keyword,
            }
        )
    return output


def fetch_brazil_news(keyword: str, limit: int) -> dict[str, Any]:
    articles: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    per_source_limit = max(1, min(limit, 4))
    for source_name, fetcher in [
        ("newsapi", fetch_newsapi_br_news),
        ("mediastack", fetch_mediastack_br_news),
    ]:
        try:
            source_articles = [
                article
                for article in fetcher(keyword, limit)
                if not is_blocked_domain(article.get("url")) and is_article_in_configured_period(article)
            ]
            articles.extend(source_articles[:per_source_limit])
        except requests.RequestException as exc:
            errors.append({"source": source_name, "error": str(exc)})

    by_id: dict[str, dict[str, Any]] = {}
    for article in articles:
        normalized = normalize_br_article(article)
        by_id[normalized["id"]] = normalized

    stored = store_br_news_articles(list(by_id.values()))
    return {
        "keyword": keyword,
        "articles": [
            {
                **article,
                "claim_text": article_claim_text(article),
            }
            for article in stored["articles"]
        ],
        "total": len(stored["articles"]),
        "inserted": stored["inserted"],
        "duplicates": stored["duplicates"],
        "dataset_path": stored["dataset_path"],
        "errors": errors,
    }


def build_runtime_record(input_claim: str, result: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    returned_claim = result.get("claim") or input_claim

    return {
        "id": build_news_id(input_claim, result),
        "created_at": now,
        "last_seen_at": now,
        "times_seen": "1",
        "input_claim": input_claim,
        "returned_claim": returned_claim,
        "source": result.get("source"),
        "label": result.get("label"),
        "confidence": result.get("confidence"),
        "model_version": result.get("model_version"),
        "publisher": result.get("publisher"),
        "textual_rating": result.get("textual_rating"),
        "review_title": result.get("review_title"),
        "review_url": result.get("review_url"),
        "warning": result.get("warning"),
        "fact_check_error": result.get("fact_check_error"),
    }


def append_runtime_dataset(input_claim: str, result: dict[str, Any]) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = build_runtime_record(input_claim, result)

    with runtime_dataset_lock:
        rows_by_id: dict[str, dict[str, Any]] = {}
        for row in read_runtime_dataset_rows():
            row_id = str(row.get("id") or "")
            if row_id in rows_by_id:
                rows_by_id[row_id] = merge_runtime_rows(rows_by_id[row_id], row)
            else:
                rows_by_id[row_id] = row

        is_duplicate = record["id"] in rows_by_id
        if is_duplicate:
            rows_by_id[record["id"]] = merge_runtime_rows(rows_by_id[record["id"]], record)
        else:
            rows_by_id[record["id"]] = record

        rows = list(rows_by_id.values())
        rows.sort(key=lambda item: str(item.get("created_at") or ""))
        write_runtime_dataset_rows(rows)

        return {
            "news_id": record["id"],
            "duplicate": is_duplicate,
            "times_seen": parse_positive_int(rows_by_id[record["id"]].get("times_seen"), 1),
        }


def attach_storage_status(input_claim: str, result: dict[str, Any]) -> dict[str, Any]:
    response = dict(result)
    try:
        storage_info = append_runtime_dataset(input_claim, response)
        response["stored_locally"] = True
        response["dataset_path"] = "data/runtime_results.csv"
        response.update(storage_info)
    except OSError as exc:
        response["stored_locally"] = False
        response["storage_error"] = str(exc)
    return response


def attach_context_when_needed(input_claim: str, result: dict[str, Any]) -> dict[str, Any]:
    response = dict(result)
    if not should_fetch_context(response):
        return response

    try:
        response.update(build_low_confidence_context(input_claim, response))
    except OSError as exc:
        response["context_used"] = False
        response["context_error"] = str(exc)

    return response


def trusted_candidate_rows() -> list[dict[str, Any]]:
    allowed_status = {"trusted_true_ready", "approved_true"}
    return [
        row
        for row in read_training_candidate_rows()
        if str(row.get("candidate_status") or "") in allowed_status
    ]


def training_candidate_fingerprint(rows: list[dict[str, Any]]) -> str:
    basis = "|".join(
        sorted(
            f"{row.get('id', '')}:{row.get('candidate_status', '')}:{row.get('times_seen', '')}"
            for row in rows
        )
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def read_auto_retrain_state() -> dict[str, Any]:
    if not AUTO_RETRAIN_STATE_PATH.exists():
        return {}

    try:
        return json.loads(AUTO_RETRAIN_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_auto_retrain_state(state: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUTO_RETRAIN_STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_safe_retrain(trigger: str) -> tuple[dict[str, Any], int]:
    if not retrain_lock.acquire(blocking=False):
        return {"error": "Retreino ja esta em andamento.", "status": "busy"}, 409

    try:
        last_retrain_status.update(
            {
                "status": "running",
                "trigger": trigger,
                "last_started_at": datetime.now(timezone.utc).isoformat(),
                "last_finished_at": "",
                "last_error": "",
            }
        )

        script_path = ROOT_DIR / "scripts" / "train_model_v3.py"
        if not script_path.exists():
            message = f"Script de retreino nao encontrado: {script_path}"
            last_retrain_status.update({"status": "error", "last_error": message})
            return {"error": message}, 500

        completed = subprocess.run(
            [sys.executable, str(script_path), "--promote-if-better"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )

        if completed.returncode != 0:
            payload = {
                "error": "Retreino falhou.",
                "returncode": completed.returncode,
                "stderr": completed.stderr[-4000:],
            }
            last_retrain_status.update(
                {
                    "status": "error",
                    "last_finished_at": datetime.now(timezone.utc).isoformat(),
                    "last_error": payload["stderr"],
                }
            )
            return payload, 500

        metrics_path = ROOT_DIR / "data" / "models_v3" / "metrics.json"
        report_path = ROOT_DIR / "data" / "models_v3" / "retrain_report.json"
        payload: dict[str, Any] = {
            "status": "ok",
            "trigger": trigger,
            "metrics_path": str(metrics_path),
            "report_path": str(report_path),
        }

        promoted = False
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            promotion = metrics.get("promotion") if isinstance(metrics.get("promotion"), dict) else {}
            promoted = bool(promotion.get("promoted")) if isinstance(promotion, dict) else False
            payload.update(
                {
                    "selected_model": metrics.get("selected_model"),
                    "decision_policy": metrics.get("decision_policy"),
                    "test_metrics": metrics.get("test_metrics"),
                    "test_true_false_positive_count": metrics.get("test_true_false_positive_count"),
                    "promotion": metrics.get("promotion"),
                    "dataset_metadata": metrics.get("dataset_metadata"),
                }
            )

        if promoted:
            clear_model_runtime_caches()

        last_retrain_status.update(
            {
                "status": "ok",
                "last_finished_at": datetime.now(timezone.utc).isoformat(),
                "last_error": "",
                "promoted": promoted,
            }
        )
        return payload, 200
    finally:
        retrain_lock.release()


def should_auto_retrain() -> tuple[bool, dict[str, Any]]:
    rows = trusted_candidate_rows()
    min_candidates = parse_positive_int(os.getenv("AUTO_RETRAIN_MIN_CANDIDATES"), 1)
    fingerprint = training_candidate_fingerprint(rows)
    state = read_auto_retrain_state()

    details = {
        "trusted_candidate_count": len(rows),
        "min_candidates": min_candidates,
        "fingerprint": fingerprint,
        "last_fingerprint": state.get("last_candidate_fingerprint"),
    }

    if len(rows) < min_candidates:
        return False, {**details, "reason": "not_enough_trusted_candidates"}

    if fingerprint and fingerprint == state.get("last_candidate_fingerprint"):
        return False, {**details, "reason": "no_new_trusted_candidates"}

    return True, {**details, "reason": "ready"}


def auto_retrain_loop() -> None:
    interval_hours = parse_float_env("AUTO_RETRAIN_INTERVAL_HOURS", 24.0)
    interval_seconds = max(interval_hours, 0.1) * 3600

    while True:
        should_run, details = should_auto_retrain()
        last_retrain_status.update(
            {
                "auto_retrain_enabled": True,
                "auto_retrain_last_check_at": datetime.now(timezone.utc).isoformat(),
                "auto_retrain_last_check": details,
            }
        )

        if should_run:
            payload, status_code = run_safe_retrain("auto")
            if status_code == 200:
                state = read_auto_retrain_state()
                state.update(
                    {
                        "last_candidate_fingerprint": details["fingerprint"],
                        "last_retrain_at": datetime.now(timezone.utc).isoformat(),
                        "last_promotion": payload.get("promotion"),
                    }
                )
                write_auto_retrain_state(state)

        sleep(interval_seconds)


def start_auto_retrain_if_enabled() -> None:
    global auto_retrain_started

    if auto_retrain_started or not parse_bool_env("AUTO_RETRAIN_ENABLED"):
        return

    auto_retrain_started = True
    thread = Thread(target=auto_retrain_loop, name="auto-retrain", daemon=True)
    thread.start()


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_version": MODEL_VERSION,
            "model_path": str(MODEL_PATH),
            "runtime_dataset_path": str(RUNTIME_RESULTS_PATH),
            "training_candidates_path": str(TRAINING_CANDIDATES_PATH),
            "model_confidence_threshold_to_mixed": get_model_confidence_threshold_to_mixed(),
            "context_confidence_threshold": MODEL_CONTEXT_CONFIDENCE_THRESHOLD,
            "news_data_start_date": DATA_START_DATE,
            "current_president": CURRENT_PRESIDENT_NAME,
            "google_factcheck_enabled": bool(get_google_factcheck_api_key()),
            "newsapi_enabled": bool(get_news_api_key()),
            "mediastack_enabled": bool(get_mediastack_api_key()),
            "admin_retrain_enabled": bool(get_admin_token()),
            "auto_retrain_enabled": parse_bool_env("AUTO_RETRAIN_ENABLED"),
            "last_retrain_status": last_retrain_status,
        }
    )


@app.get("/model/metrics")
def model_metrics():
    current_metrics = summarize_model_metrics(load_model_metrics())
    latest_training_metrics = summarize_model_metrics(read_json_file(LATEST_TRAINING_METRICS_PATH))
    current_metrics = enrich_current_metrics_with_latest(current_metrics, latest_training_metrics)

    return jsonify(
        {
            "runtime": {
                "model_version": MODEL_VERSION,
                "model_path": str(MODEL_PATH),
                "confidence_threshold_to_mixed": get_model_confidence_threshold_to_mixed(),
                "context_confidence_threshold": MODEL_CONTEXT_CONFIDENCE_THRESHOLD,
            },
            "current_model": current_metrics,
            "latest_training_run": latest_training_metrics,
        }
    )


@app.post("/admin/retrain")
def admin_retrain():
    admin_token = get_admin_token()
    if not admin_token:
        return jsonify({"error": "ADMIN_TOKEN nao configurado no backend."}), 503

    if request_admin_token() != admin_token:
        return jsonify({"error": "Token administrativo invalido."}), 401

    payload, status_code = run_safe_retrain("admin")
    return jsonify(payload), status_code


@app.get("/news/br")
def news_br():
    keyword = str(request.args.get("q", "eleição")).strip()
    try:
        limit = int(request.args.get("limit", "8"))
    except ValueError:
        limit = 8

    limit = min(max(limit, 1), 20)
    payload = fetch_brazil_news(keyword=keyword, limit=limit)
    payload["data_start_date"] = DATA_START_DATE
    return jsonify(payload)


@app.post("/analyze")
def analyze():
    data = request.get_json(silent=True) or {}
    claim = str(data.get("claim", "")).strip()

    if not claim:
        return jsonify({"error": "Campo obrigatorio ausente: claim"}), 400

    current_fact_result = detect_current_president_fact(claim)
    if current_fact_result:
        return jsonify(attach_storage_status(claim, current_fact_result))

    election_result = detect_2022_presidential_result_fact(claim)
    if election_result:
        return jsonify(attach_storage_status(claim, election_result))

    try:
        fact_check_result = search_google_fact_check(claim)
        if fact_check_result:
            return jsonify(attach_storage_status(claim, fact_check_result))
    except requests.RequestException as exc:
        model_result = predict_with_local_model(claim)
        model_result["fact_check_error"] = str(exc)
        model_result = attach_context_when_needed(claim, model_result)
        return jsonify(attach_storage_status(claim, model_result))

    model_result = predict_with_local_model(claim)
    model_result = attach_context_when_needed(claim, model_result)
    return jsonify(attach_storage_status(claim, model_result))


if __name__ == "__main__":
    start_auto_retrain_if_enabled()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

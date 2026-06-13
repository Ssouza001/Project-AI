from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

SOURCE_TOOL = "factcheckexplorer"
DEFAULT_LANGUAGE = "pt"
DEFAULT_RESULTS_PER_KEYWORD = 250
RANDOM_STATE = 42
DATA_START_DATE = "2024-01-01"
CURRENT_PRESIDENT_NAME = "Luiz Inacio Lula da Silva"
CURRENT_PRESIDENT_COMMON_NAME = "Lula"
CURRENT_PRESIDENT_TERM_START_DATE = "2023-01-01"
CURRENT_PRESIDENT_ELECTION_DATE = "2022-10-30"

BASE_KEYWORDS = [
    "elei\u00e7\u00e3o",
    "elei\u00e7\u00f5es",
    "bolsonaro",
    "lula",
    "luiz inacio lula da silva",
    "presidente atual",
    "presidente do brasil",
    "pt",
    "campanha",
    "urna",
    "urnas",
    "urna eletronica",
    "urnas eletronicas",
    "voto",
    "apuracao",
    "fraude",
]

# Extra queries focused on Brazilian election context and better class balance.
ENRICHMENT_KEYWORDS = [
    "lula presidente",
    "lula presidente eleito",
    "lula eleito presidente 2022",
    "resultado eleicao presidente 2022",
    "resultado segundo turno presidente 2022",
    "diplomacao lula alckmin",
    "posse lula 2023",
    "governo lula",
    "presidente lula planalto",
    "tse resultado lula bolsonaro",
    "tse resultado eleicoes 2022",
    "justica eleitoral resultado presidencial",
    "boletim de urna",
    "auditoria urnas",
    "teste publico de seguranca urna",
    "codigo fonte urna eletronica",
    "desinformacao eleitoral",
    "noticia falsa eleicoes",
    "fake news eleicoes",
    "voto impresso",
    "voto impresso simone tebet",
    "voto impresso deputados distrito federal",
    "pec voto impresso",
    "segundo turno",
    "eleicoes 2024",
    "eleicoes municipais 2024 voto",
    "tudo sobre eleicoes municipais 2024",
    "campanha bolsonaro",
    "camisa candidato eleicoes",
    "camisa candidato dia eleicao",
    "camiseta candidato eleicao",
    "usar camisa candidato eleicao",
    "exercito niteroi eleicoes",
    "lula estudantes universidade",
    "universidade estudantes lula",
    "verdadeiro",
    "correto lula",
    "e verdade voto impresso",
    "comprovado bolsonaro",
]

KEYWORDS = BASE_KEYWORDS + ENRICHMENT_KEYWORDS

RAW_MANIFEST_NAME = "manifest.json"
PROCESSED_DATASET_NAME = "factcheck_dataset_processed.csv"
BASELINE_MODEL_NAME = "baseline_model.joblib"
BASELINE_METRICS_NAME = "metrics.json"
MODEL_PREDICTIONS_NAME = "model_predictions.csv"


def ensure_directories() -> None:
    for directory in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

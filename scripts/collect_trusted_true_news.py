from __future__ import annotations

import csv
import html
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from factcheck_data_pipeline.normalization import (  # noqa: E402
    build_record_id,
    normalize_date,
    normalize_text,
    normalize_text_for_model,
    normalize_url,
)


OUTPUT_DIR = ROOT / "data" / "supplemental"
OUTPUT_CSV = OUTPUT_DIR / "trusted_true_news.csv"
OUTPUT_MANIFEST = OUTPUT_DIR / "trusted_true_manifest.json"

REQUEST_TIMEOUT_SECONDS = 18
MAX_SENTENCES_PER_SOURCE = 14
MIN_SENTENCE_LENGTH = 70
MAX_SENTENCE_LENGTH = 420


@dataclass(frozen=True, slots=True)
class TrustedPage:
    url: str
    publisher: str
    topic: str
    source_keyword: str


TRUSTED_PAGES = [
    TrustedPage(
        "https://www.tse.jus.br/comunicacao/noticias/2026/Maio/eleicoes-2026-testes-confirmam-mais-uma-vez-a-seguranca-do-sistema-eleitoral-brasileiro",
        "Tribunal Superior Eleitoral",
        "urna_eletronica_seguranca",
        "urna eletrônica segurança eleições",
    ),
    TrustedPage(
        "https://www.tse.jus.br/comunicacao/noticias/2026/Maio/eleicoes-2026-teste-de-confirmacao-comeca-nesta-quarta-13-e-valida-melhorias-nos-sistemas-eleitorais",
        "Tribunal Superior Eleitoral",
        "urna_eletronica_seguranca",
        "teste público de segurança urna",
    ),
    TrustedPage(
        "https://www.tse.jus.br/comunicacao/noticias/2026/Marco/tse-aprova-calendario-eleitoral-e-regulamenta-uso-de-ia-nas-eleicoes-2026",
        "Tribunal Superior Eleitoral",
        "calendario_regras_campanha",
        "calendário eleitoral IA campanha",
    ),
    TrustedPage(
        "https://www.tse.jus.br/comunicacao/noticias/2026/Marco/eleicoes-2026-conheca-a-ordem-de-votacao-na-urna-eletronica",
        "Tribunal Superior Eleitoral",
        "votacao",
        "ordem de votação urna eletrônica",
    ),
    TrustedPage(
        "https://www.tse.jus.br/comunicacao/noticias/2026/Janeiro/confira-quais-cargos-estarao-em-disputa-nas-eleicoes-2026",
        "Tribunal Superior Eleitoral",
        "votacao_cargos",
        "cargos eleições 2026",
    ),
    TrustedPage(
        "https://www.tse.jus.br/comunicacao/noticias/2026/Maio/30-anos-da-urna-eletronica-eleicoes-2024-foram-o-maior-pleito-informatizado-do-mundo",
        "Tribunal Superior Eleitoral",
        "urna_eletronica",
        "urna eletrônica processo eleitoral",
    ),
    TrustedPage(
        "https://www.tse.jus.br/comunicacao/noticias/2026/Abril/por-dentro-das-eleicoes-conheca-as-regras-sobre-uso-de-ia-na-campanha-eleitoral-de-2026",
        "Tribunal Superior Eleitoral",
        "campanha_regras_ia",
        "uso de inteligência artificial campanha eleitoral",
    ),
    TrustedPage(
        "https://www.tse.jus.br/comunicacao/noticias/2026/Marco/tse-lanca-pagina-com-informacoes-sobre-eleicoes-2026",
        "Tribunal Superior Eleitoral",
        "servico_eleitoral",
        "informações eleições 2026 TSE",
    ),
    TrustedPage(
        "https://www.tse.jus.br/comunicacao/noticias/2026/Abril/eleicoes-2026-codigos-fonte-dos-sistemas-eleitorais-continuam-abertos-para-inspecao",
        "Tribunal Superior Eleitoral",
        "transparencia_auditoria",
        "código-fonte urna eletrônica fiscalização",
    ),
    TrustedPage(
        "https://www.tse.jus.br/comunicacao/noticias/2026/Marco/eleicoes-2026-confira-as-principais-datas-do-calendario-eleitoral",
        "Tribunal Superior Eleitoral",
        "calendario_eleitoral",
        "calendário eleitoral eleições 2026",
    ),
    TrustedPage(
        "https://www.tse.jus.br/comunicacao/noticias/2026/Marco/requisitos-tecnicos-sobre-conectividade-nas-eleicoes",
        "Tribunal Superior Eleitoral",
        "apuracao_transmissao",
        "transmissão dados urna apuração",
    ),
    TrustedPage(
        "https://www.tse.jus.br/comunicacao/noticias/2026/Marco/tse-e-tres-discutem-acoes-de-seguranca-para-as-eleicoes-de-2026",
        "Tribunal Superior Eleitoral",
        "seguranca_eleitoral",
        "segurança eleições 2026 TSE TRE",
    ),
    TrustedPage(
        "https://www.tse.jus.br/comunicacao/noticias/2026/Janeiro/urna-eletronica-entenda-como-o-equipamento-transformou-o-processo-eleitoral-brasileiro",
        "Tribunal Superior Eleitoral",
        "urna_eletronica",
        "urna eletrônica processo eleitoral brasileiro",
    ),
    TrustedPage(
        "https://www.tse.jus.br/legislacao/compilada/res/2026/resolucao-no-23-760-de-2-de-marco-de-2026",
        "Tribunal Superior Eleitoral",
        "calendario_eleitoral",
        "resolução calendário eleitoral 2026",
    ),
    TrustedPage(
        "https://agenciabrasil.ebc.com.br/justica/noticia/2026-05/tse-faz-novos-testes-de-seguranca-na-urna-eletronica",
        "Agência Brasil",
        "urna_eletronica_seguranca",
        "urna eletrônica segurança TSE",
    ),
    TrustedPage(
        "https://agenciabrasil.ebc.com.br/justica/noticia/2026-05/eleicoes-2026-financiamento-coletivo-de-campanha-comeca-nesta-sexta",
        "Agência Brasil",
        "campanha_financiamento",
        "financiamento coletivo campanha eleitoral",
    ),
    TrustedPage(
        "https://agenciabrasil.ebc.com.br/justica/noticia/2026-02/tse-aprova-regras-para-eleicoes-de-outubro",
        "Agência Brasil",
        "regras_eleitorais",
        "regras eleições 2026 TSE",
    ),
    TrustedPage(
        "https://agenciabrasil.ebc.com.br/justica/noticia/2026-05/prazo-para-tirar-ou-regularizar-titulo-de-eleitor-termina-nesta-quarta",
        "Agência Brasil",
        "titulo_eleitor",
        "título de eleitor regularização",
    ),
    TrustedPage(
        "https://agenciabrasil.ebc.com.br/justica/noticia/2026-05/eleicoes-2026-eleitor-pode-se-inscrever-para-ser-mesario-voluntario",
        "Agência Brasil",
        "mesario",
        "mesário voluntário eleições 2026",
    ),
    TrustedPage(
        "https://agenciabrasil.ebc.com.br/justica/noticia/2026-05/desafio-do-tse-e-conter-uso-ilegal-de-ia-na-eleicao-diz-nunes-marques",
        "Agência Brasil",
        "campanha_regras_ia",
        "inteligência artificial eleições TSE",
    ),
]


ELECTION_TERMS = {
    "eleicao",
    "eleicoes",
    "eleitoral",
    "eleitor",
    "eleitora",
    "eleitores",
    "eleitoras",
    "votacao",
    "voto",
    "urna",
    "urnas",
    "tse",
    "tre",
    "justica eleitoral",
    "campanha",
    "candidato",
    "candidata",
    "candidatos",
    "partido",
    "partidos",
    "apuração",
    "apuracao",
    "biometria",
    "mesario",
    "titulo de eleitor",
}

NOISE_TERMS = {
    "compartilhar",
    "whatsapp",
    "facebook",
    "twitter",
    "youtube",
    "image",
    "imagem",
    "foto:",
    "crédito:",
    "credito:",
    "edição:",
    "edicao:",
    "última atualização",
    "ultima atualizacao",
    "siga o canal",
    "continuar lendo",
    "relacionadas",
}


def fetch_html(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        content_type = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(content_type, errors="replace")


def find_meta(html_text: str, *names: str) -> str:
    for name in names:
        escaped = re.escape(name)
        patterns = [
            rf'<meta[^>]+(?:property|name)=["\']{escaped}["\'][^>]+content=["\']([^"\']+)["\']',
            rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']{escaped}["\']',
        ]
        for pattern in patterns:
            match = re.search(pattern, html_text, flags=re.IGNORECASE)
            if match:
                return normalize_space(html.unescape(match.group(1)))
    return ""


def normalize_space(value: str) -> str:
    value = html.unescape(value)
    value = value.replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def html_to_text(html_text: str) -> str:
    cleaned = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html_text)
    cleaned = re.sub(r"(?is)<br\s*/?>", ". ", cleaned)
    cleaned = re.sub(r"(?is)</(p|div|li|h1|h2|h3|section|article)>", ". ", cleaned)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    return normalize_space(cleaned)


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9])", text)
    return [normalize_space(piece) for piece in pieces if normalize_space(piece)]


def sentence_has_election_context(sentence: str) -> bool:
    normalized = normalize_text(sentence)
    return any(term in normalized for term in ELECTION_TERMS)


def sentence_has_noise(sentence: str) -> bool:
    normalized = normalize_text(sentence)
    return any(term in normalized for term in NOISE_TERMS)


def is_useful_sentence(sentence: str) -> bool:
    if not MIN_SENTENCE_LENGTH <= len(sentence) <= MAX_SENTENCE_LENGTH:
        return False
    if sentence_has_noise(sentence):
        return False
    if not sentence_has_election_context(sentence):
        return False
    if sentence.count(" ") < 8:
        return False
    return True


def extract_title(html_text: str, fallback_url: str) -> str:
    title = find_meta(html_text, "og:title", "twitter:title")
    if title:
        return title

    match = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", html_text)
    if match:
        return normalize_space(re.sub(r"<[^>]+>", " ", match.group(1)))

    return fallback_url.rstrip("/").split("/")[-1].replace("-", " ").title()


def extract_description(html_text: str) -> str:
    return find_meta(html_text, "og:description", "description", "twitter:description")


def extract_review_date(html_text: str, plain_text: str) -> str:
    published = find_meta(html_text, "article:published_time", "date")
    if published:
        return normalize_date(published)

    match = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", plain_text)
    if match:
        return normalize_date(match.group(1))

    return ""


def build_record(page: TrustedPage, claim_text: str, title: str, review_date: str, row_index: int) -> dict[str, str]:
    review_url = normalize_url(page.url)
    claim_text_normalized = normalize_text(claim_text)
    review_title_normalized = normalize_text(title)

    return {
        "record_id": build_record_id(claim_text_normalized, review_url),
        "claim_text": claim_text,
        "claim_text_normalized": claim_text_normalized,
        "review_title": title,
        "review_title_normalized": review_title_normalized,
        "model_text": normalize_text_for_model(f"{claim_text} {title}"),
        "rating": "VERDADEIRO",
        "verdict_text": "VERDADEIRO",
        "rating_label": "true",
        "publisher": page.publisher,
        "review_url": review_url,
        "review_url_normalized": review_url,
        "language": "pt",
        "claim_date": review_date,
        "review_date": review_date,
        "source_keyword": page.source_keyword,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "source_tool": "trusted_true_sources",
        "raw_source_file": page.url,
        "raw_row_index": str(row_index),
        "topic": page.topic,
    }


def extract_records(page: TrustedPage) -> tuple[list[dict[str, str]], dict[str, str]]:
    html_text = fetch_html(page.url)
    title = extract_title(html_text, page.url)
    description = extract_description(html_text)
    plain_text = html_to_text(html_text)
    review_date = extract_review_date(html_text, plain_text)

    candidates = [title]
    if description:
        candidates.append(description)
    candidates.extend(split_sentences(plain_text))

    records: list[dict[str, str]] = []
    seen_claims: set[str] = set()
    for candidate in candidates:
        sentence = normalize_space(candidate)
        if not is_useful_sentence(sentence):
            continue
        claim_key = normalize_text(sentence)
        if claim_key in seen_claims:
            continue
        seen_claims.add(claim_key)
        records.append(build_record(page, sentence, title, review_date, len(records)))
        if len(records) >= MAX_SENTENCES_PER_SOURCE:
            break

    status = {
        "url": page.url,
        "publisher": page.publisher,
        "topic": page.topic,
        "records": str(len(records)),
        "status": "ok",
    }
    return records, status


def deduplicate(records: list[dict[str, str]]) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for record in records:
        by_id[record["record_id"]] = record
    return list(by_id.values())


def write_csv(records: list[dict[str, str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not records:
        raise RuntimeError("Nenhum registro verdadeiro foi coletado das fontes confiáveis.")

    fieldnames = list(records[0].keys())
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    all_records: list[dict[str, str]] = []
    statuses: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    for page in TRUSTED_PAGES:
        try:
            records, status = extract_records(page)
            all_records.extend(records)
            statuses.append(status)
            print(f"{status['records']} registros: {page.publisher} - {page.topic}")
        except Exception as exc:  # noqa: BLE001 - keep collection resilient.
            errors.append(
                {
                    "url": page.url,
                    "publisher": page.publisher,
                    "topic": page.topic,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"ERRO: {page.url} -> {type(exc).__name__}: {exc}")

    records = deduplicate(all_records)
    write_csv(records)

    manifest = {
        "source_tool": "trusted_true_sources",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_csv": str(OUTPUT_CSV.relative_to(ROOT)),
        "total_records": len(records),
        "source_pages": statuses,
        "errors": errors,
        "labeling_rule": (
            "Sentenças extraídas de fontes institucionais ou jornalísticas confiáveis "
            "foram rotuladas como true para balancear o dataset. Essa camada é "
            "suplementar e auditável, separada da coleta principal do factcheckexplorer."
        ),
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"saved": str(OUTPUT_CSV), "records": len(records), "errors": len(errors)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import requests


URL = "http://127.0.0.1:5000/analyze"
VALID_LABELS = {"true", "false", "mixed"}


def test_health_route() -> None:
    from app import app

    client = app.test_client()
    response = client.get("/health")
    data = response.get_json()

    assert response.status_code == 200
    assert data["status"] == "ok"
    assert data["model_version"] == "Modelo_V3"
    assert "model_confidence_threshold_to_mixed" in data
    assert "last_retrain_status" in data


def test_admin_retrain_requires_configured_token(monkeypatch) -> None:
    from app import app

    monkeypatch.delenv("ADMIN_TOKEN", raising=False)

    client = app.test_client()
    response = client.post("/admin/retrain")
    data = response.get_json()

    assert response.status_code == 503
    assert data["error"] == "ADMIN_TOKEN nao configurado no backend."


def test_local_model_prediction_uses_valid_label() -> None:
    from app import predict_with_local_model

    result = predict_with_local_model("A urna eletronica passa por teste publico de seguranca.")

    assert result["label"] in VALID_LABELS
    assert result["model_version"] == "Modelo_V3"


def test_current_president_fact_rule_marks_lula_as_true() -> None:
    from app import detect_current_president_fact

    result = detect_current_president_fact("Lula e o presidente atual do Brasil")

    assert result is not None
    assert result["label"] == "true"
    assert result["source"] == "current_political_fact_rules"


def test_current_president_fact_rule_marks_bolsonaro_as_false() -> None:
    from app import detect_current_president_fact

    result = detect_current_president_fact("Bolsonaro e o presidente atual do Brasil")

    assert result is not None
    assert result["label"] == "false"


def test_2022_presidential_result_rule_marks_lula_win_as_true() -> None:
    from app import detect_2022_presidential_result_fact

    result = detect_2022_presidential_result_fact(
        "Lula venceu Bolsonaro no segundo turno da eleicao presidencial de 2022"
    )

    assert result is not None
    assert result["label"] == "true"


def test_2022_presidential_result_rule_marks_bolsonaro_win_as_false() -> None:
    from app import detect_2022_presidential_result_fact

    result = detect_2022_presidential_result_fact(
        "Bolsonaro venceu Lula no segundo turno da eleicao presidencial de 2022"
    )

    assert result is not None
    assert result["label"] == "false"


def test_election_security_rule_marks_integrity_claim_as_true() -> None:
    from app import detect_election_security_true_fact

    result = detect_election_security_true_fact(
        "Nenhuma das tentativas conseguiu comprometer o sigilo nem a integridade do voto."
    )

    assert result is not None
    assert result["label"] == "true"


def test_local_model_does_not_mark_moderate_confidence_false_as_false() -> None:
    from app import predict_with_local_model

    result = predict_with_local_model(
        "Ele observou ainda que o trabalho de melhoria é contínuo e que, a cada ano, "
        "o TSE vai reforçando as barreiras de segurança."
    )

    assert result["label"] != "false"


def test_trusted_true_evidence_is_used_before_ml() -> None:
    from app import trusted_true_evidence_result

    result = trusted_true_evidence_result(
        "Eleicoes 2026: conheca a ordem de votacao na urna eletronica"
    )

    assert result is not None
    assert result["label"] == "true"
    assert result["source"] == "trusted_true_2024plus_evidence"


def test_explicit_false_allegation_can_still_return_false() -> None:
    from app import predict_with_local_model

    result = predict_with_local_model("As urnas eletronicas foram fraudadas para alterar votos.")

    assert result["label"] == "false"


def executar_analise(texto: str) -> None:
    print(f"\n--- Testando: '{texto}' ---")
    payload = {"claim": texto}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(URL, json=payload, headers=headers, timeout=15)

        if response.status_code == 200:
            data = response.json()
            print(f"Sucesso! Status: {response.status_code}")
            print(f"Fonte da Resposta: {data.get('source')}")
            print(f"Classificação: {data.get('label')}")
            print(f"Confiança: {data.get('confidence')}")
            print(f"Código da notícia: {data.get('news_id')}")
            print(f"Notícia duplicada: {data.get('duplicate')}")
            print(f"Quantidade de consultas: {data.get('times_seen')}")
            print(f"Salvo no dataset local: {data.get('stored_locally')}")
            print(f"Caminho do dataset: {data.get('dataset_path')}")
            print(f"Aviso: {data.get('warning')}")
            if data.get("review_url"):
                print(f"URL da checagem: {data.get('review_url')}")
        else:
            print(f"Erro no servidor: {response.status_code}")
            print(response.text)

    except Exception as exc:
        print(f"Falha ao conectar no backend: {exc}")


if __name__ == "__main__":
    executar_analise("O aprendizado de maquina e fascinante")
    executar_analise("A Terra e plana")

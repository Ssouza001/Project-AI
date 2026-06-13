# VerificaAI

Sistema experimental de checagem de fatos eleitorais em português. A aplicação tenta consultar a Google Fact Check API quando existe chave configurada; se não houver retorno, usa o `Modelo_V3` local como apoio.

## O que o projeto faz

- Recebe uma afirmação no chat web.
- Aplica regras de segurança para fatos eleitorais conhecidos.
- Consulta a Google Fact Check API, quando configurada.
- Usa um modelo local `TF-IDF + Logistic Regression` como fallback.
- Salva análises locais e evita duplicidade de registros.
- Permite retreino controlado do modelo com gates de qualidade.

## Estrutura limpa

```text
.
+-- 01-Projeto IA/              # Aplicacao Flask + frontend
|   +-- app.py                  # Backend principal
|   +-- Modelo_V3/              # Modelo usado em runtime
|   +-- static/                 # CSS e JavaScript
|   +-- templates/              # HTML
|   +-- data/                   # Arquivos criados durante o uso
+-- data/
|   +-- raw/                    # Coletas brutas do factcheckexplorer
|   +-- processed/              # Dataset processado
|   +-- supplemental/           # Dados confiaveis suplementares
+-- scripts/                    # Coleta suplementar e retreino V3
+-- src/factcheck_data_pipeline/ # Pipeline de coleta, limpeza e treino
+-- tests/                      # Testes do pipeline
+-- run.py                      # Entrada unica do projeto
+-- requirements.txt
```

## Requisitos

- Python 3.11 ou superior.
- Windows PowerShell, CMD ou terminal equivalente.
- Internet apenas para instalar dependências, coletar dados novos ou usar APIs externas.

## Instalação

Na raiz do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se quiser instalar como pacote editável:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

## Rodar o sistema

Inicie a aplicação web:

```powershell
.\.venv\Scripts\python.exe run.py ui
```

Abra:

```text
http://127.0.0.1:5000/
```

Rotas principais:

- `GET /health`: status do backend, modelo e chaves configuradas.
- `POST /analyze`: recebe `{ "claim": "texto para checar" }`.
- `GET /news/br?q=eleicao&limit=8`: busca notícias brasileiras quando APIs de notícias estão configuradas.
- `GET /model/metrics`: resume métricas do modelo atual.
- `POST /admin/retrain`: retreino manual, exige `ADMIN_TOKEN`.

## Configuração opcional

Copie o exemplo de ambiente do app:

```powershell
Copy-Item "01-Projeto IA\.env.example" "01-Projeto IA\.env"
```

Preencha somente o que for usar:

```env
GOOGLE_FACTCHECK_API_KEY=
NEWS_API_KEY=
MEDIASTACK_API_KEY=
ADMIN_TOKEN=
GOOGLE_FACTCHECK_LANGUAGE=pt
MODEL_CONFIDENCE_THRESHOLD_TO_MIXED=
AUTO_RETRAIN_ENABLED=false
AUTO_RETRAIN_INTERVAL_HOURS=24
AUTO_RETRAIN_MIN_CANDIDATES=1
```

Sem essas chaves, o sistema continua funcionando com regras locais e `Modelo_V3`.

## Pipeline de dados

Executar tudo:

```powershell
.\.venv\Scripts\python.exe run.py all
```

Executar por etapa:

```powershell
.\.venv\Scripts\python.exe run.py collect
.\.venv\Scripts\python.exe run.py process
.\.venv\Scripts\python.exe run.py train
```

Saídas recriadas pelo pipeline:

- `data/raw/manifest.json`
- `data/processed/factcheck_dataset_processed.csv`
- `data/models/baseline_model.joblib`
- `data/models/metrics.json`
- `data/models/model_predictions.csv`

## Modelo_V3

O app carrega o modelo de:

```text
01-Projeto IA/Modelo_V3/artifacts/model/baseline_model.joblib
```

Também mantenha:

```text
01-Projeto IA/Modelo_V3/src/
01-Projeto IA/Modelo_V3/artifacts/reports/metrics.json
```

O diretório `Modelo_V3` dentro do app é o pacote necessário para o runtime. O modelo é experimental e deve ser usado como apoio, não como prova absoluta.

## Retreino V3

Retreinar manualmente:

```powershell
.\.venv\Scripts\python.exe scripts\train_model_v3.py
```

Retreinar e promover para o app somente se passar nos gates:

```powershell
.\.venv\Scripts\python.exe scripts\train_model_v3.py --promote-if-better
```

O retreino usa o dataset processado, dados suplementares confiáveis e candidatos aprovados em `01-Projeto IA/data/training_candidates.csv`. A promoção é bloqueada quando há regressão relevante de métrica ou quando o novo modelo passa a classificar falsos/mistos como verdadeiros no teste.

## Testes

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Os testes cobrem normalização, processamento, divisão de dados e regras principais do backend.

## Observação importante

Este é um projeto acadêmico/experimental. Priorize fontes verificáveis e a Google Fact Check API quando houver chave configurada. O modelo local serve como fallback e sempre deve ser interpretado com cautela.

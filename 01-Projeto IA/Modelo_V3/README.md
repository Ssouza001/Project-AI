# Modelo_V3 - Modelo retreinado

Modelo retreinado do zero a partir do dataset processado.

## Resumo

- Algoritmo: `TF-IDF + Logistic Regression`
- Naive Bayes: não usado como modelo final
- Política de decisão: se a confiança máxima for menor que `0.55`, a resposta vira `mixed`
- Objetivo: melhorar a confiabilidade prática e reduzir respostas categóricas quando o modelo está inseguro
- Enriquecimento: exemplos `true` suplementares extraídos de TSE e Agência Brasil

## Arquivos principais

- `artifacts/model/baseline_model.joblib`: modelo treinado
- `artifacts/dataset/factcheck_dataset_processed.csv`: dataset usado
- `artifacts/reports/metrics.json`: métricas finais
- `artifacts/reports/model_predictions.csv`: predições auditáveis
- `artifacts/reports/collection_manifest.json`: coleta principal feita com `factcheckexplorer`
- `artifacts/reports/trusted_true_manifest.json`: fontes confiáveis usadas para reforçar a classe `true`
- `src/factcheck_data_pipeline/normalization.py`: normalização necessária para carregar o modelo

## Observação

O modelo continua experimental. A API Google Fact Check deve ser consultada primeiro; o modelo deve ser usado apenas como fallback.

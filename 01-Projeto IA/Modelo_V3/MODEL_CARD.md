# Model Card - Modelo_V3

## Uso previsto

Fallback experimental para classificação de afirmações quando a API Google Fact Check não retorna resultado.

## Modelo

`TF-IDF + Logistic Regression`.

## Dados usados

A coleta principal continua vindo do `factcheckexplorer`. Para reduzir o desbalanceamento, a versão atual também usa exemplos suplementares `true` extraídos de fontes confiáveis, como TSE e Agência Brasil. Esses registros ficam documentados em `trusted_true_manifest.json`.

## Política de baixa confiança

Quando a maior probabilidade prevista pelo modelo é menor que `0.55`, o sistema retorna `mixed`, indicando que a afirmação precisa de contexto e não deve ser cravada como verdadeira ou falsa.

## Limitações

O dataset ainda tem predominância de registros `false`. O modelo não substitui checagem jornalística, API oficial ou avaliação humana.

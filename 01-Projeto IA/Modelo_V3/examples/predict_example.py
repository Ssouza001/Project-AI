from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

CONFIDENCE_THRESHOLD_TO_MIXED = 0.55
model = joblib.load(BASE_DIR / "artifacts" / "model" / "baseline_model.joblib")
claim = " ".join(sys.argv[1:]).strip() or "Urnas eletronicas foram fraudadas nas eleicoes."
label = str(model.predict([claim])[0])
probabilities = model.predict_proba([claim])[0]
classes = list(model.named_steps["classifier"].classes_)
raw_confidence = float(probabilities[classes.index(label)])
max_confidence = float(probabilities.max())

if max_confidence < CONFIDENCE_THRESHOLD_TO_MIXED:
    label = "mixed"

confidence = max_confidence

print(json.dumps({
    "input": claim,
    "label": label,
    "confidence": confidence,
    "raw_model_confidence": raw_confidence,
    "confidence_threshold_to_mixed": CONFIDENCE_THRESHOLD_TO_MIXED,
    "source": "ml_model_v3",
    "model_version": "Modelo_V3",
    "disclaimer": "Classificacao experimental. Use como apoio, nao como garantia absoluta de veracidade."
}, indent=2, ensure_ascii=False))

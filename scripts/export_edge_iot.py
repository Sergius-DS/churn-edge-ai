#!/usr/bin/env python
"""
scripts/export_edge_iot.py
──────────────────────────
Exporta el modelo Edge IoT y su contrato mínimo.

Genera:
    edge/xgb_model_edge_iot.json
    edge/inference_contract_edge_iot.json
    edge/sample_scaled_edge_iot.npy
    edge/expected_proba_edge_iot.npy
"""

import json
import hashlib

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    EDGE_DIR,
    EDGE_IOT_ARTIFACTS_DIR,
    EDGE_IOT_MODEL_JSON,
    EDGE_IOT_CONTRACT_JSON,
    IOT_FEATURES,
    FORBIDDEN_FIELDS,
)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def main() -> None:
    print("=" * 70)
    print("📦 Exportando modelo Edge IoT")
    print("=" * 70)

    EDGE_DIR.mkdir(parents=True, exist_ok=True)

    model_path = EDGE_IOT_ARTIFACTS_DIR / "xgb_model.joblib"
    config_path = EDGE_IOT_ARTIFACTS_DIR / "model_config.joblib"
    scaler_path = EDGE_IOT_ARTIFACTS_DIR / "scaler.joblib"
    sample_path = EDGE_IOT_ARTIFACTS_DIR / "X_test_scaled_sample.csv"

    assert model_path.exists(), f"No existe {model_path}"
    assert config_path.exists(), f"No existe {config_path}"
    assert scaler_path.exists(), f"No existe {scaler_path}"
    assert sample_path.exists(), f"No existe {sample_path}"

    # ------------------------------------------------------------------
    # Cargar artefactos
    # ------------------------------------------------------------------
    model = joblib.load(model_path)
    config = joblib.load(config_path)
    scaler = joblib.load(scaler_path)

    feature_order = config.get("feature_order", [])
    threshold = config.get("best_threshold", 0.5)

    # ------------------------------------------------------------------
    # Exportar modelo XGBoost JSON
    # ------------------------------------------------------------------
    model.save_model(EDGE_IOT_MODEL_JSON)
    print(f"[export_edge_iot] ✅ Modelo guardado en {EDGE_IOT_MODEL_JSON}")

    model_sha256 = sha256_file(EDGE_IOT_MODEL_JSON)

    # ------------------------------------------------------------------
    # Muestra de paridad
    # ------------------------------------------------------------------
    sample = pd.read_csv(sample_path)

    if feature_order:
        sample = sample[feature_order]
    else:
        feature_order = list(sample.columns)

    X_sample = sample.values.astype(np.float32)

    dmat_sample = xgb.DMatrix(
        X_sample,
        feature_names=feature_order,
    )

    expected_proba = model.predict(dmat_sample)

    sample_scaled_path = EDGE_DIR / "sample_scaled_edge_iot.npy"
    expected_proba_path = EDGE_DIR / "expected_proba_edge_iot.npy"

    np.save(sample_scaled_path, X_sample)
    np.save(expected_proba_path, expected_proba)

    print(f"[export_edge_iot] ✅ Muestra escalada guardada en {sample_scaled_path}")
    print(f"[export_edge_iot] ✅ Predicción esperada guardada en {expected_proba_path}")

    # ------------------------------------------------------------------
    # Scaler para contrato
    # ------------------------------------------------------------------
    scaler_mean = scaler.mean_.tolist()
    scaler_scale = scaler.scale_.tolist()

    # ------------------------------------------------------------------
    # Contrato mínimo
    # ------------------------------------------------------------------
    contract = {
        "contract_version": "1.0.0-edge-iot",
        "model_id": "churn-edge-iot-xgb-v1",
        "model_artifact": EDGE_IOT_MODEL_JSON.name,
        "model_sha256": model_sha256,
        "privacy_mode": "edge_local_inference",
        "consent_required": True,
        "threshold": threshold,
        "feature_order": feature_order,
        "iot_features": IOT_FEATURES,
        "forbidden_fields": FORBIDDEN_FIELDS,
        "scaler": {
            "mean": scaler_mean,
            "scale": scaler_scale,
        },
        "output": {
            "type": "churn_probability",
            "range": [0.0, 1.0],
        },
        "note": (
            "Contrato mínimo para Edge IoT. "
            "feature_order corresponde a features post-preprocessing. "
            "Luego deberá mapearse a payload raw en Rust/mock/demo."
        ),
    }

    EDGE_IOT_CONTRACT_JSON.write_text(
        json.dumps(contract, indent=2),
        encoding="utf-8",
    )

    print(f"[export_edge_iot] ✅ Contrato guardado en {EDGE_IOT_CONTRACT_JSON}")


if __name__ == "__main__":
    main()
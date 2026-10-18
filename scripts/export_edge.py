#!/usr/bin/env python
"""
scripts/export_edge.py
──────────────────────
Entry point: python scripts/export_edge.py

Exporta el modelo entrenado a formato edge (JSON + contrato WASM).
Requiere que primero se haya ejecutado scripts/train_model.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from src.config import ARTIFACTS_DIR, BEST_THRESHOLD
from src.export_edge import export_edge_artifacts


def main():
    print("=" * 60)
    print("📦 CHURN EDGE AI — Export Edge (WASM/Rust)")
    print("=" * 60)

    # Verificar que existen los artefactos
    model_path = ARTIFACTS_DIR / "xgb_model.joblib"
    assert model_path.exists(), "Primero ejecuta: python scripts/train_model.py"

    # ─── Cargar artefactos ───
    print("\n📂 Cargando artefactos del modelo...")
    model = joblib.load(model_path)
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    encoder = joblib.load(ARTIFACTS_DIR / "onehot_encoder.joblib")
    num_imputer = joblib.load(ARTIFACTS_DIR / "num_imputer.joblib")
    cat_imputer = joblib.load(ARTIFACTS_DIR / "cat_imputer.joblib")
    ordinal_imputer = joblib.load(ARTIFACTS_DIR / "ordinal_imputer.joblib")
    config = joblib.load(ARTIFACTS_DIR / "model_config.joblib")

    threshold = config.get("best_threshold", BEST_THRESHOLD)

    # Verificar que el modelo tenga feature names
    if model.feature_names is None:
        raise ValueError("El modelo no tiene feature_names. Re-entrena con el script.")

    # ─── Cargar muestra real escalada (guardada por train_model.py) ───
    sample_path = ARTIFACTS_DIR / "X_train_scaled_sample.csv"

    assert sample_path.exists(), (
        f"No se encontró {sample_path}.\n"
        "Primero ejecuta: python scripts/train_model.py"
    )

    X_sample = pd.read_csv(sample_path)

    print(f"[export_edge] ✅ Muestra real cargada: {sample_path}")
    print(f"[export_edge]    Shape: {X_sample.shape}")

    # Verificar que las columnas coinciden con el modelo
    assert list(X_sample.columns) == list(model.feature_names), (
        "Las columnas de X_train_scaled_sample.csv no coinciden "
        "con los feature names del modelo."
    )

    # ─── Exportar ───
    print("\n🚀 Exportando artefactos edge...")
    export_edge_artifacts(
        model=model,
        X_train_scaled=X_sample,
        scaler=scaler,
        encoder=encoder,
        num_imputer=num_imputer,
        cat_imputer=cat_imputer,
        ordinal_imputer=ordinal_imputer,
        threshold=threshold,
    )

    print("\n✅ Export edge completado")


if __name__ == "__main__":
    main()
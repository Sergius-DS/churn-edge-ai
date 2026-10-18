#!/usr/bin/env python
"""
scripts/validate_parity.py
──────────────────────────
Entry point: python scripts/validate_parity.py

Valida que Python y el JSON exportado producen los mismos resultados.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import EDGE_DIR, PARITY_TOLERANCE


def main():
    print("=" * 60)
    print("🔍 CHURN EDGE AI — Validación de Paridad")
    print("=" * 60)

    # ─── Cargar parity sample ───
    parity_path = EDGE_DIR / "parity_sample.json"
    assert parity_path.exists(), (
        f"No existe {parity_path}. Ejecuta export_edge primero."
    )

    parity = json.loads(parity_path.read_text())

    features = np.array(parity["features"][:10])
    expected = np.array(parity["expected_proba"][:10])

    # ─── Cargar modelo desde JSON ───
    model = xgb.Booster()
    model.load_model(str(EDGE_DIR / "xgb_model.json"))

    # Obtener feature order
    feature_order = parity.get("feature_order")

    if feature_order is None:
        feature_order = model.feature_names

    assert feature_order is not None, (
        "No se encontró feature_order en parity_sample.json "
        "ni feature_names en el modelo JSON."
    )

    print(f"[parity] Features cargadas: {features.shape}")
    print(f"[parity] Feature order: {len(feature_order)} columnas")

    # ─── FIX: Crear DMatrix CON feature names ───
    df = pd.DataFrame(features, columns=feature_order)
    dmat = xgb.DMatrix(df)

    python_proba = model.predict(dmat)

    # ─── Comparar ───
    try:
        np.testing.assert_allclose(
            python_proba,
            expected,
            atol=PARITY_TOLERANCE
        )

        print(f"\n✅ PARIDAD VERIFICADA (atol={PARITY_TOLERANCE})")
        print(f"   Python[:5]:   {python_proba[:5]}")
        print(f"   Expected[:5]: {expected[:5]}")
        print(f"   Max diff:     {np.max(np.abs(python_proba - expected)):.2e}")

    except AssertionError as e:
        print(f"\n❌ PARIDAD FALLÓ: {e}")
        sys.exit(1)

    print("\n🎉 Python ↔ JSON son matemáticamente idénticos")


if __name__ == "__main__":
    main()
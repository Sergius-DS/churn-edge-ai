#!/usr/bin/env python
"""
scripts/generate_synthetic_iot.py
─────────────────────────────────
Genera el dataset extendido con features IoT sintéticas.

Input:
    data/raw/Credit_card_churn.csv

Output:
    data/processed/credit_card_churn_iot_synthetic.csv
    data/synthetic/iot_features_metadata.json
"""

import json

import pandas as pd

from src.config import (
    DATA_PATH,
    SYNTHETIC_IOT_CSV,
    SYNTHETIC_IOT_METADATA,
    TARGET,
    SEED,
    SYNTHETIC_IOT_PARAMS,
    IOT_FEATURES,
)

from src.synthetic_iot import add_synthetic_iot_features


def main() -> None:
    print("=" * 70)
    print("🧪 Generando features IoT sintéticas")
    print("=" * 70)

    SYNTHETIC_IOT_CSV.parent.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_IOT_METADATA.parent.mkdir(parents=True, exist_ok=True)

    print(f"[synthetic_iot] Leyendo dataset raw: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    print(f"[synthetic_iot] Shape raw: {df.shape}")

    # IMPORTANTE:
    # Se genera sobre el CSV raw, donde Attrition_Flag todavía viene como texto.
    # No se debe generar después de encode_target().
    df_iot = add_synthetic_iot_features(
        df,
        target_col=TARGET,
        seed=SEED,
        **SYNTHETIC_IOT_PARAMS,
    )

    print(f"[synthetic_iot] Shape con IoT: {df_iot.shape}")

    missing_iot = [col for col in IOT_FEATURES if col not in df_iot.columns]
    assert not missing_iot, f"Faltan features IoT generadas: {missing_iot}"

    df_iot.to_csv(SYNTHETIC_IOT_CSV, index=False)
    print(f"[synthetic_iot] ✅ Dataset sintético guardado en {SYNTHETIC_IOT_CSV}")

    # ------------------------------------------------------------------
    # Metadata de trazabilidad
    # ------------------------------------------------------------------
    metadata = {
        "seed": SEED,
        "target_col": TARGET,
        "synthetic_params": SYNTHETIC_IOT_PARAMS,
        "iot_features": IOT_FEATURES,
        "input_csv": str(DATA_PATH),
        "output_csv": str(SYNTHETIC_IOT_CSV),
        "rows": int(df_iot.shape[0]),
        "cols": int(df_iot.shape[1]),
        "note": (
            "Features IoT sintéticas para demo Edge AI Banking IoT. "
            "No representan telemetría real. "
            "Generadas con referencia fija IOT_REFERENCE y target_strength=0.0."
        ),
    }

    SYNTHETIC_IOT_METADATA.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"[synthetic_iot] ✅ Metadata guardada en {SYNTHETIC_IOT_METADATA}")

    # ------------------------------------------------------------------
    # Check rápido de distribución
    # ------------------------------------------------------------------
    if TARGET in df_iot.columns:
        print("\n[synthetic_iot] Media IoT por clase:")
        print(
            df_iot.groupby(TARGET)[IOT_FEATURES]
            .mean()
            .T
        )


if __name__ == "__main__":
    main()
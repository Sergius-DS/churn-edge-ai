#!/usr/bin/env python
"""
scripts/generate_synthetic_iot.py
─────────────────────────────────
Genera el dataset extendido con features IoT sintéticas.

Input:
    data/curated/churn_curated.csv  (SSOT, validada por contrato)

Output:
    data/processed/credit_card_churn_iot_synthetic.csv
    data/synthetic/iot_features_metadata.json
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_PATH,
    SYNTHETIC_IOT_CSV,
    SYNTHETIC_IOT_METADATA,
    TARGET,
    SEED,
    SYNTHETIC_IOT_PARAMS,
    IOT_FEATURES,
    IOT_REFERENCE,
)
from src.data_loading import load_raw_data
from src.synthetic_iot import add_synthetic_iot_features
from src.data_contract import validate
import src.data_contract as dc


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    print("=" * 70)
    print("🧪 Generando features IoT sintéticas")
    print("=" * 70)

    SYNTHETIC_IOT_CSV.parent.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_IOT_METADATA.parent.mkdir(parents=True, exist_ok=True)

    # Carga con validación de contrato (fail-closed).
    # No se usa pd.read_csv directo: el contrato valida antes de generar.
    print(f"[synthetic_iot] Leyendo dataset curated: {DATA_PATH}")
    df = load_raw_data(DATA_PATH, with_iot=False, stage="curated")

    print(f"[synthetic_iot] Shape curated: {df.shape}")

    # Generación sobre el curated, donde Attrition_Flag todavía es texto.
    # seed=SEED explícito. Nunca confiar en el default de la función.
    df_iot = add_synthetic_iot_features(
        df,
        target_col=TARGET,
        seed=SEED,
        **SYNTHETIC_IOT_PARAMS,
    )

    print(f"[synthetic_iot] Shape con IoT: {df_iot.shape}")

    missing_iot = [col for col in IOT_FEATURES if col not in df_iot.columns]
    assert not missing_iot, f"Faltan features IoT generadas: {missing_iot}"

    # Validación del contrato IoT sobre el resultado.
    # Si el generador produce algo fuera de bounds físicos, muere acá.
    validate(df_iot, stage="iot", with_iot=True)
    print("[synthetic_iot] ✅ Contrato IoT validado sobre el resultado")

    df_iot.to_csv(SYNTHETIC_IOT_CSV, index=False)
    print(f"[synthetic_iot] ✅ Dataset sintético guardado en {SYNTHETIC_IOT_CSV}")

    # ------------------------------------------------------------------
    # Metadata de trazabilidad con procedencia completa
    # ------------------------------------------------------------------
    metadata = {
        "seed": SEED,
        "target_col": TARGET,
        "synthetic_params": SYNTHETIC_IOT_PARAMS,
        "iot_features": IOT_FEATURES,
        "iot_reference": {k: list(v) for k, v in IOT_REFERENCE.items()},
        "input_csv": str(DATA_PATH),
        "input_sha256": sha(DATA_PATH),
        "output_csv": str(SYNTHETIC_IOT_CSV),
        "output_sha256": sha(SYNTHETIC_IOT_CSV),
        "contract_version": dc.CONTRACT_VERSION,
        "contract_sha256": sha(Path(dc.__file__)),
        "script_sha256": sha(Path(__file__)),
        "rows": int(df_iot.shape[0]),
        "cols": int(df_iot.shape[1]),
        "note": (
            "Features IoT sintéticas para demo Edge AI Banking IoT. "
            "No representan telemetría real. "
            "Generadas con referencia fija IOT_REFERENCE (medida sobre TRAIN "
            "del curated, ddof=1) y target_strength=0.0."
        ),
    }

    SYNTHETIC_IOT_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[synthetic_iot] ✅ Metadata guardada en {SYNTHETIC_IOT_METADATA}")

    # ------------------------------------------------------------------
    # Check rápido de distribución
    # ------------------------------------------------------------------
    if TARGET in df_iot.columns:
        print("\n[synthetic_iot] Media IoT por clase:")
        print(df_iot.groupby(TARGET)[IOT_FEATURES].mean().T)


if __name__ == "__main__":
    main()
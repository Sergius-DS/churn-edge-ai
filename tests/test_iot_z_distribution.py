# tests/test_iot_z_distribution.py
import pytest
import numpy as np
import pandas as pd
from src.config import SYNTHETIC_IOT_CSV, IOT_REFERENCE

@pytest.fixture(scope="module")
def z_scores():
    if not SYNTHETIC_IOT_CSV.exists():
        pytest.fail(f"Falta {SYNTHETIC_IOT_CSV.name}: regenerá con scripts/generate_synthetic_iot.py")
    df = pd.read_csv(SYNTHETIC_IOT_CSV)
    missing = [c for c in IOT_REFERENCE if c not in df.columns]
    if missing:
        pytest.fail(f"Columnas de referencia ausentes: {missing}")  # fail-closed, no skip
    results = {}
    for col, (mean, std) in IOT_REFERENCE.items():
        z = np.abs((df[col] - mean) / (std + 1e-9))
        results[col] = {"max": float(z.max()),
                        "frac_gt_3": float((z > 3).mean()),
                        "frac_gt_10": float((z > 10).mean())}
        print(f"{col:<32} max={results[col]['max']:6.2f}  "
              f">3={results[col]['frac_gt_3']:.4f}  >10={results[col]['frac_gt_10']:.4f}")
    assert results, "Gate vacuo: ninguna columna evaluada"
    return results

def test_no_extreme_saturation(z_scores):
    """El síntoma del +2727 no debe regresar."""
    assert z_scores["Total_Trans_Ct"]["max"] < 5.0

def test_batch_z_distribution_gate(z_scores):
    """Gate de LOTE, no por fila: 13.48 en los trends es cola legítima del cociente."""
    for col, s in z_scores.items():
        assert s["frac_gt_10"] < 0.01, f"{col}: {s['frac_gt_10']:.2%} con |z|>10"
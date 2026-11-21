# tests/test_data_contract.py
"""Grupo de control del audit: el gate rechaza el origen mutado y acepta el curado.

Las literales son la salida real de validate(raw), no una paráfrasis.
Los trends mueren en dtype (errors="raise" + continue), NO en bounds.
"""
import pandas as pd
import pytest

from src.config import DATA_PATH, RAW_ORIGIN_PATH
from src.data_contract import ContractViolation, validate


def test_gate_rejects_mutated_origin():
    df = pd.read_csv(RAW_ORIGIN_PATH)
    with pytest.raises(ContractViolation) as e:
        validate(df, stage="raw")
    msg = str(e.value)
    assert "[raw] 4 violaciones" in msg
    assert "Total_Amt_Chng_Q4_Q1: no numérico" in msg
    assert "Total_Ct_Chng_Q4_Q1: no numérico" in msg
    assert "Avg_Utilization_Ratio: 6880 fuera de [0.0,1.0]" in msg
    assert "INV-2 violada: 32.0628%" in msg


def test_gate_accepts_curated_ssot():
    """Control positivo: sin él, un validate() que rechaza todo pasa el test de arriba."""
    rep = validate(pd.read_csv(DATA_PATH), stage="curated")
    assert rep["n_rows"] == 9996
    assert rep["INV1_open_to_buy"] == 1.0
    assert rep["INV2_utilization"] == 1.0
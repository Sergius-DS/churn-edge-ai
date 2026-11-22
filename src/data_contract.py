"""
Contrato de ingestión fail-closed.
No imputa, no coerce, no clip: valida o rechaza.
Tres capturas independientes del defecto original: dtype, unidades, identidad.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACT_VERSION = "1.1.0"  # 1.0.0 base + IoT extension

# ───────────────────────────────────────────────────────────
# SCHEMA CANÓNICO (23 columnas del origen BankChurners)
# ───────────────────────────────────────────────────────────
NB_PREFIX = ("Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon"
             "_Dependent_count_Education_Level_Months_Inactive_12_mon_")
NB_COLUMNS = [NB_PREFIX + "1", NB_PREFIX + "2"]

EXPECTED_COLUMNS = [
    "CLIENTNUM", "Attrition_Flag", "Customer_Age", "Gender", "Dependent_count",
    "Education_Level", "Marital_Status", "Income_Category", "Card_Category",
    "Months_on_book", "Total_Relationship_Count", "Months_Inactive_12_mon",
    "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
    "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
    "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio",
] + NB_COLUMNS

# ───────────────────────────────────────────────────────────
# EXTENSIÓN IoT (6 columnas sintéticas Fase 2)
# ───────────────────────────────────────────────────────────
IOT_FEATURES = [
    "app_open_count_30d",
    "contactless_usage_count_30d",
    "payment_flow_abandon_count_30d",
    "nfc_failed_attempts_30d",
    "app_crash_count_30d",
    "pos_latency_p95_ms",
]

# Límites FÍSICOS: fuera de acá es imposible, no improbable → reject.
IOT_PHYSICAL_BOUNDS = {
    "app_open_count_30d":             (0, 365),
    "contactless_usage_count_30d":    (0, 500),
    "payment_flow_abandon_count_30d": (0, 100),
    "nfc_failed_attempts_30d":        (0, 100),
    "app_crash_count_30d":            (0, 100),
    "pos_latency_p95_ms":             (50, 20_000),
}

# Soporte del GENERADOR (λ caps + clip de latencia).
# No es un gate: es telemetría de saturación.
# Una feature pegada al techo transporta el techo, no señal.
IOT_GENERATOR_CAPS = {
    "app_open_count_30d":             80,
    "contactless_usage_count_30d":    60,
    "payment_flow_abandon_count_30d": 20,
    "nfc_failed_attempts_30d":        12,
    "app_crash_count_30d":            10,
    "pos_latency_p95_ms":             6000,
}

# ───────────────────────────────────────────────────────────
# BOUNDS NUMÉRICOS (unidades por columna)
# El ×1000 muere acá: 999 > 1.0, 3714 > 5.0.
# ───────────────────────────────────────────────────────────
NUMERIC_BOUNDS = {
    "Customer_Age":              (18, 100),
    "Dependent_count":           (0, 5),
    "Months_on_book":            (1, 60),
    "Total_Relationship_Count":  (1, 6),
    "Months_Inactive_12_mon":    (0, 6),
    "Contacts_Count_12_mon":     (0, 6),
    "Credit_Limit":              (1_000.0, 40_000.0),
    "Total_Revolving_Bal":       (0, 3_000),
    "Avg_Open_To_Buy":           (0.0, 40_000.0),
    "Total_Amt_Chng_Q4_Q1":      (0.0, 5.0),
    "Total_Trans_Amt":           (0, 25_000),
    "Total_Trans_Ct":            (0, 200),
    "Total_Ct_Chng_Q4_Q1":       (0.0, 5.0),
    "Avg_Utilization_Ratio":     (0.0, 1.0),
}

# ───────────────────────────────────────────────────────────
# CATEGORÍAS VÁLIDAS
# ───────────────────────────────────────────────────────────
CATEGORIES = {
    "Attrition_Flag":  {"Existing Customer", "Attrited Customer"},
    "Gender":          {"M", "F"},
    "Education_Level": {"Uneducated", "High School", "College", "Graduate",
                        "Post-Graduate", "Doctorate", "Unknown"},
    "Marital_Status":  {"Single", "Married", "Divorced", "Unknown"},
    "Income_Category": {"Less than $40K", "$40K - $60K", "$60K - $80K",
                        "$80K - $120K", "$120K +", "Unknown"},
    "Card_Category":   {"Blue", "Silver", "Gold", "Platinum"},
}

# ───────────────────────────────────────────────────────────
# SSOT COMPARTIDOS
# ───────────────────────────────────────────────────────────
FORBIDDEN_FOR_TRAINING = ["CLIENTNUM"] + NB_COLUMNS
TARGET_MAP = {"Attrited Customer": 1, "Existing Customer": 0}


class ContractViolation(ValueError):
    """El dato viola el contrato. Fail-closed: no se continúa."""
    pass


# ───────────────────────────────────────────────────────────
# VALIDACIÓN PRINCIPAL
# ───────────────────────────────────────────────────────────
def validate(df: pd.DataFrame, *, stage: str = "curated",
             with_iot: bool = False) -> dict:
    """
    Valida o levanta ContractViolation con TODAS las violaciones, no la primera.

    Parámetros:
        df:       DataFrame a validar.
        stage:    etiqueta para el mensaje de error ("raw", "curated", "iot").
        with_iot: True si se esperan las 6 columnas IoT adicionales.

    Retorna:
        dict de telemetría (solo si pasa).
    """
    errs: list[str] = []
    report: dict = {}

    # ── Schema ──────────────────────────────────────────────
    expected = EXPECTED_COLUMNS + (IOT_FEATURES if with_iot else [])
    missing = set(expected) - set(df.columns)
    extra = set(df.columns) - set(expected)
    if missing:
        errs.append(f"columnas faltantes: {sorted(missing)}")
    if extra:
        errs.append(f"columnas no declaradas: {sorted(extra)}")
    if errs:
        raise ContractViolation(f"[{stage}] schema: " + " | ".join(errs))

    # ── Unicidad CLIENTNUM ──────────────────────────────────
    if not df["CLIENTNUM"].is_unique:
        errs.append("CLIENTNUM no es único")

    # ── Bounds numéricos + dtype ────────────────────────────
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        s = df[col]
        # errors="raise": las fechas mueren acá, no se vuelven NaN silencioso.
        try:
            s = pd.to_numeric(s, errors="raise")
        except (ValueError, TypeError) as e:
            errs.append(f"{col}: no numérico ({e})")
            continue
        if s.isna().any():
            errs.append(f"{col}: {int(s.isna().sum())} NaN")
            continue
        bad = ~s.between(lo, hi)
        if bad.any():
            errs.append(f"{col}: {int(bad.sum())} fuera de [{lo},{hi}] "
                        f"(min={s.min():.4g} max={s.max():.4g})")
        report[col] = {"min": float(s.min()), "max": float(s.max()),
                       "mean": float(s.mean()), "std": float(s.std(ddof=1))}

    # ── Categorías ──────────────────────────────────────────
    for col, allowed in CATEGORIES.items():
        unknown = set(df[col].astype(str)) - allowed
        if unknown:
            errs.append(f"{col}: categorías no declaradas {sorted(unknown)}")

    # ── INV-1: Open_To_Buy == Limit − Revolving ─────────────
    otb = df["Credit_Limit"] - df["Total_Revolving_Bal"]
    r1 = float(np.isclose(otb, df["Avg_Open_To_Buy"], atol=0.01).mean())
    report["INV1_open_to_buy"] = r1
    if r1 < 1.0:
        errs.append(f"INV-1 violada: {r1:.4%}")

    # ── INV-2: Utilización es derivada, no primaria ─────────
    rebuilt = (df["Total_Revolving_Bal"] / df["Credit_Limit"]).round(3)
    r2 = float(np.isclose(df["Avg_Utilization_Ratio"], rebuilt, atol=0.0011).mean())
    report["INV2_utilization"] = r2
    if r2 < 1.0:
        errs.append(f"INV-2 violada: {r2:.4%}")

    # ── IoT: bounds físicos + telemetría de saturación ──────
    if with_iot:
        sat = {}
        for col, (lo, hi) in IOT_PHYSICAL_BOUNDS.items():
            try:
                s = pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError) as e:
                errs.append(f"{col}: no numérico ({e})")
                continue
            if s.isna().any():
                errs.append(f"{col}: {int(s.isna().sum())} NaN")
                continue
            bad = ~s.between(lo, hi)
            if bad.any():
                errs.append(f"{col}: {int(bad.sum())} fuera de [{lo},{hi}] "
                            f"(min={s.min():.4g} max={s.max():.4g})")
            # Saturación contra el cap del generador (no contra el bound ancho)
            cap = IOT_GENERATOR_CAPS[col]
            sat[col] = {
                "at_cap_frac": float((s >= cap).mean()),
                "max": float(s.max()),
                "cap": cap,
            }
        report["iot_saturation"] = sat

    # ── Veredicto ───────────────────────────────────────────
    if errs:
        raise ContractViolation(
            f"[{stage}] {len(errs)} violaciones:\n  - " + "\n  - ".join(errs)
        )

    report["contract_version"] = CONTRACT_VERSION
    report["n_rows"] = int(len(df))
    report["with_iot"] = with_iot
    return report
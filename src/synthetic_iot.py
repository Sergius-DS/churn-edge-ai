"""
src/synthetic_iot.py
────────────────────
Generación de features IoT sintéticas.

Reglas importantes:
- Usa IOT_REFERENCE como referencia fija.
- No usa estadísticas del dataset completo.
- target_strength default = 0.0.
- noise_scale default = 0.65.
"""

import numpy as np
import pandas as pd

from src.config import IOT_REFERENCE


def z_ref(series: pd.Series, col: str) -> pd.Series:
    """
    Normalización con referencia fija.

    Esto permite generar features para:
    - dataset completo,
    - train,
    - test,
    - una sola fila en mock bank,

    sin depender de estadísticas del lote actual.
    """
    if col not in IOT_REFERENCE:
        raise KeyError(f"Falta referencia fija para la columna: {col}")

    mean, std = IOT_REFERENCE[col]

    s = pd.to_numeric(series, errors="coerce")
    s = s.fillna(mean)

    return (s - mean) / (std + 1e-9)


def _get_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Si la columna existe, la usa.
    Si no existe, devuelve una serie con la media de referencia.
    """
    if col in df.columns:
        return df[col]

    mean, _ = IOT_REFERENCE.get(col, (0.0, 1.0))
    return pd.Series(mean, index=df.index)


def _get_churn(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Convierte target a binario si existe.
    Solo se usa si target_strength > 0.
    """
    if target_col not in df.columns:
        return pd.Series(0.0, index=df.index)

    s = df[target_col].astype(str).str.lower().str.strip()

    churn = (
        s.str.contains("attrit", regex=False)
        .astype(float)
    )

    return churn


def add_synthetic_iot_features(
    df: pd.DataFrame,
    target_col: str = "Attrition_Flag",
    seed: int = 42,
    target_strength: float = 0.0,
    noise_scale: float = 0.65,
) -> pd.DataFrame:
    """
    Agrega features IoT sintéticas al dataset.

    Recomendación:
    - target_strength = 0.0 para evitar target leakage directo.
    - noise_scale entre 0.55 y 0.80.
    """
    rng = np.random.default_rng(seed)

    df = df.copy()
    n = len(df)

    # ------------------------------------------------------------------
    # Churn solo se usa si target_strength > 0.
    # Recomendado: target_strength = 0.0
    # ------------------------------------------------------------------
    churn = _get_churn(df, target_col)

    if target_strength > 0:
        churn_z = (churn - churn.mean()) / (churn.std() + 1e-9)
    else:
        churn_z = pd.Series(0.0, index=df.index)

    # ------------------------------------------------------------------
    # Señales base con referencia fija
    # ------------------------------------------------------------------
    trans_activity = z_ref(_get_series(df, "Total_Trans_Ct"), "Total_Trans_Ct")
    amount_activity = z_ref(_get_series(df, "Total_Trans_Amt"), "Total_Trans_Amt")
    inactive = z_ref(_get_series(df, "Months_Inactive_12_mon"), "Months_Inactive_12_mon")
    contacts = z_ref(_get_series(df, "Contacts_Count_12_mon"), "Contacts_Count_12_mon")
    utilization = z_ref(_get_series(df, "Avg_Utilization_Ratio"), "Avg_Utilization_Ratio")

    trans_trend = z_ref(_get_series(df, "Total_Ct_Chng_Q4_Q1"), "Total_Ct_Chng_Q4_Q1")
    amount_trend = z_ref(_get_series(df, "Total_Amt_Chng_Q4_Q1"), "Total_Amt_Chng_Q4_Q1")
    relationship = z_ref(_get_series(df, "Total_Relationship_Count"), "Total_Relationship_Count")

    # ------------------------------------------------------------------
    # Riesgo conductual basado en comportamiento observable.
    # No usa churn si target_strength = 0.
    # ------------------------------------------------------------------
    behavioral_risk = (
        0.28 * inactive
        + 0.18 * contacts
        + 0.10 * utilization
        - 0.30 * trans_activity
        - 0.16 * amount_activity
        - 0.06 * trans_trend
        - 0.04 * amount_trend
        - 0.05 * relationship
        + 0.35 * rng.normal(0, 1, n)
    )

    if target_strength > 0:
        behavioral_risk = behavioral_risk + target_strength * churn_z

    # ------------------------------------------------------------------
    # Engagement digital
    # ------------------------------------------------------------------
    digital_engagement = (
        0.32 * trans_activity
        + 0.22 * amount_activity
        + 0.10 * trans_trend
        + 0.08 * amount_trend
        + 0.05 * relationship
        - 0.18 * inactive
        - 0.08 * contacts
        - 0.22 * behavioral_risk
        + noise_scale * rng.normal(0, 1, n)
    )

    # ------------------------------------------------------------------
    # Fricción digital
    # ------------------------------------------------------------------
    digital_friction = (
        0.45 * behavioral_risk
        + 0.20 * inactive
        + 0.14 * contacts
        + 0.10 * utilization
        - 0.12 * trans_activity
        - 0.06 * trans_trend
        + noise_scale * rng.normal(0, 1, n)
    )

    # ------------------------------------------------------------------
    # Calidad técnica independiente
    # ------------------------------------------------------------------
    app_quality = rng.normal(0, 0.45, n)
    terminal_quality = rng.normal(0, 0.55, n)

    def bounded_poisson(
        log_lambda: np.ndarray,
        min_lambda: float,
        max_lambda: float,
    ) -> np.ndarray:
        """
        Poisson con control de rango usando log-lambda.
        """
        log_lambda = np.clip(
            log_lambda,
            np.log(max(min_lambda, 1e-3)),
            np.log(max_lambda),
        )

        lam = np.exp(log_lambda)
        return rng.poisson(lam).astype(int)

    # ------------------------------------------------------------------
    # Features IoT
    # ------------------------------------------------------------------

    df["app_open_count_30d"] = bounded_poisson(
        np.log(14.0)
        + 0.42 * digital_engagement
        - 0.28 * digital_friction
        + 0.22 * app_quality
        + 0.18 * rng.normal(0, 1, n),
        min_lambda=0.05,
        max_lambda=80.0,
    )

    df["contactless_usage_count_30d"] = bounded_poisson(
        np.log(8.0)
        + 0.36 * digital_engagement
        - 0.18 * digital_friction
        + 0.18 * app_quality
        + 0.18 * rng.normal(0, 1, n),
        min_lambda=0.05,
        max_lambda=60.0,
    )

    df["payment_flow_abandon_count_30d"] = bounded_poisson(
        np.log(1.0)
        + 0.35 * digital_friction
        - 0.10 * app_quality
        + 0.25 * rng.normal(0, 1, n),
        min_lambda=0.05,
        max_lambda=20.0,
    )

    df["nfc_failed_attempts_30d"] = bounded_poisson(
        np.log(0.6)
        + 0.30 * digital_friction
        - 0.12 * terminal_quality
        + 0.22 * rng.normal(0, 1, n),
        min_lambda=0.03,
        max_lambda=12.0,
    )

    df["app_crash_count_30d"] = bounded_poisson(
        np.log(0.4)
        + 0.28 * digital_friction
        - 0.18 * app_quality
        + 0.22 * rng.normal(0, 1, n),
        min_lambda=0.02,
        max_lambda=10.0,
    )

    log_latency = (
        np.log(700.0)
        + 0.24 * digital_friction
        - 0.18 * terminal_quality
        + 0.16 * rng.normal(0, 1, n)
    )

    df["pos_latency_p95_ms"] = np.round(
        np.exp(
            np.clip(
                log_latency,
                np.log(200),
                np.log(6000),
            )
        )
    ).astype(int)

    return df

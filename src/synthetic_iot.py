"""
src/synthetic_iot.py
────────────────────
Generación de features IoT sintéticas.

Reglas:
- IOT_REFERENCE como referencia fija (medida sobre TRAIN del curated).
- NO usa estadísticas del dataset completo.
- target_strength default = 0.0 (cero leakage de target).
- noise_scale default = 0.65.
- seed default = SEED de config (44), NO hardcodeado.
- z_ref con gate de soporte: detecta referencia de otra población.
- Sin errors="coerce", sin fillna, sin fallback silencioso.
"""

import numpy as np
import pandas as pd

from src.config import IOT_REFERENCE, SEED, TARGET
from src.data_contract import TARGET_MAP, ContractViolation


def z_ref(series: pd.Series, col: str, *, strict: bool = True) -> pd.Series:
    """
    Normalización con referencia fija.

    Independiente de estadísticas del lote actual:
    - dataset completo, train, test, una sola fila.

    strict=True (default):
      Verifica que la media de referencia esté dentro del soporte
      observado. Si no lo está, la referencia describe otra población.
      Gate de una línea que habría bloqueado (140, 58) el día uno.

    strict=False:
      Solo para scripts forenses. Nunca en el pipeline de entrenamiento.
    """
    if col not in IOT_REFERENCE:
        raise KeyError(f"Falta referencia fija para: {col}")

    mean, std = IOT_REFERENCE[col]

    # errors="raise": si hay fechas o strings, muere acá, no se vuelve NaN.
    s = pd.to_numeric(series, errors="raise")

    if s.isna().any():
        raise ValueError(
            f"{col}: {int(s.isna().sum())} NaN — el contrato debió rechazarlo"
        )

    # Gate de soporte: la referencia debe describir la población observada.
    # len(s) > 1 protege el caso de una sola fila (mock bank), donde
    # el soporte es un punto y el chequeo no aplica.
    if strict and len(s) > 1 and not (s.min() <= mean <= s.max()):
        raise ValueError(
            f"{col}: media de referencia {mean} fuera del soporte "
            f"[{s.min()}, {s.max()}] — la referencia describe otra población"
        )

    return (s - mean) / (std + 1e-9)


def _get_series(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Si la columna existe, la usa.
    Si no existe, FAIL. No hay fallback a la media de referencia:
    eso es el mismo disfraz que el clipping sin flag.
    """
    if col not in df.columns:
        raise KeyError(
            f"Columna '{col}' ausente del DataFrame. "
            "El generador IoT requiere las 8 columnas base del curated."
        )
    return df[col]


def _get_churn(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Única derivación del target: TARGET_MAP del contrato.
    Elimina la derivación duplicada por str.contains("attrit").
    """
    if target_col not in df.columns:
        return pd.Series(0.0, index=df.index)

    y = df[target_col].astype(str).str.strip().map(TARGET_MAP)

    if y.isna().any():
        unmapped = df[target_col].astype(str).str.strip()[y.isna()].unique()
        raise ContractViolation(
            f"Target '{target_col}' con categorías sin mapear: {unmapped}"
        )

    return y.astype(float)


def add_synthetic_iot_features(
    df: pd.DataFrame,
    target_col: str = TARGET,
    seed: int = SEED,
    target_strength: float = 0.0,
    noise_scale: float = 0.65,
) -> pd.DataFrame:
    """
    Agrega features IoT sintéticas al dataset.

    Recomendación:
    - target_strength = 0.0 para evitar target leakage directo.
    - noise_scale entre 0.55 y 0.80.
    - seed = SEED de config (44). Nunca hardcodear.
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
    # Señales base con referencia fija (gate de soporte activo)
    # ------------------------------------------------------------------
    trans_activity   = z_ref(_get_series(df, "Total_Trans_Ct"),            "Total_Trans_Ct")
    amount_activity  = z_ref(_get_series(df, "Total_Trans_Amt"),           "Total_Trans_Amt")
    inactive         = z_ref(_get_series(df, "Months_Inactive_12_mon"),    "Months_Inactive_12_mon")
    contacts         = z_ref(_get_series(df, "Contacts_Count_12_mon"),     "Contacts_Count_12_mon")
    utilization      = z_ref(_get_series(df, "Avg_Utilization_Ratio"),     "Avg_Utilization_Ratio")
    trans_trend      = z_ref(_get_series(df, "Total_Ct_Chng_Q4_Q1"),       "Total_Ct_Chng_Q4_Q1")
    amount_trend     = z_ref(_get_series(df, "Total_Amt_Chng_Q4_Q1"),      "Total_Amt_Chng_Q4_Q1")
    relationship     = z_ref(_get_series(df, "Total_Relationship_Count"),  "Total_Relationship_Count")

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
    app_quality      = rng.normal(0, 0.45, n)
    terminal_quality = rng.normal(0, 0.55, n)

    def bounded_poisson(
        log_lambda: np.ndarray,
        min_lambda: float,
        max_lambda: float,
    ) -> np.ndarray:
        """Poisson con control de rango usando log-lambda."""
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
        min_lambda=0.05, max_lambda=80.0,
    )

    df["contactless_usage_count_30d"] = bounded_poisson(
        np.log(8.0)
        + 0.36 * digital_engagement
        - 0.18 * digital_friction
        + 0.18 * app_quality
        + 0.18 * rng.normal(0, 1, n),
        min_lambda=0.05, max_lambda=60.0,
    )

    df["payment_flow_abandon_count_30d"] = bounded_poisson(
        np.log(1.0)
        + 0.35 * digital_friction
        - 0.10 * app_quality
        + 0.25 * rng.normal(0, 1, n),
        min_lambda=0.05, max_lambda=20.0,
    )

    df["nfc_failed_attempts_30d"] = bounded_poisson(
        np.log(0.6)
        + 0.30 * digital_friction
        - 0.12 * terminal_quality
        + 0.22 * rng.normal(0, 1, n),
        min_lambda=0.03, max_lambda=12.0,
    )

    df["app_crash_count_30d"] = bounded_poisson(
        np.log(0.4)
        + 0.28 * digital_friction
        - 0.18 * app_quality
        + 0.22 * rng.normal(0, 1, n),
        min_lambda=0.02, max_lambda=10.0,
    )

    log_latency = (
        np.log(700.0)
        + 0.24 * digital_friction
        - 0.18 * terminal_quality
        + 0.16 * rng.normal(0, 1, n)
    )

    df["pos_latency_p95_ms"] = np.round(
        np.exp(np.clip(log_latency, np.log(200), np.log(6000)))
    ).astype(int)

    return df
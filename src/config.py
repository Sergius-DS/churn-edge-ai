"""
src/config.py
─────────────
Constantes globales del proyecto.
Cualquier cambio de hiperparámetros, rutas o mappings se hace AQUÍ.

SSOT de datos: data/curated/churn_curated.csv (no el raw).
SSOT de contrato: src/data_contract.py (una sola fuente por hecho).
"""

from pathlib import Path

# Una sola fuente por hecho
from src.data_contract import (
    FORBIDDEN_FOR_TRAINING as LEAKAGE_COLUMNS,
    TARGET_MAP,
    IOT_FEATURES,
    IOT_PHYSICAL_BOUNDS as IOT_FEATURE_BOUNDS,
)


# ============================================================
# RUTAS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# El origen queda para forense; la SSOT operativa es el curated
RAW_ORIGIN_PATH = PROJECT_ROOT / "data" / "raw" / "Credit_card_churn.csv"
DATA_PATH       = PROJECT_ROOT / "data" / "curated" / "churn_curated.csv"

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
SYNTHETIC_PATH = PROJECT_ROOT / "data" / "synthetic"

SYNTHETIC_IOT_CSV      = PROCESSED_PATH / "credit_card_churn_iot_synthetic.csv"
SYNTHETIC_IOT_METADATA = SYNTHETIC_PATH / "iot_features_metadata.json"

# namespace; padre de edge_iot. El directorio raíz es un nombre, no un destino.
ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"
EDGE_DIR      = PROJECT_ROOT / "edge"

EDGE_IOT_ARTIFACTS_DIR = ARTIFACTS_DIR / "edge_iot"

EDGE_IOT_MODEL_JSON    = EDGE_DIR / "xgb_model_edge_iot.json"
EDGE_IOT_CONTRACT_JSON = EDGE_DIR / "inference_contract_edge_iot.json"


# ============================================================
# REPRODUCIBILIDAD
# ============================================================
SEED   = 44
TARGET = "Attrition_Flag"


# ============================================================
# LIMPIEZA DE DATOS
# ============================================================
# LEAKAGE_COLUMNS viene del contrato (import arriba).
# NUMERIC_LIKE_COLUMNS ELIMINADO.
#   Fósil del síntoma normalizado. Registrado en ADR-0005.
#   El contrato garantiza dtype; la coerción era un disfraz.


# ============================================================
# MAPEOS ORDINALES
# ============================================================
INCOME_ORDER = {
    "Less than $40K": 0,
    "$40K - $60K": 1,
    "$60K - $80K": 2,
    "$80K - $120K": 3,
    "$120K +": 4,
    "Unknown": None,
}

EDUCATION_ORDER = {
    "Uneducated": 0,
    "High School": 1,
    "College": 2,
    "Graduate": 3,
    "Post-Graduate": 4,
    "Doctorate": 5,
    "Unknown": None,
}

ORDINAL_SOURCE_COLUMNS = ["Income_Category", "Education_Level"]


# ============================================================
# SPLIT
# ============================================================
TEST_SIZE = 0.20


# ============================================================
# XGBOOST — Modelo EDGE IoT (único linaje vivo)
# ============================================================
XGB_EDGE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 3,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_lambda": 1.5,
    "seed": SEED,
    "verbosity": 1,
}

EDGE_CV_NUM_BOOST_ROUND    = 200
EDGE_EARLY_STOPPING_ROUNDS = 10

# Consumidor vivo: train_with_cv (nfold). No pertenece a ningún linaje retirado.
CV_NFOLD = 5


# ============================================================
# REFERENCIA FIJA PARA NORMALIZACIÓN IoT
# ============================================================
# Medida sobre TRAIN del curated, ddof=1.
# Procedencia: data/curated/iot_reference_provenance.json
# curated sha256 6bced2a3bf04db7b · n_train=7996 · seed=44
IOT_REFERENCE = {
    "Total_Trans_Ct":            (65.2112, 23.3703),
    "Total_Trans_Amt":           (4446.1595, 3412.3573),
    "Months_Inactive_12_mon":    (2.3488, 1.0054),
    "Contacts_Count_12_mon":     (2.4577, 1.1089),
    "Avg_Utilization_Ratio":     (0.2764, 0.2775),
    "Total_Ct_Chng_Q4_Q1":       (0.7058, 0.2231),
    "Total_Amt_Chng_Q4_Q1":      (0.7576, 0.2167),
    "Total_Relationship_Count":  (3.8034, 1.5517),
}


# ============================================================
# PARÁMETROS DE GENERACIÓN SINTÉTICA IoT
# ============================================================
SYNTHETIC_IOT_PARAMS = {
    "target_strength": 0.0,
    "noise_scale": 0.65,
}


# ============================================================
# THRESHOLD DE NEGOCIO
# ============================================================
# ELIMINADO: BEST_THRESHOLD = 0.38 (dueño: modelo Full, retirado en ADR-0006).
# El umbral del linaje vivo vive en model_artifacts/edge_iot/model_config.joblib
# y se justifica en edge/metrics_edge_iot.json. Una fuente, un dueño, con recibo.


# ============================================================
# PARITY / EXPORT
# ============================================================
PARITY_SAMPLE_SIZE = 100
PARITY_TOLERANCE   = 1e-4
# Piso de sensibilidad del golden sensible: p(1-p) >= 0.15 <=> p en [0.1838, 0.8162].
# 60% del maximo teorico (0.25). El fixture lo declara; Rust lo lee del fixture.
SENSITIVITY_FLOOR  = 0.15


# ============================================================
# CONTRATO / PRIVACIDAD
# ============================================================
FORBIDDEN_FIELDS = [
    "pan",
    "cvv",
    "card_number",
    "account_number",
    "gps_exact",
    "latitude",
    "longitude",
    "device_id",
    "raw_merchant_name",
]
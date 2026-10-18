"""
src/config.py
─────────────
Constantes globales del proyecto.
Cualquier cambio de hiperparámetros, rutas o mappings se hace AQUÍ.
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════
# RUTAS
# ═══════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# DATA_PATH = PROJECT_ROOT / "Credit_card_churn.csv"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Credit_card_churn.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"
EDGE_DIR = PROJECT_ROOT / "edge"

# ═══════════════════════════════════════════════════════════
# REPRODUCIBILIDAD
# ═══════════════════════════════════════════════════════════
SEED = 44
TARGET = "Attrition_Flag"

# ═══════════════════════════════════════════════════════════
# LIMPIEZA DE DATOS
# ═══════════════════════════════════════════════════════════
LEAKAGE_COLUMNS = [
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

# Columnas que parecen numéricas pero vienen como object
NUMERIC_LIKE_COLUMNS = [
    "Total_Amt_Chng_Q4_Q1",
    "Total_Ct_Chng_Q4_Q1",
]

# ═══════════════════════════════════════════════════════════
# MAPEOS ORDINALES
# ═══════════════════════════════════════════════════════════
INCOME_ORDER = {
    "Less than $40K": 0,
    "$40K - $60K": 1,
    "$60K - $80K": 2,
    "$80K - $120K": 3,
    "$120K +": 4,
    "Unknown": None,  # → NaN → imputar con mediana
}

EDUCATION_ORDER = {
    "Uneducated": 0,
    "High School": 1,
    "College": 2,
    "Graduate": 3,
    "Post-Graduate": 4,
    "Doctorate": 5,
    "Unknown": None,  # → NaN → imputar con mediana
}

# Columnas que se transforman a ordinal y luego se eliminan
ORDINAL_SOURCE_COLUMNS = ["Income_Category", "Education_Level"]

# ═══════════════════════════════════════════════════════════
# SPLIT
# ═══════════════════════════════════════════════════════════
TEST_SIZE = 0.20

# ═══════════════════════════════════════════════════════════
# XGBOOST — Modelo FULL
# ═══════════════════════════════════════════════════════════
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "seed": SEED,
    "verbosity": 1,
}

CV_NUM_BOOST_ROUND = 500
CV_NFOLD = 5
CV_EARLY_STOPPING_ROUNDS = 8

# ═══════════════════════════════════════════════════════════
# XGBOOST — Modelo EDGE (Opción A: optimizado)
# ═══════════════════════════════════════════════════════════
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

EDGE_CV_NUM_BOOST_ROUND = 200
EDGE_EARLY_STOPPING_ROUNDS = 10

# Top features para el modelo edge (se actualizan tras feature importance)
EDGE_TOP_FEATURES = [
    "Total_Trans_Ct",
    "Total_Revolving_Bal",
    "Total_Ct_Chng_Q4_Q1",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Total_Trans_Amt",
    "Total_Relationship_Count",
    "Avg_Utilization_Ratio",
    "Total_Amt_Chng_Q4_Q1",
    "Dependent_count",
    "Gender_F",
    "Marital_Status_Single",
]

# ═══════════════════════════════════════════════════════════
# THRESHOLD DE NEGOCIO
# ═══════════════════════════════════════════════════════════
BEST_THRESHOLD = 0.38

# ═══════════════════════════════════════════════════════════
# PARITY / EXPORT
# ═══════════════════════════════════════════════════════════
PARITY_SAMPLE_SIZE = 100
PARITY_TOLERANCE = 1e-4

# ═══════════════════════════════════════════════════════════
# DIRECTORIOS EDGE (modelo liviano)
# ═══════════════════════════════════════════════════════════
EDGE_ARTIFACTS_DIR = ARTIFACTS_DIR / "edge"
"""
src/data_loading.py
───────────────────
Carga → validación de contrato (fail-closed) → drop leakage → target → split.
NO coerciona. NO imputa. NO clip. Si el dato viola el contrato, se rechaza acá.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, TARGET, SEED, TEST_SIZE, SYNTHETIC_IOT_CSV
from src.data_contract import (
    validate, TARGET_MAP, FORBIDDEN_FOR_TRAINING, IOT_FEATURES, ContractViolation,
)


def load_raw_data(path: Path = DATA_PATH, *, with_iot: bool = False,
                  validate_contract: bool = True, stage: str = "curated") -> pd.DataFrame:
    """
    Carga y valida. validate_contract=False es un bypass EXPLÍCITO,
    reservado a scripts forenses. Nunca al pipeline de entrenamiento.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    df = pd.read_csv(path)
    print(f"[data_loading] Shape inicial: {df.shape}  ({path.name})")

    if validate_contract:
        rep = validate(df, stage=stage, with_iot=with_iot)
        print(f"[data_loading] ✅ Contrato v{rep['contract_version']} OK | "
              f"INV-1 {rep['INV1_open_to_buy']:.2%} · INV-2 {rep['INV2_utilization']:.2%}")
    else:
        print("[data_loading] ⚠️  BYPASS de contrato (modo forense)")
    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """sorted() para logs deterministas: set() varía entre procesos."""
    nb = [c for c in df.columns if "naive_bayes_classifier" in c.lower()]
    cols = sorted(set(FORBIDDEN_FOR_TRAINING) | set(nb))
    present = [c for c in cols if c in df.columns]
    out = df.drop(columns=present).copy()
    print(f"[data_loading] Leakage/ID eliminadas: {present} → {out.shape}")
    return out


def encode_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Única derivación del target: TARGET_MAP del contrato."""
    if TARGET not in df.columns:
        raise ContractViolation(f"No se encontró la columna objetivo: {TARGET}")
    y_raw = df[TARGET]
    X = df.drop(columns=[TARGET])

    if y_raw.dtype == "object":
        y = y_raw.astype(str).str.strip().map(TARGET_MAP)
        unmapped = y_raw.astype(str).str.strip()[y.isna()].unique()
        if len(unmapped):
            raise ContractViolation(f"Categorías sin mapear en {TARGET}: {unmapped}")
        y = y.astype(int)
    else:
        y = pd.to_numeric(y_raw, errors="raise").astype(int)

    invalid = set(y.unique()) - {0, 1}
    if invalid:
        raise ContractViolation(f"{TARGET} con valores fuera de {{0,1}}: {sorted(invalid)}")
    print(f"[data_loading] Target: {y.value_counts().to_dict()} "
          f"| prevalencia={y.mean():.5f}")
    return X, y


def split_data(X, y, test_size: float = TEST_SIZE, seed: int = SEED):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    print(f"[data_loading] X_train {Xtr.shape} | X_test {Xte.shape}")
    return Xtr, Xte, ytr, yte


def verify_no_leakage(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    for name, X in (("train", X_train), ("test", X_test)):
        bad = [c for c in X.columns
               if "naive_bayes" in c.lower() or "attrition_flag" in c.lower()
               or c in FORBIDDEN_FOR_TRAINING]
        if bad:
            raise ContractViolation(f"Fuga de datos en {name}: {bad}")
    print("[data_loading] ✅ Sin columnas con fuga (train y test)")


def load_and_prepare(path: Path = None, with_iot: bool = False):
    """
    SSOT: data/curated/churn_curated.csv (o el sintético si with_iot=True).
    Las coerciones NUMERIC_LIKE / IoT fueron ELIMINADAS: el contrato garantiza dtype.
    """
    if path is not None and with_iot:
        raise ValueError("No uses path y with_iot=True al mismo tiempo.")
    if path is None:
        path = SYNTHETIC_IOT_CSV if with_iot else DATA_PATH
    path = Path(path)

    if with_iot and not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset IoT: {path}. "
            "Ejecuta: python scripts/generate_synthetic_iot.py")

    df = load_raw_data(path, with_iot=with_iot,
                       stage="iot" if with_iot else "curated")
    X, y = encode_target(drop_leakage_columns(df))
    Xtr, Xte, ytr, yte = split_data(X, y)

    if with_iot:
        for name, X in (("train", Xtr), ("test", Xte)):
            miss = [c for c in IOT_FEATURES if c not in X.columns]
            if miss:
                raise ContractViolation(f"Faltan IoT features en {name}: {miss}")

    verify_no_leakage(Xtr, Xte)
    assert Xtr.shape[0] == ytr.shape[0] and Xte.shape[0] == yte.shape[0]
    return Xtr, Xte, ytr, yte
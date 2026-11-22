"""
Curación R1–R5. Read-only sobre data/raw/. Escribe en data/curated/.
Determinista: mismo sha256(raw) + mismo sha256(script) -> mismo sha256(salida).
"""
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_ORIGIN_PATH, PROJECT_ROOT
import src.data_contract as dc 
from src.data_contract import validate, CONTRACT_VERSION

RULES_VERSION = "1.0.0"
CURATED_DIR = PROJECT_ROOT / "data" / "curated"
TRENDS = ["Total_Ct_Chng_Q4_Q1", "Total_Amt_Chng_Q4_Q1"]
AMBIG_SUPPORT_CEIL = 0.004      # frontera de la hipótesis alternativa
COUNT_RATIO_FLOOR = 1 / 60      # piso de dominio: cociente de conteos enteros


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inverse_parse(s: pd.Series):
    """Regla certificada sobre Avg_Utilization_Ratio (10.127/10.127 por INV-2)."""
    v = pd.to_numeric(s, errors="coerce")          # fechas -> NaN, contabilizadas
    is_int = v.notna() & (v == np.floor(v))
    stripped = is_int & (v >= 4)
    ambiguous = is_int & v.between(1, 3)
    repaired = v.where(~stripped, v / 1000.0).round(3)
    return repaired, v, stripped, ambiguous


def main() -> None:
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_ORIGIN_PATH)
    rules: dict = {}

    # ── R1: reconstrucción desde primarias (no inversión) ──────────────
    rebuilt = (raw["Total_Revolving_Bal"] / raw["Credit_Limit"]).round(3)
    obs_rep, _, u_stripped, u_amb = inverse_parse(raw["Avg_Utilization_Ratio"])
    exact = float((rebuilt == obs_rep).mean())
    near = float(np.isclose(rebuilt, obs_rep, atol=0.0011).mean())
    if near < 1.0 or exact < 0.99:
        raise SystemExit(f"R1 no certificada: exact={exact:.4%} near={near:.4%}")
    if int(u_amb.sum()) != 0:
        raise SystemExit("R1: zona ambigua no vacía en utilización")
    rules["R1_utilization_rebuilt"] = {
        "source_columns": ["Total_Revolving_Bal", "Credit_Limit"],
        "rows": len(raw), "inverted_crosscheck": int(u_stripped.sum()),
        "exact_match_vs_inverse_parse": exact, "within_0.0011": near,
        "imputed_values": 0,
    }

    # ── R2/R3: inversa certificada + precondiciones de dominio ─────────
    repaired = {}
    for col in TRENDS:
        rep, v, stripped, amb = inverse_parse(raw[col])
        nz = rep[rep > 0]
        p1 = int(((rep > 0) & (rep < AMBIG_SUPPORT_CEIL)).sum())
        floor = float(nz.min())
        other = [c for c in TRENDS if c != col][0]
        o_rep, _, _, _ = inverse_parse(raw[other])
        paired = float(o_rep.loc[amb.index].mean())
        glob = float(o_rep.mean())

        pre = {
            "P1_support_below_0.004_empty": p1 == 0,
            "P2_floor_above_alternative": floor > AMBIG_SUPPORT_CEIL,
            "P3_count_ratio_domain": (col != "Total_Ct_Chng_Q4_Q1"
                                      or floor >= COUNT_RATIO_FLOOR),
            "P4_cross_column_coherent": paired >= 0.5 * glob,
        }
        if not all(pre.values()):
            raise SystemExit(f"{col}: precondiciones {pre} — revisión manual requerida")

        mass = float((v[amb] == 1.0).mean()) if int(amb.sum()) else None
        rules[f"R2R3_{col}"] = {
            "inverted_div_1000": int(stripped.sum()),
            "kept_intact": int((v.notna() & ~stripped & ~amb).sum()),
            "ambiguous_resolved_intact": int(amb.sum()),
            "ambiguous_distribution": {str(k): int(x) for k, x in
                                       v[amb].value_counts().sort_index().items()},
            "ambiguous_mass_at_1.0": mass,
            "support_floor_nonzero": floor,
            "paired_trend_mean_ambiguous": paired,
            "paired_trend_mean_global": glob,
            "preconditions": pre,
            "quarantined_cells": int(v.isna().sum()),
        }
        repaired[col] = rep

    # ── R4: cuarentena MNAR, sin imputar ───────────────────────────────
    bad = pd.Series(False, index=raw.index)
    for col in TRENDS:
        bad |= repaired[col].isna()
    comp = raw.loc[bad, "Attrition_Flag"].value_counts().to_dict()
    p_full = float((raw["Attrition_Flag"] == "Attrited Customer").mean())
    n_q = int(bad.sum())
    exp = n_q * p_full
    sd = float(np.sqrt(n_q * p_full * (1 - p_full)))
    removed = int(comp.get("Attrited Customer", 0))

    # ── R5: ensamblado ─────────────────────────────────────────────────
    cur = raw.copy()
    cur["Avg_Utilization_Ratio"] = rebuilt
    for col in TRENDS:
        cur[col] = repaired[col]
    quarantine = cur.loc[bad].copy()
    quarantine["quarantine_reason"] = "R4: celda fecha irrecuperable en trend (MNAR)"
    cur = cur.loc[~bad].reset_index(drop=True)
    for col in TRENDS + ["Avg_Utilization_Ratio"]:
        cur[col] = cur[col].astype("float64")

    report = validate(cur, stage="curated")   # fail-closed: no se escribe si viola

    f_cur = CURATED_DIR / "churn_curated.csv"
    f_qua = CURATED_DIR / "quarantine.csv"
    cur.to_csv(f_cur, index=False)
    quarantine.to_csv(f_qua, index=False)

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "curation_rules_version": RULES_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "CURATED — no es BankChurners canónico",
        "source": {"path": str(RAW_ORIGIN_PATH), "sha256": sha(RAW_ORIGIN_PATH),
                   "n_rows": len(raw), "n_cols": raw.shape[1],
                   "git_first_commit": "2779afc", "never_modified_in_repo": True},
        "script_sha256": sha(Path(__file__)),
        "contract_sha256": sha(Path(dc.__file__)), 
        "output": {"path": str(f_cur), "sha256": sha(f_cur),
                   "n_rows": len(cur), "n_cols": cur.shape[1]},
        "quarantine": {"path": str(f_qua), "sha256": sha(f_qua), "n_rows": n_q,
                       "composition": comp, "missingness": "MNAR",
                       "attrited_expected_under_MCAR": exp, "attrited_sd": sd,
                       "attrited_observed": removed,
                       "z": (removed - exp) / sd if sd else None,
                       "prevalence_before": p_full,
                       "prevalence_after": float((cur["Attrition_Flag"]
                                                  == "Attrited Customer").mean()),
                       "imputation_applied": False},
        "rules": rules,
        "validation": report,
        "unresolved_anomalies": [{
            "columns": ["Naive_Bayes_..._1", "Naive_Bayes_..._2"],
            "observation": "NB_2 escala ×1e5 (no encaja en regla de 3 decimales); "
                           "NB_1 no complementa a NB_2. Segundo paso de manipulación.",
            "action": "sin reparar; prohibidas para entrenamiento (FORBIDDEN_FOR_TRAINING)",
        }],
        "reproducibility": "requiere uv.lock pineado (pandas/numpy) — ver ADR-0004",
    }
    (CURATED_DIR / "curation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"curated  {len(cur)} filas  sha256 {manifest['output']['sha256'][:16]}")
    print(f"quarantine {n_q} filas  z={manifest['quarantine']['z']:+.2f}")
    print(f"INV-1 {report['INV1_open_to_buy']:.4%}  INV-2 {report['INV2_utilization']:.4%}")


if __name__ == "__main__":
    main()
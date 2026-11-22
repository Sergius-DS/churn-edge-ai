# scripts/check_artifact_hashes.py
#!/usr/bin/env python
"""
Gate de integridad READ-ONLY. No regenera, no escribe, no muta evidencia.

Tres clases de verificación:
  integridad   — el artefacto trackeado vs el hash que declaró su manifest
  acuerdo      — dos manifests independientes declaran el mismo hash
  acoplamiento — el código que generó el dato sigue siendo el de HEAD

LIMITACIÓN DECLARADA (no hueco silencioso):
  No verifica iot_features_metadata.output_sha256. El CSV IoT está ignorado
  por git y se escribió con to_csv sin lineterminator explícito: CRLF en
  Windows, LF en Linux. Su hash es dependiente de plataforma. Verificarlo
  exige pinear el terminador, lo que MOVERÍA el hash del curated (evidencia
  publicada). Ver ADR-0004.

Stdlib pura: no puede fallar por dependencias.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(rel: str) -> str:
    p = ROOT / rel
    if not p.exists():
        raise FileNotFoundError(rel)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    cur = json.loads((ROOT / "data/curated/curation_manifest.json").read_text("utf-8"))
    iot = json.loads((ROOT / "data/synthetic/iot_features_metadata.json").read_text("utf-8"))
    exp = json.loads((ROOT / "edge/export_manifest_edge_iot.json").read_text("utf-8"))

    # Indexación directa: clave ausente -> KeyError. Nunca .get() con default.
    checks = [
        ("integridad",   "data/raw/Credit_card_churn.csv",        cur["source"]["sha256"],     "curation.source"),
        ("integridad",   "data/curated/churn_curated.csv",        cur["output"]["sha256"],     "curation.output"),
        ("integridad",   "data/curated/quarantine.csv",           cur["quarantine"]["sha256"], "curation.quarantine"),
        ("acuerdo",      "data/curated/churn_curated.csv",        iot["input_sha256"],         "iot_metadata.input"),
        ("acoplamiento", "scripts/curate_raw.py",                 cur["script_sha256"],        "curation.script"),
        ("acoplamiento", "src/data_contract.py",                  iot["contract_sha256"],      "iot_metadata.contract"),
        ("acoplamiento", "scripts/generate_synthetic_iot.py",     iot["script_sha256"],        "iot_metadata.script"),
        ("integridad",   "edge/xgb_model_edge_iot.json",          exp["model_sha256"],         "export.model"),
        ("integridad",   "edge/inference_contract_edge_iot.json", exp["contract_sha256"],      "export.contract"),
        ("integridad",   "edge/parity_sample_edge_iot.json",      exp["parity_sample_sha256"], "export.parity_sample"),
        ("integridad",   "edge/raw_parity_case_iot.json",         exp["raw_parity_sha256"],    "export.raw_parity"),
        ("integridad",   "edge/metrics_edge_iot.json",            exp["metrics_sha256"],       "export.metrics"),
    ]

    failures = []
    print("=" * 78)
    print("GATE DE INTEGRIDAD (read-only) — artefacto vs manifest")
    print("=" * 78)
    for kind, rel, expected, origin in checks:
        try:
            actual = sha256_file(rel)
        except FileNotFoundError:
            failures.append(f"{rel}: AUSENTE")
            print(f"  FAIL  {rel:<44} {kind:<12} AUSENTE")
            continue
        ok = actual.lower() == str(expected).lower()
        print(f"  {'ok  ' if ok else 'FAIL'}  {rel:<44} {kind:<12} {actual[:12]}")
        if not ok:
            failures.append(f"{rel}: actual={actual} vs {origin}={expected}")

    print("-" * 78)
    if failures:
        print(f"{len(failures)} discrepancias:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"{len(checks)} verificaciones OK · 0 discrepancias")
    return 0


if __name__ == "__main__":
    sys.exit(main())
# scripts/check_contract_posture.py
#!/usr/bin/env python
"""
Gate de POSTURA del contrato desplegado. Read-only, stdlib pura.

No verifica presencia: verifica que los controles estén ARMADOS.
`forbidden_fields: []` presente-pero-vacío es el estado defectuoso de los
cuatro #[serde(default)] de contract.rs. Este gate lo vuelve un rojo de CI.

Además valida la forma de iot_bounds: [lo, hi] con lo <= hi y finitos.
Un bound mal formado hace paniquear f64::clamp en wasm32 (sin unwinding:
la instancia muere). Detectarlo acá es más barato que descubrirlo allá.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "edge" / "inference_contract_edge_iot.json"


def main() -> int:
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    bad: list[str] = []

    posture = {
        "iot_features":     lambda v: isinstance(v, list) and len(v) == 6,
        "iot_bounds":       lambda v: isinstance(v, dict) and len(v) == 6,
        "forbidden_fields": lambda v: isinstance(v, list) and len(v) > 0,
        "consent_required": lambda v: v is True,
    }
    for key, ok in posture.items():
        if key not in c:
            bad.append(f"{key}: AUSENTE (serde default lo dejaría apagado)")
        elif not ok(c[key]):
            bad.append(f"{key}: presente pero APAGADO o mal formado")

    for feat, b in c.get("iot_bounds", {}).items():
        if not (isinstance(b, list) and len(b) == 2):
            bad.append(f"iot_bounds[{feat}]: no es [lo, hi] -> index panic en Rust")
        elif not all(isinstance(x, (int, float)) and math.isfinite(x) for x in b):
            bad.append(f"iot_bounds[{feat}]: bound no finito -> clamp panic")
        elif b[0] > b[1]:
            bad.append(f"iot_bounds[{feat}]: min > max {b} -> clamp panic")

    print("=" * 78)
    print("GATE DE POSTURA — controles presentes Y armados")
    print("=" * 78)
    if bad:
        for b in bad:
            print(f"  FAIL  {b}")
        print(f"\n{len(bad)} controles ausentes, apagados o mal formados.")
        return 1
    print("  ok    iot_features=6 · iot_bounds=6 bien formados · "
          "forbidden_fields no vacío · consent_required=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
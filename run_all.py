#!/usr/bin/env python
"""
run_all.py
──────────
Equivalente a `make all` para Windows.

Ejecuta:
1. scripts/generate_synthetic_iot.py
2. scripts/train_model.py
3. scripts/export_edge.py
4. scripts/validate_parity.py
5. scripts/train_edge_model.py
6. scripts/export_edge_lite.py
7. scripts/train_edge_iot_model.py
8. scripts/export_edge_iot.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    # Fase IoT sintética
    PROJECT_ROOT / "scripts" / "generate_synthetic_iot.py",

    # Baseline
    PROJECT_ROOT / "scripts" / "train_model.py",
    PROJECT_ROOT / "scripts" / "export_edge.py",
    PROJECT_ROOT / "scripts" / "validate_parity.py",

    # Edge actual
    PROJECT_ROOT / "scripts" / "train_edge_model.py",
    PROJECT_ROOT / "scripts" / "export_edge_lite.py",

    # Edge IoT
    PROJECT_ROOT / "scripts" / "train_edge_iot_model.py",
    PROJECT_ROOT / "scripts" / "export_edge_iot.py",
]


def run_script(script_path: Path) -> None:
    print("\n" + "=" * 70)
    print(f"▶ Ejecutando: {script_path.name}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print(f"\n❌ El script falló: {script_path}")
        sys.exit(result.returncode)


def main() -> None:
    print("=" * 70)
    print("🚀 CHURN EDGE AI — Pipeline completo en Windows")
    print("=" * 70)

    for script in SCRIPTS:
        if not script.exists():
            raise FileNotFoundError(f"No se encontró el script: {script}")
        run_script(script)

    print("\n" + "=" * 70)
    print("✅ Pipeline completo ejecutado correctamente")
    print("=" * 70)


if __name__ == "__main__":
    main()
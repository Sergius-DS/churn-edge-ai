#!/usr/bin/env python
"""
run_all.py
──────────
Cadena de producción del linaje Edge IoT. Cuatro pasos, en orden:

  1. scripts/generate_synthetic_iot.py
  2. Gate de distribución z  (bloqueante: aborta antes de entrenar)
  3. scripts/train_edge_iot_model.py
  4. scripts/export_edge_iot.py

FUERA de la cadena por defecto, a propósito:
  - curate_raw.py / calibrate_iot_reference.py: reescriben evidencia
    trackeada. Un pipeline que regenera su propio grupo de control no
    tiene grupo de control. Se invocan explícitamente y se verifican por hash.
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS = PROJECT_ROOT / "scripts"


def _run(cmd: list[str], label: str) -> None:
    print("\n" + "=" * 70)
    print(f"▶ {label}")
    print("=" * 70)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def script(name: str) -> None:
    path = SCRIPTS / name
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el script: {path}")
    _run([sys.executable, str(path)], name)


def gate_z() -> None:
    _run([sys.executable, "-m", "pytest",
          "tests/test_iot_z_distribution.py", "-s"],
         "Gate de distribución z (IoT)")


def main() -> None:
    print("=" * 70)
    print("🚀 CHURN EDGE AI — cadena Edge IoT")
    print("=" * 70)
    script("generate_synthetic_iot.py")
    gate_z()
    script("train_edge_iot_model.py")
    script("export_edge_iot.py")
    print("\n" + "=" * 70)
    print("✅ Cadena completa ejecutada correctamente")
    print("=" * 70)


if __name__ == "__main__":
    main()
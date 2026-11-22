# ADR-0004 — Qué se versiona y qué puede probar el repo

## Decisión
Se versionan: `data/raw/` (origen inmutable), `data/curated/` (SSOT + manifests +
quarantine), `data/synthetic/iot_features_metadata.json`, `edge/*.json` del linaje
vivo, `uv.lock`, `Cargo.lock`, `edge/churn-wasm/pkg/` (Pages lo sirve).
Se ignoran: `data/processed/*`, `model_artifacts/`, `edge/*.npy`.

## Criterio
Un artefacto trackeado que depende de inputs ignorados es **evidencia huérfana**.
El `.gitignore` no es higiene: decide qué puede probar el repositorio. Los
`.joblib` se reconstruyen desde inputs trackeados; los JSON de `edge/` y los
manifests son el acta.

## Limitación declarada
Pinear paquetes es necesario pero **no suficiente**. `to_csv` sin
`lineterminator` explícito emite CRLF en Windows y LF en Linux. Medido:
`CRLF 9997 / LF 9997`, sin `.gitattributes`. El hash `6bced2a3bf04…` es
Windows-específico.

Consecuencia operativa: el gate de CI es **read-only sobre blobs trackeados**
(`check_artifact_hashes.py`), no regenerar-y-comparar. Pinear el terminador
movería el hash del curated, citado en el post público y en dos manifests:
es un commit aparte con re-curación, o no se hace.

## Verificación
`uv.lock` + `Cargo.lock` trackeados. `uv sync --frozen` en CI falla si el lock
diverge de `pyproject.toml`.
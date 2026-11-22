# ADR-0006 — Un solo linaje de inferencia

## Decisión
El repo sirve un único linaje: **Edge IoT**, 34 features, 200 árboles,
threshold 0.22.

## Por qué
`edge/demo.js:52,56` solo hace fetch de `xgb_model_edge_iot.json` e
`inference_contract_edge_iot.json`. Full 28f y Lite 12f no servían tráfico y
descendían del raw mutado `41f4e92a…`.

El eslabón que subió la prioridad: `contract.rs` declara
`// ── Campos IoT (opcionales, solo en edge_iot) ──` sobre cuatro
`#[serde(default)]` cuyos defaults son **apagados** (`forbidden_fields → []`,
`iot_bounds → {}`, `consent_required → false`). Existen para que Full y Lite
sigan cargando. **Mientras Fase 1 viva en main, no se pueden volver obligatorios
los cuatro controles del linaje vivo.** El retiro le quita la justificación.

## Qué se retira — 33 archivos
7 scripts (`train_model`, `export_edge`, `validate_parity`, `train_edge_model`,
`export_edge_lite`, `validate_parity_raw`, `src/export_edge`) · 7 JSON de `edge/`
(Full 4 + Lite 3) · 5 `.npy` · `tests/parity_raw.rs` · `Makefile` · 12 binarios
trackeados de `model_artifacts/` raíz.

Los 12 binarios son el hallazgo del inventario: **el repo publicaba los binarios
del linaje muerto y ocultaba los del vivo**. El ignore no des-trackea lo que
entró antes del commit que agregó la regla.

## Dónde están
Tag `baseline-fase3` (`da8bebf`). Verificado:
`git diff --name-only baseline-fase3 HEAD -- edge/ model_artifacts/` devuelve
solo archivos IoT + wasm. Cero artefactos de Fase 1 cambiaron desde el tag.

## Makefile
Producía un sistema sin IoT (5 scripts, ninguno del linaje vivo) mientras
`run_all.py` se declaraba su equivalente. Dos orquestadores, dos productos, uno
afirmando ser el otro. Se elimina el lado falso.

## Constantes sin dueño eliminadas
`BEST_THRESHOLD` (0.38 — el vivo es 0.22, desde `model_config.joblib`),
`XGB_PARAMS`, `CV_NUM_BOOST_ROUND`, `CV_EARLY_STOPPING_ROUNDS`,
`EDGE_TOP_FEATURES`, `EDGE_IOT_TOP_FEATURES` (ya era muerta: el modelo usa 34,
no 18), `EDGE_ARTIFACTS_DIR`.
Conservadas: `ARTIFACTS_DIR` (padre de `EDGE_IOT_ARTIFACTS_DIR`), `CV_NFOLD`
(consumidor vivo en `train_with_cv`).

## Gate de salida
Los 5 hashes del linaje vivo no se mueven. `config.py` y `train.py` —los dos
podados— no están hasheados en ningún manifest; los tres que sí lo están
(`data_contract.py`, `curate_raw.py`, `generate_synthetic_iot.py`) no se tocaron.
Cero cambio semántico, probado por hashes.
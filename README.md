<h1 align="center">🪶 Churn Edge AI</h1>
<p align="center">
  <b>Predicción de churn 100% en el navegador con Rust + WebAssembly</b><br><br>
  <a href="https://sergius-ds.github.io/churn-edge-ai/" target="_blank">
    <img src="https://img.shields.io/badge/🚀_PROBAR_DEMO_EN_VIVO-2EA44F?style=for-the-badge&logo=githubpages&logoColor=white" alt="Probar Demo en Vivo">
  </a>
</p>

<p align="center">
  <img src="https://github.com/Sergius-DS/churn-edge-ai/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/badge/Dev_Container-Ready-blue?logo=visualstudiocode&logoColor=white" alt="Dev Container">
  <img src="https://img.shields.io/badge/Rust-stable-orange" alt="Rust">
  <img src="https://img.shields.io/badge/WASM-edge-blue" alt="WASM">
  <img src="https://img.shields.io/badge/XGBoost-JSON-green" alt="XGBoost">
  <img src="https://img.shields.io/badge/edge_model-224_KB-orange" alt="Edge Model">
  <img src="https://img.shields.io/badge/inference-100%25_client--side-blue" alt="Inference">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>



> **Churn Edge AI** entrena un modelo XGBoost en Python, lo exporta como JSON optimizado para edge y ejecuta la inferencia completa en **Rust/WebAssembly** dentro del navegador.  
> El cliente no envía datos sensibles a un servidor: la predicción ocurre localmente, con paridad numérica certificada respecto a Python.

---

## 🎯 Resumen ejecutivo

Este proyecto demuestra cómo llevar un modelo clásico de Machine Learning desde un pipeline Python de entrenamiento hacia un motor de inferencia liviano, portable y privado, ejecutado completamente en el cliente.

| Métrica | Resultado |
| --- | --- |
| Modelo Edge IoT — Test AUC | 0.9951 |
| Features raw | 25 (19 bancarias + 6 IoT sintéticas) |
| Features post-preprocessing | 34 |
| Árboles | 200 (max_depth 3) |
| Threshold de negocio | 0.22 (F2 sobre split interno pre-SMOTE) |
| Peso del modelo JSON | 223.760 bytes (~224 KB) |
| Peso gzip | 47.175 bytes (~47 KB, ratio 4.7×) |
| Paridad RAW Python ↔ Rust | 1.81e-11 |
| Réplica del contrato ≡ pipeline sklearn | max\|Δ\| = 0.000e+00 |
| SSOT | `data/curated/churn_curated.csv` · 9.996 filas |
| Inferencia | 100% client-side |
| Datos enviados al servidor | 0 bytes |

---

## 🧩 Problema

En sistemas tradicionales de predicción de churn:

- El modelo vive en un servidor.
- Cada predicción requiere enviar datos del cliente.
- Aumenta la latencia.
- Aumenta el costo de inferencia.
- Se incrementa la superficie de riesgo en privacidad.
- El despliegue depende de Python, containers, APIs o servicios cloud.

Para casos de uso donde la privacidad, el costo marginal y la velocidad importan, enviar datos crudos a un backend no siempre es la mejor arquitectura.

---

## ✅ Solución

**Churn Edge AI** implementa un pipeline completo de ML orientado a producción:

1. Entrenamiento y evaluación en Python con XGBoost.
2. Eliminación de fugas de información y preprocessing controlado.
3. Curación forense con contrato de ingestión fail-closed: dtype, unidades por
   columna, identidad algebraica. Cuarentena MNAR declarada, sin imputar.
4. Exportación del modelo a JSON.
5. Generación de un contrato de inferencia.
6. Motor de inferencia escrito en Rust.
7. Compilación a WebAssembly.
8. Validación de paridad matemática entre Python y Rust/WASM.
9. Demo en navegador sin backend.

---

## 🏗️ Arquitectura

```text
┌─────────────────────────── PYTHON — CURATE + TRAIN ────────────────────────┐
│                                                                            │
│  data/raw/Credit_card_churn.csv        (inmutable · grupo de control)      │
│        │                                                                   │
│        ▼   curate_raw.py · R1–R5 · fail-closed                             │
│  data/curated/churn_curated.csv        9.996 filas · INV-1/INV-2 = 100%    │
│  data/curated/quarantine.csv           131 filas MNAR, sin imputar         │
│        │                                                                   │
│        ▼   generate_synthetic_iot.py · z_ref con gate de soporte           │
│  + 6 features IoT sintéticas           → 29 columnas                       │
│        │                                                                   │
│        ▼   GATE z (bloqueante) · tests/test_iot_z_distribution.py          │
│        │                                                                   │
│        ▼   data_contract.validate → drop leakage/IDs → split               │
│        │                                                                   │
│        ▼   Imputación → Ordinal → One-Hot → SMOTE → Scaling                │
│        │                              25 raw → 34 features                 │
│        ▼   XGBoost edge (depth 3) + CV · optimize_threshold (F2)           │
│        │                                                                   │
│        ├──► model_artifacts/edge_iot/   (ignorado: derivados)              │
│        │       *.joblib · metrics.json · sample_provenance.json            │
│        │                                                                   │
│        └──► export_edge_iot.py · triple checkpoint                         │
│                xgb_model_edge_iot.json                                     │
│                inference_contract_edge_iot.json   (v2, byte-determinista)  │
│                metrics_edge_iot.json              (justifica el umbral)    │
│                parity_sample_edge_iot.json · raw_parity_case_iot.json      │
│                export_manifest_edge_iot.json      (7 hashes + procedencia) │
│                                                                            │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     ▼
┌────────────────────────── RUST / WASM — INFER ─────────────────────────────┐
│  contract.rs        deserializa el contrato v2                             │
│  preprocessing.rs   firewall → bounds IoT → impute → ordinal → onehot      │
│                     → scale                                                │
│  model.rs           parser de árboles XGBoost (arrays paralelos)           │
│  inference.rs       traversal en f32 + sigmoide                            │
│  lib.rs             init_engine (OnceLock) · predict_churn                 │
└────────────────────────────────────┬───────────────────────────────────────┘
                                     ▼
┌───────────────────────────────── BROWSER ──────────────────────────────────┐
│  index.html + demo.js                                                      │
│      carga churn_wasm.js                                                   │
│      carga xgb_model_edge_iot.json                                         │
│      carga inference_contract_edge_iot.json                                │
│      predicción local · 0 bytes al servidor                                │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔒 Privacidad por diseño

La inferencia se ejecuta completamente en el navegador.

| Propiedad | Estado |
|---|---|
| Datos del cliente enviados al servidor | 0 bytes |
| Backend requerido para predecir | No |
| Funciona offline después de cargar assets | Sí |
| Costo de inferencia por request en servidor | $0.00 |
| Privacidad | Local-first / Zero-Trust |
| Superficie de tratamiento de datos personales | Reducida |

> **Nota:** este enfoque favorece arquitecturas privacy-by-design y puede facilitar estrategias de minimización de datos. No constituye asesoría legal ni garantiza cumplimiento GDPR por sí mismo.

---

## 📊 Resultados del modelo

### Comparativa Full vs Edge (linaje retirado — ADR-0006, tag `baseline-fase3`)

> ⚠️ **Histórico.** El baseline Full 28f y el Edge Lite 12f se retiraron porque
> descendían del archivo mutado. El modelo vigente es **Edge IoT** (34 features,
> threshold 0.22, AUC 0.9951). Ver sección "Métricas — Modelo Edge IoT".

| Métrica | Full | Edge | Δ |
|---|---:|---:|---:|
| Features | 28 | 12 | -16 |
| Árboles | 500 | 200 | -300 |
| `max_depth` | 4 | 3 | -1 |
| CV AUC | 0.9986 | 0.9975 | -0.0011 |
| Test AUC | 0.9930 | 0.9894 | -0.0036 |
| Precision clase churn | 0.91 | 0.88 | -0.03 |
| Recall clase churn | 0.91 | 0.90 | -0.01 |
| F1 clase churn | 0.91 | 0.89 | -0.02 |
| Threshold | 0.38 | 0.38 | igual |
| Peso JSON | 818,904 bytes | 228,629 bytes | -72.1% |


---

## 🧠 Dataset, curación forense y SSOT

El archivo de origen no era el dataset canónico. Tres columnas de tipo ratio
venían con un punto decimal leído como separador de miles, y 131 celdas
contenían fechas donde debía ir un ratio.

| Artefacto | Filas | sha256 (prefijo) | Rol |
| --- | --- | --- | --- |
| `data/raw/Credit_card_churn.csv` | 10.127 | `41f4e92afcf6…` | Origen inmutable. Grupo de control |
| `data/curated/churn_curated.csv` | 9.996 | `6bced2a3bf04…` | SSOT operativa |
| `data/curated/quarantine.csv` | 131 | `a49c8ecee4cd…` | Celdas irrecuperables, sin imputar |

**El contrato de ingestión rechaza el origen con 4 violaciones:** dos por tipo,
una por unidades, la cuarta por identidad algebraica.

```text
Total_Amt_Chng_Q4_Q1: no numérico (Unable to parse string "2024-03-01 00:00:00" at position 158)
Total_Ct_Chng_Q4_Q1: no numérico (Unable to parse string "2024-05-02 00:00:00" at position 4)
Avg_Utilization_Ratio: 6880 fuera de [0.0,1.0] (min=0 max=999)
INV-2 violada: 32.0628%
```

La cuarta no depende de haber adivinado las unidades:
`utilización = saldo revolvente / límite`. Habría funcionado con un factor de
siete en lugar de mil.

Las 131 celdas irrecuperables **no se imputaron**: la ausencia resultó
correlacionada con el valor (8 clientes attrited removidos contra 21 esperados
bajo ausencia aleatoria; z = −3.10, aproximación normal a la binomial,
P(X ≤ 8) exacta = 0.0005). Imputar la media habría inyectado sesgo alineado
con el target.

Reglas R1–R5, precondiciones de dominio y cadena de custodia completas en
`data/curated/curation_manifest.json`. Evidencia del audit en `docs/evidence/`.


### Eliminación de leakage

Se eliminaron columnas incompatibles con un entorno productivo real:

```text
CLIENTNUM
Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1
Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2
```

Las dos columnas de trend llegaban como `object` porque contenían fechas. El
pipeline original las coercionaba con `errors="coerce"` y las imputaba con la
media: tres lavados silenciosos que convertían una violación de contrato en un
valor plausible. Esa coerción fue **eliminada**. El contrato garantiza dtype;
si el dato no parsea, se rechaza.

---

## 🚫 Anti-data leakage pipeline

El orden del preprocessing está diseñado para evitar fuga de información entre entrenamiento y validación.

```text
Split train/test
    ↓
Imputación numérica ajustada solo con train
    ↓
Imputación categórica ajustada solo con train
    ↓
Transformación ordinal ajustada solo con train
    ↓
One-Hot encoding ajustado solo con train
    ↓
SMOTE aplicado solo sobre train
    ↓
Escalado ajustado solo con train
    ↓
Entrenamiento
    ↓
Evaluación final sobre test intacto
```

### Principios

- El conjunto de test nunca se usa para ajustar transformaciones.
- SMOTE solo se aplica en train.
- Imputers, encoders y scaler se aprenden únicamente con train.
- El threshold se define sobre validación, no sobre test.
- Las métricas finales se calculan sobre test sin contaminación.

⚠️ **Caveat declarado:** SMOTE se aplica al train completo antes del CV. Los
puntos sintéticos se reparten entre folds, así que el AUC de validación es
optimista por construcción. El número publicable es `test_auc`. SMOTE
intra-fold va en rama aparte: cambia todos los números, incluido `base_score`.

---

## 🪶 Modelo Edge IoT

El modelo Edge IoT fue entrenado para inferencia liviana en WASM, combinando
features bancarias tradicionales con telemetría IoT sintética.

### Estrategia

- **Sin selección de features:** usa las 19 features bancarias del curated + 6 features IoT.
- Menor profundidad (`max_depth: 3` vs `4` del Full histórico).
- Menor número de árboles (`200` vs `500`).
- Regularización más fuerte (`gamma: 0.1`, `reg_lambda: 1.5`).
- Early stopping con cross-validation (aunque estructuralmente inerte post-SMOTE).
- Exportación a JSON con contrato de inferencia v2.
- Validación de paridad en 5 capas (Python ↔ sklearn ↔ Rust).

### Features (25 raw → 34 post-preprocessing)

**19 features bancarias** (del curated):
```text
Customer_Age, Gender, Dependent_count, Education_Level, Marital_Status,
Income_Category, Card_Category, Months_on_book, Total_Relationship_Count,
Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit,
Total_Revolving_Bal, Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1,
Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio
```

**6 features IoT sintéticas** (generadas con gate de soporte):
```text
app_open_count_30d, contactless_usage_count_30d,
payment_flow_abandon_count_30d, nfc_failed_attempts_30d,
app_crash_count_30d, pos_latency_p95_ms
```

**Post-preprocessing (34 features):**
- **20 numéricas directas:** 14 bancarias + 6 IoT
- **4 derivadas ordinales:** `Income_Category_ordinal`, `Education_Level_ordinal`
  + sus flags `is_unknown` (columnas numéricas, no one-hot)
- **10 one-hot:** Gender × 2, Marital_Status × 4, Card_Category × 4

Las 5 categóricas bancarias se reparten: 2 → ordinal + flag, 3 → one-hot.
20 + 4 + 10 = 34. Verificado contra el export (`onehot=10`) y el contrato.

### Parámetros Edge IoT

Config usada para el modelo Edge IoT (`src/config.py::XGB_EDGE_PARAMS`):

```python
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
    "seed": 44,
    "verbosity": 1,
}

EDGE_CV_NUM_BOOST_ROUND = 200
EDGE_EARLY_STOPPING_ROUNDS = 10
```

> ⚠️ **Caveat:** `best_round = 200` coincide con `EDGE_CV_NUM_BOOST_ROUND`. Es el
> techo, no un óptimo medido. Ver sección "Threshold de negocio".


---

## 📈 Métricas — Modelo Edge IoT

Test AUC: **0.995074** · Threshold: **0.22** · n_test: 2.000

```text
Matriz de confusión:
[[1568  108]
 [   5  319]]

TP = 319 · FN = 5 · FP = 108 · TN = 1568
```

| Clase | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Existing (0) | 1.00 | 0.94 | 0.97 |
| Attrited (1) | 0.75 | 0.98 | 0.85 |

Accuracy: 0.94 · Split: 7.996 train / 2.000 test · SMOTE: 7.996 → 13.402

Fuente de verdad: `edge/metrics_edge_iot.json`. Esta tabla es un espejo legible; el artefacto es el acta.

---

## 🎚️ Threshold de negocio

`threshold = 0.22`

Optimizado por F2 sobre un split interno del train **antes de SMOTE**, con
scaler propio ajustado en ese split. No se usó `0.5` porque el costo es
asimétrico: un falso negativo es un cliente que se va sin detectar.

El umbral vive en `model_artifacts/edge_iot/model_config.joblib` y se publica
en el contrato. **Ningún test ni frontend lo hardcodea**: lo leen del artefacto.

Trade-off declarado: FN 5 contra FP 108. Es el objetivo F2 operando como se
diseñó, no un defecto.

⚠️ `best_round = 200` coincide con `EDGE_CV_NUM_BOOST_ROUND`. Es el techo, no
un óptimo medido: el CV seguía subiendo en la iteración 199 y el early-stopping
es estructuralmente inerte sobre una curva monótona post-SMOTE. Registrado en
`metrics_edge_iot.json`.

---

## 🧪 Paridad certificada

| Capa | Qué compara | Dónde | Max diff |
| --- | --- | --- | --- |
| 1 | Réplica del contrato ≡ pipeline sklearn | `export_edge_iot.py` · Checkpoint 3 | **0.000e+00** |
| 2 | RAW → preprocessing → árboles → proba, Python ↔ Rust | `tests/parity_raw_iot.rs` | **1.81e-11** |
| 3 | El contrato desplegado deserializa en Rust | `tests/contract_live.rs` | pass/fail |
| 4 | Integridad de artefactos vs manifests | `check_artifact_hashes.py` | 12 checks |
| 5 | Controles del contrato presentes **y armados** | `check_contract_posture.py` | 4 controles |

La capa 1 es la que cierra el eslabón intermedio: sin ella, la réplica podría
ser auto-consistente con Rust y distinta de lo que el modelo vio al entrenar.

**Golden case:** fila real del curated + IoT (`CLIENTNUM 720014358`), con
doble checkpoint — `expected_scaled` (34 f32) y `expected_proba` — y tolerancia
declarada en el propio archivo. Cumple INV-1 e INV-2.

Pendiente: test Rust sobre las 100 muestras escaladas de
`parity_sample_edge_iot.json` (hueco 3).

---

## 🛡️ Integridad Metodológica y Transparencia (Anti-Gaming)

En Machine Learning es común inflar métricas accidentalmente mediante *data leakage*, *p-hacking* o manipulando el entorno de ejecución. Este proyecto aplica reglas estrictas para garantizar que los resultados sean honestos, reproducibles y aptos para producción:

| Práctica | Implementación |
|---|---|
| **Cero Data Leakage** | El split se realiza antes de imputación, escalado y SMOTE. El test no ajusta transformaciones. **Caveat:** SMOTE precede al CV, así que el AUC de validación es optimista; el número publicable es `test_auc` |
| **Trade-offs Honestos** | Se declara FN 5 contra FP 108 en el umbral 0.22, y que `best_round` está en el techo del CV. No se oculta el costo del objetivo F2 |
| **Threshold con recibo** | El umbral (0.22) se definió por F2 sobre split interno pre-SMOTE y vive en `model_config.joblib`. Su justificación está trackeada en `metrics_edge_iot.json`. Ningún test lo hardcodea |
| **Sin `RUSTFLAGS` mágicos** | Perfiles estándar (`--release`). Sin `target-cpu=native` ni flags ocultos |
| **Paridad Exacta** | Aritmética `f32` de XGBoost replicada deliberadamente. RAW→proba: `1.81e-11`. Réplica del contrato ≡ pipeline sklearn: `0.000e+00` |
| **Curación declarada** | El curated **no es BankChurners canónico**: 131 filas menos, prevalencia 0.16066 → 0.16196. El raw mutado se conserva inmutable como grupo de control |

> 📌 **Nota sobre Benchmarks de Rendimiento:** La optimización de *precisión, peso y paridad matemática* está certificada. Las métricas de latencia actuales (`< 1ms`) son empíricas, medidas directamente en el navegador. El proyecto **aún no implementa una suite formal de benchmarks de throughput con `criterion.rs`** en Rust nativo para evitar sesgos de medición del entorno del navegador (JIT, GC, fetch). Añadir `criterion` para benchmarks aislados es el siguiente paso en el roadmap de MLOps.

---

## 🦀 Motor Rust / WASM (Espejo de Inferencia Python)

El motor **no utiliza librerías externas de Machine Learning**: implementa una
réplica del pipeline de preprocesamiento y de la lógica de inferencia de Python.
Paridad RAW→proba certificada en **1.81e-11** (`tests/parity_raw_iot.rs`).

El motor en Rust es responsable de:

1. **Cargar el modelo XGBoost en JSON:** parsing de árboles con arrays paralelos
   (`left_children`, `right_children`, `split_indices`, `split_conditions`,
   `base_weights`, `sum_hessian`). No hay objetos-nodo ni campo `nodeid`.
2. **Cargar el contrato de inferencia:** `inference_contract_edge_iot.json` (v2).
3. **Replicar el preprocesamiento RAW:**
   - Imputación numérica `strategy="mean"` y categórica `most_frequent`,
     replicando los `SimpleImputer` ajustados en el train de Python.
   - Encoding ordinal de income/education con flags `is_unknown`.
   - One-Hot con el orden exacto de `get_feature_names_out()`.
   - `StandardScaler` con media y escala del train.
4. **Ejecutar la inferencia con aritmética `f32`:** réplica deliberada de la
   aritmética interna de XGBoost (usar `f64` introducía divergencias en splits
   cercanos al umbral).
5. **Calcular la probabilidad final:** sigmoide sobre el margen acumulado más el
   `base_score` del modelo, comparada contra el umbral que **declara el contrato**.

### 🏗️ Decisiones de ingeniería — alcance declarado

| Decisión | Alcance real |
|---|---|
| **Claves zero-copy** | Las claves del contrato se prestan (`Cow`/`&str`). **`predict_churn` parsea un DOM JSON por llamada**: el p50 está dominado por esa asignación, no por el traversal |
| **Caché `OnceLock`** | Modelo y contrato se parsean una vez en `init_engine`. La inferencia no los re-parsea |
| **Traversal sin asignaciones** | Cierto para el recorrido del árbol. No para la frontera JSON |
| **Firewall de privacidad** | Rechaza campos prohibidos (`pan`, `cvv`, GPS…) antes de imputar |
| **Gate de consentimiento** | `lib.rs:66` rechaza inferir sin consent cuando el contrato lo exige |
| **Parsing con `serde`** | Los nodos se modelan como `enum` (`Leaf`/`Split`) para eliminar estados inválidos en compilación |

> ⚠️ **Gap declarado:** el crate no valida la versión del contrato ni verifica el
> `model_sha256` (`sha2` ausente de `Cargo.toml`). Cuatro campos del contrato son
> opcionales con default apagado. Ver "Estado actual".

### 💡 Caso de estudio: por qué `f32` y no `f64`

XGBoost evalúa splits y acumula márgenes usando precisión de 32 bits internamente.

Inicialmente, el motor Rust calculaba todo en `f64` (precisión doble), que es la opción por defecto en Rust para cálculos matemáticos. Sin embargo, esto rompía la paridad en casos límite donde un valor caía muy cerca del umbral de un split.

**El problema:** diferencias de redondeo entre `f32` y `f64` causaban que Rust tomara ramas diferentes en el árbol de decisión, acumulando errores que resultaban en divergencias mayores a `1e-06`.

**La solución:** replicar deliberadamente la aritmética interna de XGBoost:

```text
features f64 → features f32
comparaciones de split en f32
acumulación de margin en f32
margin final convertido a f64 para sigmoid
```

Este detalle fue clave para lograr la paridad. Medición histórica del linaje
Lite: `max diff ≈ 7.31e-09`. El linaje IoT vigente alcanza **1.81e-11**.

Lección aprendida: en sistemas de inferencia ML, la precisión numérica no es solo una cuestión de exactitud matemática, sino de replicación del comportamiento del framework original. Lo que es "más preciso" (f64) puede ser incorrecto si el modelo fue entrenado con f32.

### 📁 Módulos principales

```text
edge/churn-wasm/src/
    lib.rs             Entry point expuesto hacia JavaScript (wasm-bindgen)
    contract.rs        Deserialización y validación del contrato de inferencia
    preprocessing.rs   Réplica exacta del pipeline crudo → features escaladas
    model.rs           Parser tipado del árbol JSON de XGBoost
    inference.rs       Motor de recorrido de árboles y cálculo de márgenes (f32)
```

---

## 🌐 Demo en navegador

La demo permite:

- Cargar el modelo WASM localmente.
- Cargar el modelo edge JSON.
- Cargar el contrato edge.
- Ingresar features del cliente.
- Calcular probabilidad de churn sin backend.
- Verificar paridad automáticamente.

### Ejemplo de predicción

```text
Probabilidad: 0.15%
Threshold:    0.22
Decisión:     SIN RIESGO DE CHURN
```

---

## 📦 Entorno de Desarrollo (DevContainer)

Este proyecto está configurado con **Dev Containers** (estándar de Microsoft/GitHub) para garantizar una reproducibilidad del 100% y evitar conflictos de dependencias locales. No necesitas instalar Python, Rust, `uv` ni `wasm-pack` manualmente en tu sistema operativo.

### 🌩️ Opción A: GitHub Codespaces (Cero instalación)
1. Ve al repositorio en GitHub.
2. Haz clic en el botón verde **`<> Code`**.
3. Selecciona la pestaña **Codespaces** y haz clic en **"Create codespace on main"**.
4. En menos de 2 minutos, tendrás un VS Code completo en tu navegador con Python 3.10, Rust, `uv` y `wasm-pack` preconfigurados y listos para usar.

### 💻 Opción B: VS Code Local
1. Clona el repositorio y ábrelo en **Visual Studio Code**.
2. Asegúrate de tener instalada la extensión oficial **Dev Containers** de Microsoft.
3. VS Code detectará el archivo `.devcontainer/devcontainer.json` y te mostrará una notificación en la esquina inferior derecha.
4. Haz clic en **"Reopen in Container"**.
5. El entorno aislado se construirá automáticamente. Al terminar, la terminal integrada ya estará dentro del contenedor con todas las herramientas listas.

---

## 🚀 Quickstart

Requisitos: Python **3.10.11** (pineado por `uv.lock`), `uv`, Rust stable, `wasm-pack`.

```bash
# 1. Entorno fail-closed
uv sync --frozen

# 2. Cadena de producción (3 scripts + 1 gate bloqueante)
uv run python run_all.py
#    generate_synthetic_iot → gate z → train_edge_iot → export_edge_iot

# 3. Gates de verificación
uv run python scripts/check_artifact_hashes.py     # 12 verificaciones
uv run python scripts/check_contract_posture.py    # 4 controles armados
uv run pytest tests/ -v                            # contrato + distribución z

# 4. Tests Rust
cd edge/churn-wasm && cargo test -- --nocapture

# 5. Compilar WASM
wasm-pack build --target web --release

# 6. Demo local
cd edge && python -m http.server 8000
```

**Fuera de la cadena por defecto, a propósito:** `curate_raw.py` y
`calibrate_iot_reference.py` reescriben evidencia trackeada. Un pipeline que
regenera su propio grupo de control no tiene grupo de control. Se invocan
explícitamente y su resultado se verifica por hash.

---

## 📦 Estructura del proyecto

```text
churn-edge-ai/
├── .github/workflows/
│   ├── ci.yml                          # python-gates + rust-wasm
│   └── deploy-pages.yml                # sube ./edge completo
├── .gitignore
├── pyproject.toml
├── uv.lock                             # pin fail-closed
├── run_all.py                          # 3 scripts + gate z
├── README.md · README.en.md
│
├── data/
│   ├── raw/Credit_card_churn.csv       # origen inmutable, grupo de control
│   ├── curated/                        # SSOT + manifests + quarantine
│   └── synthetic/iot_features_metadata.json
│   # data/processed/ ignorado: derivado regenerable
│
├── docs/
│   ├── adr/0004 · 0005 · 0006
│   └── evidence/2026-11-audit-raw.{json,log}
│
├── model_artifacts/                    # ignorado por completo
│   └── edge_iot/                       # binarios + metrics + provenance
│
├── edge/
│   ├── xgb_model_edge_iot.json         # 34 features, 200 árboles
│   ├── inference_contract_edge_iot.json# v2, byte-determinista
│   ├── metrics_edge_iot.json           # justificación del threshold
│   ├── export_manifest_edge_iot.json   # 7 hashes + procedencia (run_fingerprint, created_at)
│   ├── parity_sample_edge_iot.json     # 100 × 34 f32
│   ├── raw_parity_case_iot.json        # golden real, doble checkpoint
│   ├── index.html · demo.js
│   └── churn-wasm/
│       ├── Cargo.toml · Cargo.lock
│       ├── pkg/                        # WASM trackeado para Pages
│       ├── src/{lib,contract,preprocessing,model,inference}.rs
│       └── tests/{contract_live,parity_raw_iot}.rs
│
├── mock_bank/
├── src/
│   ├── config.py                       # sin constantes de linajes retirados
│   ├── data_contract.py                # contrato de ingestión v1.1.0
│   ├── data_loading.py · preprocessing.py · train.py · synthetic_iot.py
│
├── scripts/
│   ├── curate_raw.py · calibrate_iot_reference.py
│   ├── generate_synthetic_iot.py
│   ├── train_edge_iot_model.py · export_edge_iot.py
│   ├── check_artifact_hashes.py · check_contract_posture.py
│
└── tests/
    ├── test_data_contract.py           # rechaza el raw, acepta el curated
    └── test_iot_z_distribution.py      # el síntoma tiene techo
```

---

## 📄 Contrato de inferencia

El contrato `inference_contract_edge_iot.json` (v2) incluye:

```text
feature_order · n_features
numeric_imputer · categorical_imputer
ordinal mappings + flags is_unknown
one-hot categories (orden de get_feature_names_out)
scaler mean/scale
best_threshold
xgboost base_score
iot_features · iot_bounds
forbidden_fields · consent_required
model_sha256
```

Esto evita problemas clásicos de despliegue: features desordenadas, categorías
nuevas mal manejadas, escalado inconsistente, imputación distinta entre train y
serving, threshold incorrecto, base score mal interpretado.

El contrato es **byte-determinista**: `created_at` vive en
`export_manifest_edge_iot.json`, no acá. Dos exports consecutivos producen el
mismo `sha256`.

> ⚠️ El `model_sha256` viaja en el contrato; el motor todavía no lo verifica.
> Gap declarado, con plan de cierre en la secuencia B → C → D.

---

## 🧰 Comandos principales

| Comando | Descripción |
|---|---|
| `uv run python run_all.py` | Cadena Edge IoT completa |
| `uv run python scripts/curate_raw.py` | Curación forense R1–R5 (explícito) |
| `uv run python scripts/calibrate_iot_reference.py` | Recalibra IOT_REFERENCE sobre train (explícito) |
| `uv run python scripts/check_artifact_hashes.py` | Gate de integridad read-only (12 checks) |
| `uv run python scripts/check_contract_posture.py` | Gate de postura (4 controles armados) |
| `uv run pytest tests/ -v -s` | Contrato de datos + distribución z |
| `cargo test` | Tests Rust: contrato vivo + paridad IoT + firewall |
| `wasm-pack build --target web --release` | Compila WASM |
| `python -m http.server 8000` | Sirve demo local |

---

## ✅ Tests

### Python

```bash
uv run pytest tests/ -v -s
```

| Test | Qué certifica |
| --- | --- |
| `test_data_contract.py::test_gate_rejects_mutated_origin` | El origen mutado falla con las 4 literales. **Es el grupo de control del audit** |
| `test_data_contract.py::test_gate_accepts_curated_ssot` | Control positivo: sin él, un `validate()` que rechaza todo pasaría el test anterior |
| `test_iot_z_distribution.py` | Imprime la tabla de z por feature y exige techo. El síntoma no puede regresar |

Más dos gates read-only, invocados por CI antes de instalar dependencias:

```bash
python scripts/check_artifact_hashes.py     # 12 verificaciones: integridad · acuerdo · acoplamiento
python scripts/check_contract_posture.py    # 4 controles presentes Y armados
```

### Rust

```bash
cd edge/churn-wasm && cargo test -- --nocapture
```

11 tests: 7 unitarios (`contract`, `inference`, `lib`) + `contract_live` (1) +
`parity_raw_iot` (3: paridad end-to-end, firewall de privacidad, caracterización
del clipping silencioso).

Salida esperada de la paridad IoT:

```text
🐍 Python (expected): 0.00150453
🦀 Rust   (got):      0.00150453
📉 Diff:              1.81e-11
🏁 Resultado:         ✅ PASS
```

Los tres asserts de consistencia (`threshold`, `n_features`, `n_trees`) **leen
de los artefactos desplegados**, no de constantes del test.

---

## 🔄 CI/CD

`on: [push, pull_request]` — cobertura en todas las ramas.

**Job `python-gates`:** integridad y postura con Python del sistema (stdlib pura,
fallan en 2 s), luego `uv sync --frozen`, gate de contrato de datos, regeneración
de telemetría y gate de distribución z con la tabla impresa en el log.

**Job `rust-wasm`:** `cargo test` (todos los binaries) + `wasm-pack build`.

```yaml
name: CI

on: [push, pull_request]

jobs:
  python-gates:
    name: Python gates (integridad · postura · contrato · IoT z)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Gate de integridad (read-only, 12 verificaciones)
        run: python3 scripts/check_artifact_hashes.py

      - name: Gate de postura del contrato (controles ARMADOS)
        run: python3 scripts/check_contract_posture.py

      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.10.11"

      - name: Sync dependencias (lockfile fail-closed)
        run: uv sync --frozen

      - name: Gate de contrato de datos
        run: uv run pytest tests/test_data_contract.py -v -s

      - name: Regenerar telemetría sintética
        run: uv run python scripts/generate_synthetic_iot.py

      - name: Gate de distribución z (tabla en el log)
        run: uv run pytest tests/test_iot_z_distribution.py -s

  rust-wasm:
    name: Rust tests + WASM build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Install wasm-pack
        run: cargo install wasm-pack --locked
      - name: Cargo tests
        run: cd edge/churn-wasm && cargo test -- --nocapture
      - name: Build WASM
        run: cd edge/churn-wasm && wasm-pack build --target web --release
```

El job legacy `python-parity` se retiró en ADR-0006 junto con los scripts de
Fase 1 que invocaba. Usaba Python 3.11 con `pip install` sin pines mientras el
resto del repo respetaba `uv.lock` (3.10.11): dos intérpretes, un lockfile.

---

## 🧾 Artefactos generados

### Locales, ignorados por git (derivados regenerables)

```text
model_artifacts/edge_iot/
    xgb_model.joblib · scaler.joblib · onehot_encoder.joblib
    num_imputer.joblib · cat_imputer.joblib · ordinal_imputer.joblib
    ordinal_mappings.joblib · model_config.joblib
    X_test_scaled_sample.csv · X_test_raw_sample.csv
    sample_provenance.json · metrics.json

data/processed/credit_card_churn_iot_synthetic.csv
edge/*.npy
```

### Trackeados — fuente de verdad para Rust y para el auditor

```text
edge/
    xgb_model_edge_iot.json           34 features · 200 árboles
    inference_contract_edge_iot.json  v2 · byte-determinista
    metrics_edge_iot.json             AUC, matriz, criterio F2, caveats
    export_manifest_edge_iot.json     7 hashes + procedencia (run_fingerprint, created_at)
    parity_sample_edge_iot.json       100 × 34 f32
    raw_parity_case_iot.json          golden real · doble checkpoint

data/curated/                         SSOT + manifests + quarantine
data/synthetic/iot_features_metadata.json
```

**Criterio:** un artefacto trackeado que depende de inputs ignorados es
evidencia huérfana. Los `.joblib` se reconstruyen; los JSON de `edge/` y los
manifests son el acta. Ver ADR-0004.

---

## 💼 Impacto de negocio

| Área | Impacto |
|---|---|
| Infraestructura | Reduce costo de inferencia en servidor |
| Privacidad | Los datos crudos no salen del dispositivo |
| Latencia | Predicción local inmediata |
| Retención | Detección temprana de riesgo de churn |
| Producto | Experiencia interactiva sin backend |
| Cumplimiento | Favorece minimización de datos |
| Costos | Menos dependencias de APIs o servicios cloud |

---

## 🔐 Seguridad y consideraciones

Este proyecto sigue principios de privacidad por diseño, pero hay consideraciones importantes:

- El modelo se distribuye al cliente.
- El modelo no debe considerarse secreto si se publica en el navegador.
- La inferencia local protege datos del usuario, no necesariamente el IP del modelo.
- No debe usarse como único mecanismo antifraude.
- Si se despliega en producción, se recomienda:
  - Firmar artefactos.
  - Usar Subresource Integrity para WASM/JS.
  - Versionar modelos.
  - Registrar hashes de contratos.
  - Monitorear drift.
  - Mantener auditoría de cambios.

---

## 📡 Producción y monitoreo

Para llevar este sistema a producción, se recomienda:

### Monitoreo de datos

- Drift de features numéricas.
- Drift de categorías.
- Cambio en distribución de predicciones.
- Cambio en tasa de churn real.

### Monitoreo de modelo

- AUC/F2 en ventanas móviles.
- Estabilidad del threshold.
- Diferencia entre predicción y resultado real.

### MLOps

- Versionado de modelos.
- Hashes de artefactos.
- Registro de contratos de inferencia.
- Tests de paridad en CI.
- Reproducción de entrenamiento con seed fija.
- Canary deployment de nuevos modelos edge.

---

## 🐞 Troubleshooting

### Paridad falla por muy poca diferencia

Verificar que Rust esté usando aritmética `f32` para:

- Comparaciones de split.
- Acumulación de margin.
- Evaluación de árboles.

### `UnicodeEncodeError` en Windows

Si un script escribe JSON con caracteres Unicode, usar siempre:

```python
encoding="utf-8"
```

Ejemplo:

```python
out_path.write_text(payload, encoding="utf-8")
```

O ejecutar con:

```bash
PYTHONUTF8=1
```

### La demo no carga WASM

Verificar:

```text
edge/churn-wasm/pkg/churn_wasm.js
edge/churn-wasm/pkg/churn_wasm_bg.wasm
```

Ambos deben existir después de:

```bash
wasm-pack build --target web --release
```

### El navegador cachea el WASM

Hacer hard reload:

```text
Ctrl + Shift + R
```

o limpiar caché.

---

## 🗺️ Evolución del Proyecto

Este repositorio documenta la construcción de un motor de inferencia determinista en el Edge. Lo que comenzó como un modelo de churn bancario tradicional (Hito 1), ha evolucionado hacia un SDK completo de IoT con contratos de privacidad explícitos para la industria financiera.

### 🏁 Hito 1 — Motor determinista (retirado, archivado)
El baseline Full 28f y el Edge Lite 12f se retiraron en ADR-0006. No servían
tráfico y descendían del archivo mutado. Recuperables desde el tag
`baseline-fase3`. Su paridad histórica (7.31e-09) queda como medición de archivo.

### 🌱 Hito 2 — Curación forense y SSOT (completado ✅)
- [x] Contrato de ingestión fail-closed: dtype, unidades, identidad algebraica
- [x] Curación R1–R5 con precondiciones de dominio re-verificadas por corrida
- [x] Cuarentena MNAR sin imputar (131 filas, z = −3.10)
- [x] Cadena de custodia: hashes de origen, script, contrato y salida
- [x] Generador IoT con gate de soporte sobre la referencia fija

### 🛡️ Hito 3 — Núcleo Rust (parcial 🚧)
- [x] Firewall de privacidad: rechazo de campos prohibidos
- [x] Gate de consentimiento (`lib.rs:66`)
- [x] `init_engine` con `OnceLock`
- [x] Paridad RAW→proba con doble checkpoint (1.81e-11)
- [x] El contrato desplegado deserializa en Rust (`contract_live.rs`)
- [ ] **Verificación de integridad SHA-256 en el motor** — `sha2` no está en
      `Cargo.toml`. El hash viaja en el contrato; el motor todavía no lo comprueba
- [ ] **Clipping con flag** — hoy clipea en silencio. Un input imposible
      (`app_open = 9999`) produce una predicción plausible sin marcador
- [ ] **Controles obligatorios en el esquema** — cuatro `#[serde(default)]`
      dejan `forbidden_fields`, `iot_bounds`, `iot_features` y
      `consent_required` apagados por omisión
- [ ] Test Rust sobre las muestras escaladas
- [ ] Compilación multiplataforma (Android `.so`, iOS `.a`)
- [ ] Benchmarking formal con `criterion.rs`

### 🔌 Hito 4 — Ecosistema B2B (parcial 🚧)
- [x] Mock Bank API con gestión de consentimiento
- [x] Modo offline-first (fallback embebido + localStorage)
- [ ] SDK Web (NPM + Web Workers)
- [ ] Envoltorios móviles

## 💼 Hito 5 — UX bancaria (parcial 🚧)

- [x] Interfaz dual: Vista Operador + Vista Cliente
- [ ] **Política de retención fuera del motor** — `offer_retention` está
      compilada en WASM (`lib.rs:84`). La decisión de negocio pertenece a la
      capa de aplicación, no al binario de inferencia
- [ ] Consentimiento explícito, visible y revocable — sin disfrazar de
      banner de beneficios
- [x] **Threshold dinámico desde el contrato** — el frontend lee `best_threshold`
      de `inference_contract_edge_iot.json`, ningún test ni componente lo
      hardcodea. Evidencia: sección "Threshold de negocio" y `edge/demo.js`

---

## 🎯 Estado actual

**Verificado y en producción:**
- SSOT curada con contrato fail-closed y cadena de custodia por hash
- Motor Rust/WASM con paridad RAW→proba de 1.81e-11
- 12 verificaciones de integridad + 4 de postura en CI
- Inferencia 100% client-side, 0 bytes al servidor
- Demo dual en GitHub Pages con fallback offline

**Gaps conocidos, declarados:**
1. El motor no verifica el SHA-256 del modelo (`sha2` ausente del crate)
2. Cuatro controles del contrato son opcionales con default permisivo
3. El clipping IoT no emite flag: input imposible → predicción plausible
4. Las 14 columnas bancarias no tienen bounds en inferencia
5. `best_round` está en el techo del CV; el early-stopping es inerte sobre
   una curva post-SMOTE monótona
6. El golden vive en p ≈ 0.15%: tolera un error de logit ~166× mayor que uno
   en la zona media
7. SMOTE se aplica antes del CV: el AUC de validación es optimista.
   **El número publicable es el de test**
8. La política de retención está en el binario, no en la aplicación

Los gaps 1–4 tienen plan de cierre en la secuencia B → C → D.

---

## 📚 Stack tecnológico

```text
Python
pandas
scikit-learn
imbalanced-learn
XGBoost
Rust
wasm-bindgen
wasm-pack
WebAssembly
JavaScript vanilla
GitHub Actions
Uv
Pytest
```

Reproducibilidad:

```text
SEED = 44
```

---

## 📌 Resumen final

**Churn Edge AI** convierte un modelo tradicional de churn bancario en un sistema de inferencia edge liviano, privado y de costo marginal cero.

Combina:

```text
Python para entrenamiento
XGBoost para modelado
JSON como formato portable
Rust para inferencia segura y rápida
WebAssembly para ejecución en navegador
Paridad certificada mediante tests
```

Es un caso de uso relevante para:

- Bancos.
- Fintech.
- Retail.
- Telecomunicaciones.
- Productos digitales sensibles a privacidad.
- Entornos offline o de conectividad limitada.
- Arquitecturas client-side ML.

---

## 📫 Contacto

Autor: `Sergio Francisco Rivera Bustamante`  
GitHub: `https://github.com/Sergius-DS`  
LinkedIn: `https://www.linkedin.com/in/sergio-francisco-rivera-bustamante/`  
Proyecto: **Churn Edge AI**

---

## 📄 Licencia

MIT

Copyright (c) 2026 Sergio Francisco Rivera Bustamante

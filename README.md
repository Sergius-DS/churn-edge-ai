# 🪶 Churn Edge AI — Predicción de churn 100% en el navegador con Rust + WebAssembly

![CI](https://github.com/Sergius-DS/churn-edge-ai/actions/workflows/ci.yml/badge.svg)
![Dev Container](https://img.shields.io/badge/Dev_Container-Ready-blue?logo=visualstudiocode&logoColor=white)
![Rust](https://img.shields.io/badge/Rust-stable-orange)
![WASM](https://img.shields.io/badge/WASM-edge-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-JSON-green)
![Model Size](https://img.shields.io/badge/edge_model-229_KB-orange)
![Parity](https://img.shields.io/badge/parity_python%E2%86%94wasm-max_diff_7.3e--9-brightgreen)
![Inference](https://img.shields.io/badge/inference-100%25_client--side-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> **Churn Edge AI** entrena un modelo XGBoost en Python, lo exporta como JSON optimizado para edge y ejecuta la inferencia completa en **Rust/WebAssembly** dentro del navegador.  
> El cliente no envía datos sensibles a un servidor: la predicción ocurre localmente, con paridad numérica certificada respecto a Python.

---

## 🎯 Resumen ejecutivo

Este proyecto demuestra cómo llevar un modelo clásico de Machine Learning desde un pipeline Python de entrenamiento hacia un motor de inferencia liviano, portable y privado, ejecutado completamente en el cliente.

| Métrica | Resultado |
|---|---:|
| Modelo full — Test AUC | **0.9930** |
| Modelo edge — Test AUC | **0.9894** |
| Caída de AUC entre full y edge | **0.0036** |
| Features full | 28 |
| Features edge | 12 |
| Árboles full | 500 |
| Árboles edge | 200 |
| Peso del modelo full JSON | ~819 KB |
| Peso del modelo edge JSON | ~229 KB |
| Reducción de peso | **72.1%** |
| Paridad Python ↔ JSON | `0.00e+00` |
| Paridad Rust/WASM ↔ Python en navegador | `2.02e-10` |
| Paridad end-to-end RAW ↔ Rust | `7.31e-09` |
| Inferencia | 100% client-side |
| Envío de datos al servidor | 0 bytes |

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
3. Optimización de un modelo edge:
   - Menos features.
   - Menos árboles.
   - Menor profundidad.
   - Regularización más fuerte.
   - Menor peso de artefacto.
4. Exportación del modelo a JSON.
5. Generación de un contrato de inferencia.
6. Motor de inferencia escrito en Rust.
7. Compilación a WebAssembly.
8. Validación de paridad matemática entre Python y Rust/WASM.
9. Demo en navegador sin backend.

---

## 🏗️ Arquitectura

```text
┌────────────────────────────── PYTHON — TRAIN ──────────────────────────────┐
│                                                                             │
│  Credit_card_churn.csv                                                      │
│        │                                                                    │
│        ▼                                                                    │
│  Eliminación de leakage / IDs                                               │
│        │                                                                    │
│        ▼                                                                    │
│  Split train/test                                                           │
│        │                                                                    │
│        ▼                                                                    │
│  Imputación → Ordinal → One-Hot → SMOTE → Scaling                           │
│        │                                                                    │
│        ▼                                                                    │
│  XGBoost full + CV + early stopping                                         │
│        │                                                                    │
│        ├──► model_artifacts/                                                │
│        │       xgb_model.joblib                                             │
│        │       scaler.joblib                                                │
│        │       encoders / imputers / config                                 │
│        │                                                                    │
│        └──► edge/                                                           │
│                xgb_model.json                                               │
│                inference_contract.json                                      │
│                parity_sample.json                                           │
│                                                                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────── EDGE OPTIMIZATION ─────────────────────────────┐
│                                                                             │
│  Selección de top features                                                  │
│  Menos árboles / menor profundidad                                          │
│  Regularización más fuerte                                                  │
│  Export edge:                                                               │
│      xgb_model_edge.json                                                    │
│      inference_contract_edge.json                                           │
│      parity_sample_edge.json                                                │
│      raw_parity_case.json                                                   │
│                                                                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────── RUST / WASM — INFER ───────────────────────────┐
│                                                                             │
│  churn-wasm/                                                                │
│      contract.rs          parsing del contrato de inferencia                │
│      preprocessing.rs     raw → impute → ordinal → one-hot → scale          │
│      model.rs             parsing del JSON de árboles XGBoost               │
│      inference.rs         recorrido de árboles con aritmética f32           │
│      lib.rs               APIs wasm_bindgen                                 │
│                                                                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────── BROWSER ──────────────────────────────────┐
│                                                                             │
│  index.html + demo.js                                                       │
│      carga churn_wasm.js                                                    │
│      carga xgb_model_edge.json                                              │
│      carga inference_contract_edge.json                                     │
│      ejecuta predicción local                                               │
│      valida paridad con Python                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
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

### Comparativa Full vs Edge

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

### Interpretación

El modelo edge pierde muy poca capacidad predictiva a cambio de:

- Reducir el peso del modelo en **72.1%**.
- Disminuir la cantidad de features de 28 a 12.
- Reducir la complejidad de inferencia.
- Hacer viable la ejecución en navegador.
- Mantener un threshold de negocio estable.

La caída de AUC fue:

```text
Δ AUC = 0.003620
```

Este valor está dentro del rango aceptable para un despliegue edge.

---

## 🧠 Dataset y preparación

El pipeline parte de un dataset de churn bancario con:

```text
Filas iniciales:     10,127
Columnas iniciales:  23
Target:
  0 → 8,500
  1 → 1,627
```

### Eliminación de leakage

Se eliminaron columnas incompatibles con un entorno productivo real:

```text
CLIENTNUM
Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1
Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2
```

Además, se convirtieron a numéricas:

```text
Total_Amt_Chng_Q4_Q1
Total_Ct_Chng_Q4_Q1
```

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

---

## 🪶 Optimización Edge

El modelo edge fue entrenado específicamente para inferencia liviana en WASM.

### Estrategia

- Selección de las features más importantes del modelo full.
- Menor profundidad.
- Menor número de árboles.
- Regularización más fuerte.
- Early stopping con cross-validation.
- Exportación a JSON.
- Validación de paridad.

### Features edge

Las 12 features seleccionadas fueron:

```text
Total_Trans_Ct
Total_Revolving_Bal
Total_Ct_Chng_Q4_Q1
Months_Inactive_12_mon
Contacts_Count_12_mon
Total_Trans_Amt
Total_Relationship_Count
Avg_Utilization_Ratio
Total_Amt_Chng_Q4_Q1
Dependent_count
Gender_F
Marital_Status_Single
```

### Parámetros edge

Config usada para el modelo edge:

```python
edge_params = {
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
```

---

## 📈 Métricas de clasificación

### Modelo Full

```text
Test AUC:  0.993047
Threshold: 0.38

Matriz de confusión:
[[1671   30]
 [  28  297]]

TP = 297
FN = 28
FP = 30
TN = 1671
```

### Modelo Edge

```text
Test AUC:  0.989427
Threshold: 0.38

Matriz de confusión:
[[1662   39]
 [  31  294]]

TP = 294
FN = 31
FP = 39
TN = 1662
```

---

## 🎚️ Threshold de negocio

El threshold usado es:

```text
threshold = 0.38
```

No se usa `0.5` porque el costo de negocio no es simétrico.

En churn:

- Un falso negativo implica no detectar a un cliente que se va.
- Un falso positivo implica alertar a un cliente que finalmente no se va.

Por eso se optimizó una métrica orientada a recall, usando criterio tipo **F2**, donde penaliza más no detectar un churn real.

---

## 🧪 Paridad matemática certificada

Uno de los requisitos críticos del proyecto es que Rust/WASM produzca el mismo resultado que Python/XGBoost.

La paridad fue validada en múltiples capas.

| Capa | Qué compara | Herramienta | Max diff |
|---|---|---|---:|
| 1 | Python ↔ JSON exportado | `scripts/validate_parity.py` | `0.00e+00` |
| 2 | JSON ↔ Rust usando features ya escaladas | `verify_parity` en navegador | `2.02e-10` |
| 3 | Input RAW → probabilidad, Python ↔ Rust | golden file + `cargo test` | `7.31e-09` |
| 4 | Demo WASM vs Python | cross-check manual | `0.0042%` por redondeo visual |

### Caso dorado RAW

Para validar todo el preprocessing Rust, se generó un golden file con Python:

```text
Python edge:   59.6658%
Demo WASM:     59.67%
Diff visual:   0.0042%
```

Test Rust:

```text
Python: 0.59665781
Rust:   0.59665781
diff:   7.31e-09
```

Esto certifica que la cadena completa se comporta de forma equivalente:

```text
Input crudo
    ↓
Imputación numérica
    ↓
Imputación categórica
    ↓
Ordinal Income/Education
    ↓
Flags is_unknown
    ↓
One-Hot Gender/Marital/Card
    ↓
StandardScaler
    ↓
Árboles XGBoost en Rust
    ↓
Sigmoid
    ↓
Probabilidad de churn
```

---

---

## 🛡️ Integridad Metodológica y Transparencia (Anti-Gaming)

En Machine Learning es común inflar métricas accidentalmente mediante *data leakage*, *p-hacking* o manipulando el entorno de ejecución. Este proyecto aplica reglas estrictas para garantizar que los resultados sean honestos, reproducibles y aptos para producción:

| Práctica de Transparencia | Implementación en este proyecto |
|---|---|
| **Cero Data Leakage** | El `split train/test` se realiza *antes* de cualquier imputación, escalado o SMOTE. El test set permanece completamente intocado hasta la evaluación final. |
| **Trade-offs Honestos** | La optimización edge no oculta su costo. Se reporta abiertamente la caída de AUC (`-0.0036`) al reducir la complejidad del modelo, en lugar de manipular métricas. |
| **Threshold Congelado** | El umbral de negocio (`0.38`) se definió mediante validación (F2-score) y se congeló. No se ajustó iterativamente contra el test set para inflar el *Recall*. |
| **Sin `RUSTFLAGS` mágicos** | El motor WASM se compiló con perfiles estándar (`--release`). **No** se utilizó `target-cpu=native` ni flags ocultos para forzar artificialmente la paridad o la velocidad. |
| **Paridad Exacta (No aproximada)** | No nos conformamos con "diferencias pequeñas" por redondeo. Se replicó deliberadamente la aritmética `f32` interna de XGBoost en Rust para lograr un `max diff` de `7.31e-09`. |

> 📌 **Nota sobre Benchmarks de Rendimiento:** La optimización de *precisión, peso y paridad matemática* está certificada. Las métricas de latencia actuales (`< 1ms`) son empíricas, medidas directamente en el navegador. El proyecto **aún no implementa una suite formal de benchmarks de throughput con `criterion.rs`** en Rust nativo para evitar sesgos de medición del entorno del navegador (JIT, GC, fetch). Añadir `criterion` para benchmarks aislados es el siguiente paso en el roadmap de MLOps.

---

## 🦀 Motor Rust / WASM (Espejo de Inferencia Python)

Para garantizar una paridad numérica certificada (`max diff ~ 7.31e-09`), el motor en Rust **no utiliza librerías externas de Machine Learning**, sino que implementa una réplica exacta, línea por línea y función por función, del pipeline de preprocesamiento y la lógica de inferencia desarrollada originalmente en Python.

El motor en Rust es responsable de:

1. **Cargar el modelo XGBoost en JSON:** Parsear la estructura de árboles, nodos, splits y pesos base exactamente como lo exporta XGBoost.
2. **Cargar el contrato de inferencia:** Leer y validar los metadatos de configuración (`inference_contract_edge.json`).
3. **Replicar el preprocesamiento RAW:**
   - Aplicar imputación numérica (`strategy='median'`) y categórica (`strategy='most_frequent'`) replicando los `SimpleImputer` ajustados en el `train` de Python.
   - Ejecutar el **encoding ordinal** para variables como ingresos y educación.
   - Generar los flags de control (`is_unknown`) y las categorías **One-Hot** con idéntico orden y comportamiento ante categorías no vistas.
   - Aplicar el `StandardScaler` usando la media y desviación estándar fijas aprendidas en Python.
4. **Ejecutar la inferencia con aritmética `f32`:** Recorrer los árboles de decisión acumulando los márgenes replicando el comportamiento interno del motor de XGBoost (usar `f64` introducía divergencias en splits cercanos al umbral).
5. **Calcular la probabilidad final:** Convertir el margen acumulado (sumado al `base_score` del modelo) a probabilidad mediante la función **sigmoide** y compararlo contra el umbral de negocio (`0.38`).

### 🏗️ Decisiones de ingeniería clave

| Decisión | Justificación |
|---|---|
| **Parsing con `serde` + tipos estrictos** | Los nodos del árbol se modelan como `enum` (`Leaf` / `Split`) para eliminar estados inválidos en tiempo de compilación. |
| **`Zero-allocation` en el hot-path** | La inferencia reutiliza buffers pre-asignados; no hay allocaciones durante el recorrido del árbol. |
| **Fail-safe en el contrato** | Si el contrato tiene una versión incompatible, el módulo falla en la carga, no durante la inferencia. |
| **Interop JS ↔ Rust** | Los features entran como `&[f32]` desde un `Float32Array` de JavaScript, evitando copias de memoria innecesarias. |

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

Este detalle fue clave para lograr:

```text
max diff ≈ 7.31e-09
```

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
Probabilidad: 59.67%
Threshold:    0.38
Decisión:     ALTO RIESGO DE CHURN
```

---

## 📦 Entorno de Desarrollo (DevContainer)

Este proyecto está configurado con **Dev Containers** (estándar de Microsoft/GitHub) para garantizar una reproducibilidad del 100% y evitar conflictos de dependencias locales. No necesitas instalar Python, Rust, `uv` ni `wasm-pack` manualmente en tu sistema operativo.

### 🌩️ Opción A: GitHub Codespaces (Cero instalación)
1. Ve al repositorio en GitHub.
2. Haz clic en el botón verde **`<> Code`**.
3. Selecciona la pestaña **Codespaces** y haz clic en **"Create codespace on main"**.
4. En menos de 2 minutos, tendrás un VS Code completo en tu navegador con Python 3.11, Rust, `uv` y `wasm-pack` preconfigurados y listos para usar.

### 💻 Opción B: VS Code Local
1. Clona el repositorio y ábrelo en **Visual Studio Code**.
2. Asegúrate de tener instalada la extensión oficial **Dev Containers** de Microsoft.
3. VS Code detectará el archivo `.devcontainer/devcontainer.json` y te mostrará una notificación en la esquina inferior derecha.
4. Haz clic en **"Reopen in Container"**.
5. El entorno aislado se construirá automáticamente. Al terminar, la terminal integrada ya estará dentro del contenedor con todas las herramientas listas.

---

## 🚀 Quickstart

### Requisitos

```text
Python 3.11+
uv o pip
Rust stable
wasm-pack
Navegador moderno
```

### 1. Instalar wasm-pack

```bash
cargo install wasm-pack
```

### 2. Ejecutar pipeline Python

En Windows:

```bash
uv run py run_all.py
```

En Linux/macOS, si tienes `make` configurado:

```bash
make all
```

El pipeline ejecuta:

```text
1. train_model.py
2. export_edge.py
3. validate_parity.py
4. train_edge_model.py
5. export_edge_lite.py
```

### 3. Validar paridad RAW

```bash
uv run py scripts/validate_parity_raw.py
```

Esto genera:

```text
edge/raw_parity_case.json
```

### 4. Ejecutar tests Rust

```bash
cd edge/churn-wasm
cargo test -- --nocapture
```

Test de paridad RAW:

```bash
cargo test --test parity_raw -- --nocapture
```

### 5. Compilar WASM

```bash
cd edge/churn-wasm
wasm-pack build --target web --release
```

### 6. Levantar demo local

```bash
cd edge
python -m http.server 8000
```

Abrir:

```text
http://localhost:8000
```

---

## 📦 Estructura del proyecto

```text
churn-edge-ai/
│
├── .gitignore
├── pyproject.toml
├── README.md
├── run_all.py
├── Makefile
│
├── data/
│   └── raw/
│       └── Credit_card_churn.csv
│
├── model_artifacts/
│   ├── xgb_model.joblib
│   ├── xgb_model_edge.joblib
│   ├── scaler.joblib
│   ├── onehot_encoder.joblib
│   ├── num_imputer.joblib
│   ├── cat_imputer.joblib
│   ├── ordinal_imputer.joblib
│   ├── model_config.joblib
│   ├── model_config_edge.joblib
│   ├── X_train_scaled_sample.csv
│   └── X_train_edge_sample.csv
│
├── edge/
│   ├── xgb_model.json
│   ├── xgb_model_edge.json
│   ├── inference_contract.json
│   ├── inference_contract_edge.json
│   ├── parity_sample.json
│   ├── parity_sample_edge.json
│   ├── raw_parity_case.json
│   ├── sample_scaled.npy
│   ├── sample_scaled_edge.npy
│   ├── expected_proba.npy
│   ├── expected_proba_edge.npy
│   ├── expected_margin.npy
│   ├── expected_margin_edge.npy
│   ├── index.html
│   ├── demo.js
│   └── churn-wasm/
│       ├── Cargo.toml
│       ├── src/
│       │   ├── lib.rs
│       │   ├── contract.rs
│       │   ├── preprocessing.rs
│       │   ├── model.rs
│       │   └── inference.rs
│       └── tests/
│           └── parity_raw.rs
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── train.py
│   └── export_edge.py
│
├── scripts/
│   ├── train_model.py
│   ├── export_edge.py
│   ├── validate_parity.py
│   ├── train_edge_model.py
│   ├── export_edge_lite.py
│   └── validate_parity_raw.py
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 📄 Contrato de inferencia

El contrato de inferencia garantiza que Python, Rust y frontend interpreten el modelo exactamente igual.

El contrato edge incluye:

```text
feature_order
numeric_imputer
categorical_imputer
ordinal mappings
one-hot categories
scaler mean/scale
best_threshold
xgboost base_score
```

Esto evita problemas clásicos de despliegue:

- Features desordenadas.
- Categorías nuevas mal manejadas.
- Escalado inconsistente.
- Imputación distinta entre train y serving.
- Threshold incorrecto.
- Base score mal interpretado.

---

## 🧰 Comandos principales

| Comando | Descripción |
|---|---|
| `uv run py run_all.py` | Ejecuta pipeline completo en Windows |
| `make all` | Ejecuta pipeline completo en Linux/macOS |
| `uv run py scripts/train_model.py` | Entrena modelo full |
| `uv run py scripts/export_edge.py` | Exporta modelo full a edge |
| `uv run py scripts/validate_parity.py` | Valida paridad Python ↔ JSON |
| `uv run py scripts/train_edge_model.py` | Entrena modelo edge |
| `uv run py scripts/export_edge_lite.py` | Exporta modelo edge liviano |
| `uv run py scripts/validate_parity_raw.py` | Genera golden file RAW |
| `cargo test` | Ejecuta tests Rust |
| `cargo test --test parity_raw` | Ejecuta paridad RAW Python ↔ Rust |
| `wasm-pack build --target web --release` | Compila WASM |
| `python -m http.server 8000` | Sirve demo local |

---

## ✅ Tests

### Tests Python

Los validadores principales son:

```bash
uv run py scripts/validate_parity.py
uv run py scripts/validate_parity_raw.py
```

El primero valida:

```text
Python/XGBoost ↔ JSON exportado
```

El segundo valida:

```text
Input crudo → preprocessing → probabilidad final
Python ↔ Rust
```

### Tests Rust

```bash
cd edge/churn-wasm
cargo test -- --nocapture
```

Incluye:

- Sigmoid.
- Error handling.
- Lógica one-hot.
- Conversión de base score.
- Paridad RAW end-to-end.

Salida esperada del test de paridad RAW:

```text
Python: 0.59665781
Rust:   0.59665781
diff:   7.31e-09
```

---

## 🔄 CI/CD

El proyecto está preparado para CI con GitHub Actions.

Objetivo del CI:

- Validar paridad Python ↔ JSON.
- Validar paridad RAW Python ↔ Rust.
- Ejecutar tests de Rust.
- Compilar WASM.
- Evitar regresiones numéricas.

Workflow sugerido:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  python-parity:
    name: Python parity checks
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install numpy pandas xgboost joblib scikit-learn imbalanced-learn

      - name: Parity Python ↔ JSON
        run: python scripts/validate_parity.py

      - name: Parity RAW → proba
        run: python scripts/validate_parity_raw.py

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

      - name: RAW parity test
        run: cd edge/churn-wasm && cargo test --test parity_raw -- --nocapture

      - name: Build WASM
        run: cd edge/churn-wasm && wasm-pack build --target web --release
```

---

## 🧾 Artefactos generados

Después del pipeline completo, se generan artefactos Python y edge.

### Python artifacts

```text
model_artifacts/
    xgb_model.joblib
    scaler.joblib
    onehot_encoder.joblib
    num_imputer.joblib
    cat_imputer.joblib
    ordinal_imputer.joblib
    model_config.joblib
```

### Edge artifacts full

```text
edge/
    xgb_model.json
    inference_contract.json
    parity_sample.json
    sample_scaled.npy
    expected_proba.npy
    expected_margin.npy
```

### Edge artifacts lite

```text
edge/
    xgb_model_edge.json
    inference_contract_edge.json
    parity_sample_edge.json
    sample_scaled_edge.npy
    expected_proba_edge.npy
    expected_margin_edge.npy
    raw_parity_case.json
```

---

## 🧠 Modelo Full vs Modelo Edge

### Modelo Full

- Máxima precisión.
- Más features.
- Más árboles.
- Mayor peso.
- Apto para backend, API o batch scoring.

### Modelo Edge

- Peso reducido.
- Inferencia rápida.
- Menor complejidad.
- Compatible con navegador.
- Caída controlada de performance.
- Ideal para edge, demos interactivas y privacidad local.

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
- Comparación periódica contra modelo full.

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
- Accumulación de margin.
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

## 🌱 Roadmap

- [ ] Frontend interactivo con React/Svelte.
- [ ] Sliders para análisis what-if.
- [ ] Gráfica de probabilidad en tiempo real.
- [ ] Modo offline con Service Worker.
- [ ] Publicación del paquete WASM en npm.
- [ ] Integración con MLflow/DVC.
- [ ] Explicabilidad local aproximada.
- [ ] Dashboard de monitoreo de drift.
- [ ] Benchmark en dispositivos móviles reales.
- [ ] Soporte para múltiples modelos edge.
- [ ] Exportación adicional a ONNX como comparación.
- [ ] CLI nativo Rust para serverless.

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

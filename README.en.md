# 🪶 Churn Edge AI — 100% in-browser churn prediction with Rust + WebAssembly

![CI](https://github.com/Sergius-DS/churn-edge-ai/actions/workflows/ci.yml/badge.svg)
![Dev Container](https://img.shields.io/badge/Dev_Container-Ready-blue?logo=visualstudiocode&logoColor=white)
![Rust](https://img.shields.io/badge/Rust-stable-orange)
![WASM](https://img.shields.io/badge/WASM-edge-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-JSON-green)
![Model Size](https://img.shields.io/badge/edge_model-229_KB-orange)
![Parity](https://img.shields.io/badge/parity_python%E2%86%94wasm-max_diff_7.3e--9-brightgreen)
![Inference](https://img.shields.io/badge/inference-100%25_client--side-blue)
![License](https://img.shields.io/badge/License-MIT-green)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Sergius-DS/churn-edge-ai?quickstart=1)

> Churn Edge AI trains an XGBoost model in Python, exports it as an edge-optimized JSON artifact, and executes full inference via Rust and WebAssembly directly within the browser.
> The client sends zero sensitive data to a server: predictions occur locally, with certified numerical parity against the Python baseline.

## 🎯 Executive Summary

This project demonstrates how to transition a classic Machine Learning model from a Python-based training pipeline into a lightweight, portable, and private inference engine that runs entirely on the client side.

| Metric | Result |
| --- | --- |
| Full Model — Test AUC | 0.9930 |
| Edge Model — Test AUC | 0.9894 |
| AUC drop (Full vs. Edge) | 0.0036 |
| Full Features | 28 |
| Edge Features | 12 |
| Full Trees | 500 |
| Edge Trees | 200 |
| Full JSON Model Weight | ~819 KB |
| Edge JSON Model Weight | ~229 KB |
| Weight Reduction | 72.1% |
| Python ↔ JSON Parity | 0.00e+00 |
| Rust/WASM ↔ Python Parity (In-browser) | 2.02e-10 |
| End-to-End RAW ↔ Rust Parity | 7.31e-09 |
| Inference | 100% client-side |
| Data sent to server | 0 bytes |

## 🧩 The Problem

In traditional churn prediction systems:
- The model resides on a server.
- Every prediction requires transmitting client data.
- Latency increases.
- Inference costs scale up.
- The privacy risk surface expands.
- Deployment relies heavily on Python, containers, APIs, or cloud services.

For use cases where privacy, marginal cost, and speed are paramount, piping raw data to a backend is not always the optimal architecture.

## ✅ The Solution

Churn Edge AI implements a complete, production-grade ML pipeline:
- Training and evaluation in Python using XGBoost.
- Elimination of data leakage and strictly controlled preprocessing.
- Edge model optimization:
  - Feature reduction.
  - Fewer trees.
  - Shallower depth.
  - Stronger regularization.
  - Smaller artifact footprint.
- Model export to JSON.
- Generation of a strict inference contract.
- Inference engine written in Rust.
- Compilation to WebAssembly.
- Mathematical parity validation between Python and Rust/WASM.
- Backend-less browser demo.

## 🏗️ Architecture

```text
┌────────────────────────────── PYTHON — TRAIN ──────────────────────────────┐
│                                                                             │
│  Credit_card_churn.csv                                                      │
│        │                                                                    │
│        ▼                                                                    │
│  Leakage / ID removal                                                       │
│        │                                                                    │
│        ▼                                                                    │
│  Train/test split                                                           │
│        │                                                                    │
│        ▼                                                                    │
│  Imputation → Ordinal → One-Hot → SMOTE → Scaling                           │
│        │                                                                    │
│        ▼                                                                    │
│  Full XGBoost + CV + early stopping                                         │
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
│  Top feature selection                                                      │
│  Fewer trees / shallower depth                                              │
│  Stronger regularization                                                    │
│  Edge export:                                                               │
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
│      contract.rs          inference contract parsing                        │
│      preprocessing.rs     raw → impute → ordinal → one-hot → scale          │
│      model.rs             XGBoost JSON tree parser                          │
│      inference.rs         tree traversal with f32 arithmetic                │
│      lib.rs               wasm_bindgen APIs                                 │
│                                                                             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────── BROWSER ──────────────────────────────────┐
│                                                                             │
│  index.html + demo.js                                                       │
│      loads churn_wasm.js                                                    │
│      loads xgb_model_edge.json                                              │
│      loads inference_contract_edge.json                                     │
│      executes local prediction                                              │
│      validates parity with Python                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔒 Privacy by Design

Inference runs entirely in the browser.

| Property | Status |
| --- | --- |
| Client data sent to server | 0 bytes |
| Backend required to predict | No |
| Works offline after loading assets | Yes |
| Server-side inference cost per request | $0.00 |
| Privacy | Local-first / Zero-Trust |
| Personal data processing surface | Minimized |

*Note: This approach favors privacy-by-design architectures and facilitates data minimization strategies. It does not constitute legal advice nor guarantee GDPR compliance on its own.*

---

## 📊 Model Results

### Full vs Edge Comparison

| Metric | Full | Edge | Δ |
| --- | --- | --- | --- |
| Features | 28 | 12 | -16 |
| Trees | 500 | 200 | -300 |
| max_depth | 4 | 3 | -1 |
| CV AUC | 0.9986 | 0.9975 | -0.0011 |
| Test AUC | 0.9930 | 0.9894 | -0.0036 |
| Precision (churn class) | 0.91 | 0.88 | -0.03 |
| Recall (churn class) | 0.91 | 0.90 | -0.01 |
| F1 (churn class) | 0.91 | 0.89 | -0.02 |
| Threshold | 0.38 | 0.38 | same |
| JSON Weight | 818,904 bytes | 228,629 bytes | -72.1% |

### Interpretation

The edge model sacrifices very little predictive power in exchange for:
- Reducing model weight by 72.1%.
- Decreasing feature count from 28 to 12.
- Lowering inference complexity.
- Making browser execution viable.
- Maintaining a stable business threshold.

The AUC drop was:
`Δ AUC = 0.003620`

This value is well within the acceptable range for an edge deployment.


---

## 🧠 Dataset and Preparation

The pipeline starts with a bank churn dataset containing:
- Initial rows:     10,127
- Initial columns:  23
- Target:
  - 0 → 8,500
  - 1 → 1,627

### Leakage Removal

Columns incompatible with a real-world production environment were dropped:
- `CLIENTNUM`
- `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1`
- `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2`

Additionally, the following were cast to numeric:
- `Total_Amt_Chng_Q4_Q1`
- `Total_Ct_Chng_Q4_Q1`

---

## 🚫 Anti-Data Leakage Pipeline

The preprocessing order is strictly designed to prevent data leakage between training and validation.

```text
Train/test split
     ↓
Numeric imputation (fitted on train only)
     ↓
Categorical imputation (fitted on train only)
     ↓
Ordinal transformation (fitted on train only)
     ↓
One-Hot encoding (fitted on train only)
     ↓
SMOTE (applied to train only)
     ↓
Scaling (fitted on train only)
     ↓
Training
     ↓
Final evaluation on untouched test set
```

---

## 🪶 Edge Optimization

The edge model was specifically trained for lightweight WASM inference.

### Strategy

- Selection of the most important features from the full model.
- Shallower depth.
- Fewer trees.
- Stronger regularization.
- Early stopping with cross-validation.
- Export to JSON.
- Parity validation.

### Edge Features

The 12 selected features were:

- `Total_Trans_Ct`
- `Total_Revolving_Bal`
- `Total_Ct_Chng_Q4_Q1`
- `Months_Inactive_12_mon`
- `Contacts_Count_12_mon`
- `Total_Trans_Amt`
- `Total_Relationship_Count`
- `Avg_Utilization_Ratio`
- `Total_Amt_Chng_Q4_Q1`
- `Dependent_count`
- `Gender_F`
- `Marital_Status_Single`

### Edge Parameters

Configuration used for the edge model:

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

## 📈 Classification Metrics

### Full Model

| Metric | Value |
| --- | --- |
| Test AUC | 0.9930 |
| F1 Score (churn class) | 0.91 |
| Precision (churn class) | 0.91 |
| Recall (churn class) | 0.91 |
| Threshold | 0.38 |

### Edge Model

| Metric | Value |
| --- | --- |
| Test AUC | 0.9894 |
| F1 Score (churn class) | 0.89 |
| Precision (churn class) | 0.88 |
| Recall (churn class) | 0.90 |
| Threshold | 0.38 |

### Confusion Matrix Comparison

**Full Model:**

| Actual \ Predicted | No Churn | Churn |
| --- | --- | --- |
| **No Churn** | 1,662 | 39 |
| **Churn** | 31 | 294 |

**Edge Model:**

| Actual \ Predicted | No Churn | Churn |
| --- | --- | --- |
| **No Churn** | 1,662 | 39 |
| **Churn** | 31 | 294 |

### Analysis

The edge model maintains strong discriminative power with an AUC of 0.9894, only 0.0036 points below the full model. The slight reduction in precision (0.88 vs 0.91) is an acceptable trade-off given the 72.1% reduction in model size and the elimination of server-side inference costs.

The recall remains stable at 0.90, ensuring that the model continues to identify the vast majority of churn cases, which is the primary business objective.

---

## 🎚️ Business Threshold

The threshold used is:

`threshold = 0.38`

`0.5` is not used because the business cost is asymmetrical.

In churn prediction:

- A false negative means failing to detect a customer who is about to leave.
- A false positive means flagging a customer who ultimately stays.

Therefore, a recall-oriented metric was optimized using an F2-score criterion, which heavily penalizes missing a true churn event.

---

## 🧪 Certified Mathematical Parity

One of the project's most critical requirements is that Rust/WASM produces the exact same result as Python/XGBoost.

Parity was validated across multiple layers.

| Layer | What it compares | Tool | Max diff |
| --- | --- | --- | --- |
| 1 | Python ↔ Exported JSON | `scripts/validate_parity.py` | 0.00e+00 |
| 2 | JSON ↔ Rust (using pre-scaled features) | In-browser `verify_parity` | 2.02e-10 |
| 3 | RAW Input → Probability (Python ↔ Rust) | golden file + `cargo test` | 7.31e-09 |
| 4 | WASM Demo vs Python | Manual cross-check | 0.0042% (visual rounding) |

---

### Golden RAW Case

To validate the entire Rust preprocessing pipeline, a golden file was generated using Python:

```text
Python edge:   59.6658%
WASM Demo:     59.67%
Visual diff:   0.0042%
```

Rust Test:

```text
Python: 0.59665781
Rust:   0.59665781
diff:   7.31e-09
```

This certifies that the entire chain behaves identically:

```text
Raw Input
     ↓
Numeric Imputation
     ↓
Categorical Imputation
     ↓
Ordinal Income/Education
     ↓
is_unknown Flags
     ↓
One-Hot Gender/Marital/Card
     ↓
StandardScaler
     ↓
XGBoost Trees in Rust
     ↓
Sigmoid
     ↓
Churn Probability
```
---

## 🛡️ Methodological Integrity and Transparency (Anti-Gaming)

In Machine Learning, it is common to accidentally inflate metrics through data leakage, p-hacking, or manipulating the execution environment. This project enforces strict rules to ensure results are honest, reproducible, and production-ready:

| Transparency Practice | Implementation in this project |
| --- | --- |
| **Zero Data Leakage** | The `train/test split` occurs *before* any imputation, scaling, or SMOTE. The test set remains completely untouched until final evaluation. |
| **Honest Trade-offs** | Edge optimization does not hide its costs. The AUC drop (`-0.0036`) resulting from reduced model complexity is openly reported, rather than manipulating metrics. |
| **Frozen Threshold** | The business threshold (`0.38`) was defined via validation (F2-score) and frozen. It was not iteratively tuned against the test set to artificially inflate `Recall`. |
| **No magic RUSTFLAGS** | The WASM engine was compiled using standard profiles (`--release`). No `target-cpu=native` or hidden flags were used to force parity or speed. |
| **Exact Parity (Not approximate)** | We do not settle for "small differences" due to rounding. XGBoost's internal `f32` arithmetic was deliberately replicated in Rust to achieve a `max diff` of `7.31e-09`. |

*📌 Note on Performance Benchmarks: Precision, weight, and mathematical parity optimizations are certified. Current latency metrics (`< 1ms`) are empirical, measured directly in the browser. The project has not yet implemented a formal throughput benchmark suite with `criterion.rs` in native Rust to avoid browser environment measurement biases (JIT, GC, fetch). Adding `criterion` for isolated benchmarks is the next step in the MLOps roadmap.*

---

## 🦀 Rust / WASM Engine (Python Inference Mirror)

To guarantee certified numerical parity (`max diff ~ 7.31e-09`), the Rust engine does not rely on external Machine Learning libraries. Instead, it implements an exact, line-by-line and function-by-function replica of the preprocessing pipeline and inference logic originally developed in Python.

The Rust engine is responsible for:

- **Loading the XGBoost model from JSON:** Parsing the tree structure, nodes, splits, and base weights exactly as exported by XGBoost.
- **Loading the inference contract:** Reading and validating configuration metadata (`inference_contract_edge.json`).
- **Replicating RAW preprocessing:**
  - Applying numeric (`strategy='median'`) and categorical (`strategy='most_frequent'`) imputation, mirroring the `SimpleImputers` fitted on the Python train set.
  - Executing ordinal encoding for variables like income and education.
  - Generating control flags (`is_unknown`) and One-Hot categories with identical ordering and handling of unseen categories.
  - Applying the `StandardScaler` using the fixed mean and standard deviation learned in Python.
- **Executing inference with `f32` arithmetic:** Traversing decision trees and accumulating margins, replicating XGBoost's internal engine behavior (using `f64` introduced divergences near split thresholds).
- **Calculating final probability:** Converting the accumulated margin (added to the model's `base_score`) into a probability via the sigmoid function and comparing it against the business threshold (`0.38`).

---

## 🏗️ Key Engineering Decisions

| Decision | Justification |
| --- | --- |
| **Parsing with `serde` + strict types** | Tree nodes are modeled as `enum` (`Leaf` / `Split`) to eliminate invalid states at compile time. |
| **Zero-allocation in the hot-path** | Inference reuses pre-allocated buffers; no allocations occur during tree traversal. |
| **Fail-safe contract** | If the contract has an incompatible version, the module fails at load time, not during inference. |
| **JS ↔ Rust Interop** | Features are passed as `&[f32]` from a JavaScript `Float32Array`, avoiding unnecessary memory copies. |

---

## 💡 Case Study: why `f32` and not `f64`

XGBoost evaluates splits and accumulates margins using 32-bit precision internally.
Initially, the Rust engine calculated everything in `f64` (double precision), which is Rust's default for math. However, this broke parity in edge cases where a value fell extremely close to a split threshold.

**The problem:** Rounding differences between `f32` and `f64` caused Rust to take different branches in the decision tree, accumulating errors that resulted in divergences greater than `1e-06`.

**The solution:** Deliberately replicate XGBoost's internal arithmetic:
- `f64` features → `f32` features
- split comparisons in `f32`
- margin accumulation in `f32`
- final margin converted to `f64` for sigmoid

This detail was key to achieving:
`max diff ≈ 7.31e-09`

**Lesson learned:** In ML inference systems, numerical precision is not just about mathematical exactness, but about replicating the original framework's behavior. What is "more precise" (`f64`) can be incorrect if the model was trained with `f32`.

---

## 📁 Main Modules

```text
edge/churn-wasm/src/
    lib.rs             Entry point exposed to JavaScript (wasm-bindgen)
    contract.rs        Deserialization and validation of the inference contract
    preprocessing.rs   Exact replica of the raw → scaled features pipeline
    model.rs           Typed parser for the XGBoost JSON tree
    inference.rs       Tree traversal and margin calculation engine (f32)
```
---

## 🌐 Browser Demo

The demo allows you to:
- Load the WASM model locally.
- Load the edge JSON model.
- Load the edge contract.
- Input client features.
- Calculate churn probability without a backend.
- Automatically verify parity.

### Prediction Example

```text
Probability: 59.67%
Threshold:    0.38
Decision:     HIGH CHURN RISK
```
---

## 📦 Development Environment (DevContainer)

This project is configured with Dev Containers (Microsoft/GitHub standard) to guarantee 100% reproducibility and avoid local dependency conflicts. You do not need to manually install Python, Rust, `uv`, or `wasm-pack` on your host OS.

### 🌩️ Option A: GitHub Codespaces (Zero installation)
1. Go to the repository on GitHub.
2. Click the green `<> Code` button.
3. Select the **Codespaces** tab and click **"Create codespace on main"**.
4. In under 2 minutes, you will have a full VS Code environment in your browser with Python 3.11, Rust, `uv`, and `wasm-pack` pre-configured and ready to use.

### 💻 Option B: Local VS Code
1. Clone the repository and open it in Visual Studio Code.
2. Ensure you have the official **Dev Containers** extension by Microsoft installed.
3. VS Code will detect the `.devcontainer/devcontainer.json` file and show a notification in the bottom right corner.
4. Click **"Reopen in Container"**.
5. The isolated environment will build automatically. Once finished, the integrated terminal will be inside the container with all tools ready.
---

## 🚀 Quickstart

### Requirements
- Python 3.11+
- `uv` or `pip`
- Rust stable
- `wasm-pack`
- Modern browser

### 1. Install wasm-pack
```bash
cargo install wasm-pack
```

### 2. Run Python pipeline
On Windows:
```bash
uv run py run_all.py
```
On Linux/macOS (if `make` is configured):
```bash
make all
```
*The pipeline executes: `train_model.py` ➔ `export_edge.py` ➔ `validate_parity.py` ➔ `train_edge_model.py` ➔ `export_edge_lite.py`*

### 3. Validate RAW parity
```bash
uv run py scripts/validate_parity_raw.py
```
*This generates: `edge/raw_parity_case.json`*

### 4. Run Rust tests
```bash
cd edge/churn-wasm
cargo test -- --nocapture
```
*RAW parity test:*
```bash
cargo test --test parity_raw -- --nocapture
```

### 5. Compile WASM
```bash
cd edge/churn-wasm
wasm-pack build --target web --release
```

### 6. Start local demo
```bash
cd edge
python -m http.server 8000
```
Open: `http://localhost:8000`
---

## 📦 Project Structure

```text
churn-edge-ai/
 │
 ├── .gitignore
 ├── pyproject.toml
 ├── README.md
 ├── README.en.md
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

## 📄 Inference Contract

The inference contract ensures that Python, Rust, and the frontend interpret the model exactly the same way.

The edge contract includes:
- `feature_order`
- `numeric_imputer`
- `categorical_imputer`
- `ordinal mappings`
- `one-hot categories`
- `scaler mean/scale`
- `best_threshold`
- `xgboost base_score`

This prevents classic deployment pitfalls:
- Unordered features.
- Mishandled new categories.
- Inconsistent scaling.
- Different imputation between train and serving.
- Wrong threshold.
- Misinterpreted base score.
---

## 🧰 Main Commands

| Command | Description |
| --- | --- |
| `uv run py run_all.py` | Runs full pipeline on Windows |
| `make all` | Runs full pipeline on Linux/macOS |
| `uv run py scripts/train_model.py` | Trains full model |
| `uv run py scripts/export_edge.py` | Exports full model to edge |
| `uv run py scripts/validate_parity.py` | Validates Python ↔ JSON parity |
| `uv run py scripts/train_edge_model.py` | Trains edge model |
| `uv run py scripts/export_edge_lite.py` | Exports lightweight edge model |
| `uv run py scripts/validate_parity_raw.py` | Generates RAW golden file |
| `cargo test` | Runs Rust tests |
| `cargo test --test parity_raw` | Runs RAW Python ↔ Rust parity test |
| `wasm-pack build --target web --release` | Compiles WASM |
| `python -m http.server 8000` | Serves local demo |
---
## ✅ Tests

### Python Tests
The main validators are:

```bash
uv run py scripts/validate_parity.py
uv run py scripts/validate_parity_raw.py
```

- The first validates: **Python/XGBoost ↔ Exported JSON**
- The second validates: **Raw Input → preprocessing → final probability (Python ↔ Rust)**

### Rust Tests

```bash
cd edge/churn-wasm
cargo test -- --nocapture
```

Includes:
- Sigmoid.
- Error handling.
- One-hot logic.
- Base score conversion.
- End-to-end RAW parity.

Expected output for the RAW parity test:

```text
Python: 0.59665781
Rust:   0.59665781
diff:   7.31e-09
```
---

## 🔄 CI/CD

The project is CI-ready with GitHub Actions.

**CI Goal:**
- Validate Python ↔ JSON parity.
- Validate RAW Python ↔ Rust parity.
- Run Rust tests.
- Compile WASM.
- Prevent numerical regressions.

**Suggested Workflow:**

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

## 🧾 Generated Artifacts

After the full pipeline runs, Python and edge artifacts are generated.

**Python artifacts**

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

**Edge full artifacts**

```text
edge/
    xgb_model.json
    inference_contract.json
    parity_sample.json
    sample_scaled.npy
    expected_proba.npy
    expected_margin.npy
```

**Edge lite artifacts**

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

## 🧠 Full Model vs Edge Model

**Full Model**
- Maximum precision.
- More features.
- More trees.
- Larger weight.
- Suited for backend, API, or batch scoring.

**Edge Model**
- Reduced weight.
- Fast inference.
- Lower complexity.
- Browser-compatible.
- Controlled performance drop.
- Ideal for edge, interactive demos, and local privacy.
---

## 💼 Business Impact

| Area | Impact |
| --- | --- |
| **Infrastructure** | Reduces server-side inference costs |
| **Privacy** | Raw data never leaves the device |
| **Latency** | Immediate local prediction |
| **Retention** | Early detection of churn risk |
| **Product** | Interactive experience without a backend |
| **Compliance** | Favors data minimization |
| **Costs** | Fewer dependencies on APIs or cloud services |
---

## 🔐 Security and Considerations

This project follows privacy-by-design principles, but there are important considerations:
- The model is distributed to the client.
- The model should not be considered a secret if published in the browser.
- Local inference protects user data, but not necessarily the model's IP.
- It should not be used as the sole anti-fraud mechanism.

If deployed to production, it is recommended to:
- Sign artifacts.
- Use Subresource Integrity for WASM/JS.
- Version models.
- Register contract hashes.
- Monitor drift.
- Maintain an audit trail of changes.
---

## 📡 Production and Monitoring

To take this system to production, the following is recommended:

**Data Monitoring**
- Numeric feature drift.
- Categorical drift.
- Shifts in prediction distribution.
- Changes in actual churn rate.

**Model Monitoring**
- AUC/F2 over rolling windows.
- Threshold stability.
- Delta between prediction and actual outcome.
- Periodic comparison against the full model.

**MLOps**
- Model versioning.
- Artifact hashing.
- Inference contract registry.
- Parity tests in CI.
- Reproducible training with fixed seeds.
- Canary deployments for new edge models.
---

## 🐞 Troubleshooting

**Parity fails by a tiny margin**
- Verify that Rust is using `f32` arithmetic for:
  - Split comparisons.
  - Margin accumulation.
  - Tree evaluating.

**`UnicodeEncodeError` on Windows**
- If a script writes JSON with Unicode characters, always use:
  `encoding="utf-8"`
  Example:
  `out_path.write_text(payload, encoding="utf-8")`
  Or run with:
  `PYTHONUTF8=1`

**The demo doesn't load WASM**
- Verify that these exist:
  `edge/churn-wasm/pkg/churn_wasm.js`
  `edge/churn-wasm/pkg/churn_wasm_bg.wasm`
  Both must exist after:
  `wasm-pack build --target web --release`

**The browser caches the WASM**
- Perform a hard reload:
  `Ctrl + Shift + R`
  Or clear the cache.
---

## 🌱 Roadmap

- [ ] Interactive frontend with React/Svelte.
- [ ] Sliders for what-if analysis.
- [ ] Real-time probability graph.
- [ ] Offline mode with Service Worker.
- [ ] Publish WASM package to npm.
- [ ] Integration with MLflow/DVC.
- [ ] Approximate local explainability.
- [ ] Drift monitoring dashboard.
- [ ] Benchmarking on real mobile devices.
- [ ] Support for multiple edge models.
- [ ] Additional ONNX export for comparison.
- [ ] Native Rust CLI for serverless.
---

## 📚 Tech Stack

- **Python:** pandas, scikit-learn, imbalanced-learn, XGBoost
- **Rust:** wasm-bindgen, wasm-pack
- **WebAssembly**
- **JavaScript:** Vanilla
- **CI/CD:** GitHub Actions

**Reproducibility:**
`SEED = 44`
---

## 📌 Final Summary

Churn Edge AI transforms a traditional bank churn model into a lightweight, private, zero-marginal-cost edge inference system.

It combines:
- Python for training
- XGBoost for modeling
- JSON as a portable format
- Rust for secure, fast inference
- WebAssembly for browser execution
- Certified parity via testing

It is a highly relevant use case for:
- Banks.
- Fintech.
- Retail.
- Telecommunications.
- Privacy-sensitive digital products.
- Offline or low-connectivity environments.
- Client-side ML architectures.
---

## 📫 Contact

Author: `Sergio Francisco Rivera Bustamante`  
GitHub: `https://github.com/Sergius-DS`  
LinkedIn: `https://www.linkedin.com/in/sergio-francisco-rivera-bustamante/`  
Project: **Churn Edge AI**

---

## 📄 License

MIT

Copyright (c) 2026 Sergio Francisco Rivera Bustamante

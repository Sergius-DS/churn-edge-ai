<h1 align="center">🪶 Churn Edge AI</h1>
<p align="center">
  <b>100% in-browser churn prediction with Rust + WebAssembly</b><br><br>
  <a href="https://sergius-ds.github.io/churn-edge-ai/" target="_blank">
    <img src="https://img.shields.io/badge/🚀_TRY_LIVE_DEMO-2EA44F?style=for-the-badge&logo=githubpages&logoColor=white" alt="Try Live Demo">
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



> **Churn Edge AI** trains an XGBoost model in Python, exports it as edge-optimized JSON, and runs the full inference in **Rust/WebAssembly** inside the browser.  
> The client sends no sensitive data to a server: prediction happens locally, with certified numerical parity against Python.

---

## 🎯 Executive summary

This project demonstrates how to take a classic Machine Learning model from a Python training pipeline into a lightweight, portable, private inference engine running entirely on the client side.

| Metric | Result |
| --- | --- |
| Edge IoT Model — Test AUC | 0.9951 |
| Raw features | 25 (19 banking + 6 synthetic IoT) |
| Post-preprocessing features | 34 |
| Trees | 200 (max_depth 3) |
| Business threshold | 0.22 (F2 on internal pre-SMOTE split) |
| JSON model weight | 223,760 bytes (~224 KB) |
| Gzip weight | 47,175 bytes (~47 KB, 4.7× ratio) |
| RAW Python ↔ Rust parity | 1.81e-11 |
| Contract replica ≡ sklearn pipeline | max\|Δ\| = 0.000e+00 |
| SSOT | `data/curated/churn_curated.csv` · 9,996 rows |
| Inference | 100% client-side |
| Data sent to server | 0 bytes |

---

## 🧩 Problem

In traditional churn prediction systems:

- The model lives on a server.
- Every prediction requires sending client data.
- Latency increases.
- Inference cost increases.
- The privacy risk surface expands.
- Deployment depends on Python, containers, APIs, or cloud services.

For use cases where privacy, marginal cost, and speed matter, sending raw data to a backend is not always the best architecture.

---

## ✅ Solution

**Churn Edge AI** implements a complete, production-oriented ML pipeline:

1. Training and evaluation in Python with XGBoost.
2. Data leakage removal and controlled preprocessing.
3. Forensic curation with a fail-closed ingestion contract: dtype, per-column
   units, algebraic identity. Declared MNAR quarantine, no imputation.
4. Model export to JSON.
5. Inference contract generation.
6. Inference engine written in Rust.
7. Compilation to WebAssembly.
8. Mathematical parity validation between Python and Rust/WASM.
9. Backend-less browser demo.

---

## 🏗️ Architecture

```text
┌────────────────────────── PYTHON — CURATE + TRAIN ───────────────────────────┐
│                                                                              │
│  data/raw/Credit_card_churn.csv        (immutable · control group)           │
│        │                                                                     │
│        ▼   curate_raw.py · R1–R5 · fail-closed                               │
│  data/curated/churn_curated.csv        9,996 rows · INV-1/INV-2 = 100%       │
│  data/curated/quarantine.csv           131 MNAR rows, not imputed            │
│        │                                                                     │
│        ▼   generate_synthetic_iot.py · z_ref with support gate               │
│  + 6 synthetic IoT features            → 29 columns                          │
│        │                                                                     │
│        ▼   z GATE (blocking) · tests/test_iot_z_distribution.py              │
│        │                                                                     │
│        ▼   data_contract.validate → drop leakage/IDs → split                 │
│        │                                                                     │
│        ▼   Imputation → Ordinal → One-Hot → SMOTE → Scaling                  │
│        │                              25 raw → 34 features                   │
│        ▼   XGBoost edge (depth 3) + CV · optimize_threshold (F2)             │
│        │                                                                     │
│        ├──► model_artifacts/edge_iot/   (ignored: derivatives)               │
│        │       *.joblib · metrics.json · sample_provenance.json              │
│        │                                                                     │
│        └──► export_edge_iot.py · triple checkpoint                           │
│                xgb_model_edge_iot.json                                       │
│                inference_contract_edge_iot.json   (v2, byte-deterministic)   │
│                metrics_edge_iot.json              (justifies the threshold)  │
│                parity_sample_edge_iot.json · raw_parity_case_iot.json        │
│                export_manifest_edge_iot.json      (8 hashes + provenance)    │
│                                                                              │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      ▼
┌──────────────────────────── RUST / WASM — INFER ─────────────────────────────┐
│  contract.rs        deserializes the v2 contract                             │
│  preprocessing.rs   firewall → IoT bounds → impute → ordinal → onehot        │
│                     → scale                                                  │
│  model.rs           XGBoost tree parser (parallel arrays)                    │
│  inference.rs       f32 traversal + sigmoid                                  │
│  lib.rs             init_engine (OnceLock) · predict_churn                   │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      ▼
┌────────────────────────────────── BROWSER ───────────────────────────────────┐
│  index.html + demo.js                                                        │
│      loads churn_wasm.js                                                     │
│      loads xgb_model_edge_iot.json                                           │
│      loads inference_contract_edge_iot.json                                  │
│      local prediction · 0 bytes to the server                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔒 Privacy by design

Inference runs entirely in the browser.

| Property | Status |
|---|---|
| Client data sent to server | 0 bytes |
| Backend required to predict | No |
| Works offline after loading assets | Yes |
| Server-side inference cost per request | $0.00 |
| Privacy | Local-first / Zero-Trust |
| Personal data processing surface | Reduced |

> **Note:** this approach favors privacy-by-design architectures and can facilitate data minimization strategies. It does not constitute legal advice nor guarantee GDPR compliance on its own.

---

## 📊 Model results

### Full vs Edge comparison (retired lineage — ADR-0006, tag `baseline-fase3`)

> ⚠️ **Historical.** The Full 28f baseline and the Edge Lite 12f were retired
> because they descended from the mutated file. The current model is
> **Edge IoT** (34 features, threshold 0.22, AUC 0.9951). See the
> "Metrics — Edge IoT Model" section.

| Metric | Full | Edge | Δ |
| --- | ---: | ---: | ---: |
| Features | 28 | 12 | -16 |
| Trees | 500 | 200 | -300 |
| `max_depth` | 4 | 3 | -1 |
| CV AUC | 0.9986 | 0.9975 | -0.0011 |
| Test AUC | 0.9930 | 0.9894 | -0.0036 |
| Precision (churn class) | 0.91 | 0.88 | -0.03 |
| Recall (churn class) | 0.91 | 0.90 | -0.01 |
| F1 (churn class) | 0.91 | 0.89 | -0.02 |
| Threshold | 0.38 | 0.38 | same |
| JSON weight | 818,904 bytes | 228,629 bytes | -72.1% |


---

## 🧠 Dataset, forensic curation, and SSOT

The source file was not the canonical dataset. Three ratio-type columns
arrived with a decimal point read as a thousands separator, and 131 cells
contained dates where a ratio should have been.

| Artifact | Rows | sha256 (prefix) | Role |
| --- | --- | --- | --- |
| `data/raw/Credit_card_churn.csv` | 10,127 | `41f4e92afcf6…` | Immutable origin. Control group |
| `data/curated/churn_curated.csv` | 9,996 | `6bced2a3bf04…` | Operational SSOT |
| `data/curated/quarantine.csv` | 131 | `a49c8ecee4cd…` | Irrecoverable cells, not imputed |

**The ingestion contract rejects the origin with 4 violations:** two for
dtype, one for units, the fourth for algebraic identity (verbatim gate
output — these four literals are asserted in `test_data_contract.py`):

```text
Total_Amt_Chng_Q4_Q1: no numérico (Unable to parse string "2024-03-01 00:00:00" at position 158)
Total_Ct_Chng_Q4_Q1: no numérico (Unable to parse string "2024-05-02 00:00:00" at position 4)
Avg_Utilization_Ratio: 6880 fuera de [0.0,1.0] (min=0 max=999)
INV-2 violada: 32.0628%
```

The fourth does not depend on having guessed the units:
`utilization = revolving balance / limit`. It would have worked with a factor
of seven instead of a thousand.

The 131 irrecoverable cells were **not imputed**: the missingness turned out
to be correlated with the value (8 attrited customers removed vs 21 expected
under random missingness; z = −3.10, normal approximation to the binomial,
exact P(X ≤ 8) = 0.0005). Imputing the mean would have injected bias aligned
with the target.

Rules R1–R5, domain preconditions, and the full chain of custody in
`data/curated/curation_manifest.json`. Audit evidence in `docs/evidence/`.


### Leakage removal

Columns incompatible with a real production environment were dropped:

```text
CLIENTNUM
Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1
Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2
```

The two trend columns arrived as `object` because they contained dates. The
original pipeline coerced them with `errors="coerce"` and imputed them with
the mean: three silent laundering steps that turned a contract violation into
a plausible value. That coercion was **removed**. The contract guarantees
dtype; if the data does not parse, it is rejected.

---

## 🚫 Anti-data leakage pipeline

The preprocessing order is designed to prevent information leakage between training and validation.

```text
Train/test split
    ↓
Numeric imputation fitted on train only
    ↓
Categorical imputation fitted on train only
    ↓
Ordinal transformation fitted on train only
    ↓
One-Hot encoding fitted on train only
    ↓
SMOTE applied to train only
    ↓
Scaling fitted on train only
    ↓
Training
    ↓
Final evaluation on the untouched test set
```

### Principles

- The test set is never used to fit transformations.
- SMOTE is applied only on train.
- Imputers, encoders, and scaler are learned from train only.
- The threshold is defined on validation, not on test.
- Final metrics are computed on an uncontaminated test set.

⚠️ **Declared caveat:** SMOTE is applied to the full train set before CV.
Synthetic points are spread across folds, so validation AUC is optimistic by
construction. The publishable number is `test_auc`. Intra-fold SMOTE lives in
a separate branch: it changes every number, including `base_score`.

---

## 🪶 Edge IoT Model

The Edge IoT model was trained for lightweight WASM inference, combining
traditional banking features with synthetic IoT telemetry.

### Strategy

- **No feature selection:** uses the 19 banking features from the curated set + 6 IoT features.
- Lower depth (`max_depth: 3` vs `4` in the historical Full).
- Fewer trees (`200` vs `500`).
- Stronger regularization (`gamma: 0.1`, `reg_lambda: 1.5`).
- Early stopping with cross-validation (though structurally inert post-SMOTE).
- Export to JSON with inference contract v2.
- Parity validation across 5 layers (Python ↔ sklearn ↔ Rust).

### Features (25 raw → 34 post-preprocessing)

**19 banking features** (from the curated set):
```text
Customer_Age, Gender, Dependent_count, Education_Level, Marital_Status,
Income_Category, Card_Category, Months_on_book, Total_Relationship_Count,
Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit,
Total_Revolving_Bal, Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1,
Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio
```

**6 synthetic IoT features** (generated with a support gate):
```text
app_open_count_30d, contactless_usage_count_30d,
payment_flow_abandon_count_30d, nfc_failed_attempts_30d,
app_crash_count_30d, pos_latency_p95_ms
```

**Post-preprocessing (34 features):**
- **20 direct numeric:** 14 banking + 6 IoT
- **4 ordinal-derived:** `Income_Category_ordinal`, `Education_Level_ordinal`
  + their `is_unknown` flags (numeric columns, not one-hot)
- **10 one-hot:** Gender × 2, Marital_Status × 4, Card_Category × 4

The 5 banking categorical features split as: 2 → ordinal + flag, 3 → one-hot.
20 + 4 + 10 = 34. Verified against the export (`onehot=10`) and the contract.

### Edge IoT parameters

Config used for the Edge IoT model (`src/config.py::XGB_EDGE_PARAMS`):

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

> ⚠️ **Caveat:** `best_round = 200` coincides with `EDGE_CV_NUM_BOOST_ROUND`.
> It is the ceiling, not a measured optimum. See the "Business threshold"
> section.


---

## 📈 Metrics — Edge IoT Model

Test AUC: **0.995074** · Threshold: **0.22** · n_test: 2,000

```text
Confusion matrix:
[[1568  108]
 [   5  319]]

TP = 319 · FN = 5 · FP = 108 · TN = 1568
```

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Existing (0) | 1.00 | 0.94 | 0.97 |
| Attrited (1) | 0.75 | 0.98 | 0.85 |

Accuracy: 0.94 · Split: 7,996 train / 2,000 test · SMOTE: 7,996 → 13,402

Source of truth: `edge/metrics_edge_iot.json`. This table is a readable mirror; the artifact is the official record.

---

## 🎚️ Business threshold

`threshold = 0.22`

Optimized via F2 on an internal split of the train set **before SMOTE**, with
its own scaler fitted on that split. `0.5` was not used because the cost is
asymmetric: a false negative is a customer who leaves undetected.

The threshold lives in `model_artifacts/edge_iot/model_config.joblib` and is
published in the contract. **No test or frontend hardcodes it**: they read it
from the artifact.

Declared trade-off: FN 5 vs FP 108. This is the F2 objective operating as
designed, not a defect.

⚠️ `best_round = 200` coincides with `EDGE_CV_NUM_BOOST_ROUND`. It is the
ceiling, not a measured optimum: CV was still rising at iteration 199, and
early-stopping is structurally inert on a monotonic post-SMOTE curve. Recorded
in `metrics_edge_iot.json`.

---

## 🧪 Certified parity

| Layer | What it compares | Where | Max diff |
| --- | --- | --- | --- |
| 1 | Contract replica ≡ sklearn pipeline | `export_edge_iot.py` · Checkpoint 3 | **0.000e+00** |
| 2a | RAW → preprocessing → trees → probability (typical golden case, p≈0.15%) | tests/parity_raw_iot.rs::iot_raw_input_parity_with_python | **1.81e-11** |
| 2b | RAW → preprocessing → trees → probability (sensitive golden case, p≈47.44%) | tests/parity_raw_iot.rs::iot_sensitive_golden_parity | **1.02e-9** |
| 3 | The deployed contract deserializes in Rust | `tests/contract_live.rs` | pass/fail |
| 4 | Artifact integrity vs manifests | `check_artifact_hashes.py` | 13 checks |
| 5 | Contract controls present **and armed** | `check_contract_posture.py` | 4 controls |

Layer 1 is the one that closes the intermediate link: without it, the replica
could be self-consistent with Rust yet different from what the model saw
during training.

**Golden case:** real row from the curated set + IoT (`CLIENTNUM 720014358`),
with a double checkpoint — `expected_scaled` (34 f32) and `expected_proba` —
and tolerance declared in the file itself. Satisfies INV-1 and INV-2.

Pending: Rust test over the 100 scaled samples in
`parity_sample_edge_iot.json` (gap 3).

**Sensitive golden:** second fixture, selected by argmax p(1−p) across the 100 samples
(sample_index=6, p=47.4406%, sensitivity 0.2493 — 99.7% of the theoretical
maximum). With `proba_abs=1e-6`, it tolerates a logit error of **4.01e-6** vs
the typical's **6.65e-4**: **166× stricter**. The fixture declares its own
floor (`sensitivity_floor=0.15`) and its own tolerance; the Rust test reads
them from there, not from constants. Both goldens carry an embedded
`run_fingerprint` and the test verifies they belong to the same run.

Row 6 falls within the 10 rows already certified by Checkpoint 3
(`Réplica≡sklearn`, `max|Δ|=0.000e+00`): the sensitive golden inherits that
equivalence with no additional code.

---

## 🛡️ Methodological Integrity and Transparency (Anti-Gaming)

In Machine Learning it is common to accidentally inflate metrics through *data leakage*, *p-hacking*, or by manipulating the execution environment. This project enforces strict rules to ensure results are honest, reproducible, and production-ready:

| Practice | Implementation |
|---|---|
| **Zero Data Leakage** | The split happens before imputation, scaling, and SMOTE. The test set fits no transformations. **Caveat:** SMOTE precedes CV, so validation AUC is optimistic; the publishable number is `test_auc` |
| **Honest Trade-offs** | FN 5 vs FP 108 at threshold 0.22 is declared, as is `best_round` sitting at the CV ceiling. The cost of the F2 objective is not hidden |
| **Threshold with a receipt** | The threshold (0.22) was defined via F2 on an internal pre-SMOTE split and lives in `model_config.joblib`. Its justification is tracked in `metrics_edge_iot.json`. No test hardcodes it |
| **No magic `RUSTFLAGS`** | Standard profiles (`--release`). No `target-cpu=native` or hidden flags |
| **Exact Parity** | XGBoost's `f32` arithmetic deliberately replicated. RAW→proba: `1.81e-11`. Contract replica ≡ sklearn pipeline: `0.000e+00` |
| **Declared curation** | The curated set **is not canonical BankChurners**: 131 fewer rows, prevalence 0.16066 → 0.16196. The mutated raw is kept immutable as a control group |

> 📌 **Note on Performance Benchmarks:** Precision, weight, and mathematical parity optimizations are certified. Current latency metrics (`< 1ms`) are empirical, measured directly in the browser. The project **has not yet implemented a formal throughput benchmark suite with `criterion.rs`** in native Rust, to avoid browser-environment measurement biases (JIT, GC, fetch). Adding `criterion` for isolated benchmarks is the next step in the MLOps roadmap.

---

## 🦀 Rust / WASM Engine (Python Inference Mirror)

The engine **uses no external Machine Learning libraries**: it implements a
replica of Python's preprocessing pipeline and inference logic. RAW→proba
parity certified at **1.81e-11** (`tests/parity_raw_iot.rs`).

The Rust engine is responsible for:

1. **Loading the XGBoost model from JSON:** tree parsing with parallel arrays
   (`left_children`, `right_children`, `split_indices`, `split_conditions`,
   `base_weights`, `sum_hessian`). There are no node objects and no `nodeid` field.
2. **Loading the inference contract:** `inference_contract_edge_iot.json` (v2).
3. **Replicating RAW preprocessing:**
   - Numeric imputation `strategy="mean"` and categorical `most_frequent`,
     replicating the `SimpleImputer`s fitted on the Python train set.
   - Ordinal encoding of income/education with `is_unknown` flags.
   - One-Hot in the exact order of `get_feature_names_out()`.
   - `StandardScaler` with the mean and scale from train.
4. **Running inference with `f32` arithmetic:** deliberate replica of XGBoost's
   internal arithmetic (using `f64` introduced divergences on splits close to
   the threshold).
5. **Computing the final probability:** sigmoid over the accumulated margin
   plus the model's `base_score`, compared against the threshold **declared by
   the contract**.

### 🏗️ Engineering decisions — declared scope

| Decision | Actual scope |
|---|---|
| **Zero-copy keys** | Contract keys are borrowed (`Cow`/`&str`). **`predict_churn` parses a JSON DOM per call**: p50 is dominated by that allocation, not by the traversal |
| **`OnceLock` cache** | Model and contract are parsed once in `init_engine`. Inference does not re-parse them |
| **Allocation-free traversal** | True for the tree walk. Not for the JSON boundary |
| **Privacy firewall** | Rejects forbidden fields (`pan`, `cvv`, GPS…) before imputing |
| **Consent gate** | `lib.rs:66` refuses to infer without consent when the contract requires it |
| **Parsing with `serde`** | Nodes are modeled as `enum` (`Leaf`/`Split`) to eliminate invalid states at compile time |

> ⚠️ **Declared gap:** the crate validates neither the contract version nor
> `model_sha256` (`sha2` absent from `Cargo.toml`). Four contract fields are
> optional and default to off. See "Current status".

### 💡 Case study: why `f32` and not `f64`

XGBoost evaluates splits and accumulates margins using 32-bit precision internally.

Initially, the Rust engine computed everything in `f64` (double precision), Rust's default for math. However, this broke parity in edge cases where a value fell extremely close to a split threshold.

**The problem:** rounding differences between `f32` and `f64` caused Rust to take different branches in the decision tree, accumulating errors that resulted in divergences greater than `1e-06`.

**The solution:** deliberately replicate XGBoost's internal arithmetic:

```text
f64 features → f32 features
split comparisons in f32
margin accumulation in f32
final margin converted to f64 for sigmoid
```

This detail was key to achieving parity. Historical measurement from the Lite
lineage: `max diff ≈ 7.31e-09`. The current IoT lineage reaches **1.81e-11**.

Lesson learned: in ML inference systems, numerical precision is not just about mathematical exactness, but about replicating the original framework's behavior. What is "more precise" (f64) can be wrong if the model was trained with f32.

### 📁 Main modules

```text
edge/churn-wasm/src/
    lib.rs             Entry point exposed to JavaScript (wasm-bindgen)
    contract.rs        Inference contract deserialization and validation
    preprocessing.rs   Exact replica of the raw → scaled features pipeline
    model.rs           Typed parser for the XGBoost JSON tree
    inference.rs       Tree traversal and margin calculation engine (f32)
```

---

## 🌐 Browser demo

The demo allows you to:

- Load the WASM model locally.
- Load the edge JSON model.
- Load the edge contract.
- Enter client features.
- Compute churn probability without a backend.
- Verify parity automatically.

### Prediction example

```text
Probability: 0.15%
Threshold:   0.22
Decision:    NO CHURN RISK
```

---

## 📦 Development Environment (DevContainer)

This project is configured with **Dev Containers** (Microsoft/GitHub standard) to guarantee 100% reproducibility and avoid local dependency conflicts. You do not need to manually install Python, Rust, `uv`, or `wasm-pack` on your host OS.

### 🌩️ Option A: GitHub Codespaces (Zero installation)
1. Go to the repository on GitHub.
2. Click the green **`<> Code`** button.
3. Select the **Codespaces** tab and click **"Create codespace on main"**.
4. In under 2 minutes, you will have a full VS Code environment in your browser with Python 3.10, Rust, `uv`, and `wasm-pack` preconfigured and ready to use.

### 💻 Option B: Local VS Code
1. Clone the repository and open it in **Visual Studio Code**.
2. Make sure you have the official **Dev Containers** extension by Microsoft installed.
3. VS Code will detect the `.devcontainer/devcontainer.json` file and show a notification in the bottom right corner.
4. Click **"Reopen in Container"**.
5. The isolated environment will build automatically. Once finished, the integrated terminal will already be inside the container with all tools ready.

---

## 🚀 Quickstart

Requirements: Python **3.10.11** (pinned by `uv.lock`), `uv`, Rust stable, `wasm-pack`.

```bash
# 1. Fail-closed environment
uv sync --frozen

# 2. Production chain (3 scripts + 1 blocking gate)
uv run python run_all.py
#    generate_synthetic_iot → z gate → train_edge_iot → export_edge_iot

# 3. Verification gates
uv run python scripts/check_artifact_hashes.py     # 13 checks
uv run python scripts/check_contract_posture.py    # 4 armed controls
uv run pytest tests/ -v                            # contract + z distribution

# 4. Rust tests
cd edge/churn-wasm && cargo test -- --nocapture

# 5. Build WASM
wasm-pack build --target web --release

# 6. Local demo
cd edge && python -m http.server 8000
```

**Outside the default chain, on purpose:** `curate_raw.py` and
`calibrate_iot_reference.py` rewrite tracked evidence. A pipeline that
regenerates its own control group has no control group. They are invoked
explicitly and their result is verified by hash.

---

## 📦 Project structure

```text
churn-edge-ai/
├── .github/workflows/
│   ├── ci.yml                          # python-gates + rust-wasm
│   └── deploy-pages.yml                # uploads ./edge in full
├── .gitignore
├── pyproject.toml
├── uv.lock                             # fail-closed pin
├── run_all.py                          # 3 scripts + z gate
├── README.md · README.en.md
│
├── data/
│   ├── raw/Credit_card_churn.csv       # immutable origin, control group
│   ├── curated/                        # SSOT + manifests + quarantine
│   └── synthetic/iot_features_metadata.json
│   # data/processed/ ignored: regenerable derivative
│
├── docs/
│   ├── adr/0004 · 0005 · 0006
│   └── evidence/2026-11-audit-raw.{json,log}
│
├── model_artifacts/                    # fully ignored
│   └── edge_iot/                       # binaries + metrics + provenance
│
├── edge/
│   ├── xgb_model_edge_iot.json         # 34 features, 200 trees
│   ├── inference_contract_edge_iot.json# v2, byte-deterministic
│   ├── metrics_edge_iot.json           # threshold justification
│   ├── export_manifest_edge_iot.json   # 8 hashes + provenance (run_fingerprint, created_at)
│   ├── parity_sample_edge_iot.json     # 100 × 34 f32
│   ├── raw_parity_case_iot.json        # real golden case, double checkpoint
│   ├── raw_parity_case_iot_sensitive.json # maximum-sensitivity golden (k=6)
│   ├── index.html · demo.js
│   └── churn-wasm/
│       ├── Cargo.toml · Cargo.lock
│       ├── pkg/                        # WASM tracked for Pages
│       ├── src/{lib,contract,preprocessing,model,inference}.rs
│       └── tests/{contract_live,parity_raw_iot}.rs
│
├── mock_bank/
├── src/
│   ├── config.py                       # no constants from retired lineages
│   ├── data_contract.py                # ingestion contract v1.1.0
│   ├── data_loading.py · preprocessing.py · train.py · synthetic_iot.py
│
├── scripts/
│   ├── curate_raw.py · calibrate_iot_reference.py
│   ├── generate_synthetic_iot.py
│   ├── train_edge_iot_model.py · export_edge_iot.py
│   ├── check_artifact_hashes.py · check_contract_posture.py
│
└── tests/
    ├── test_data_contract.py           # rejects the raw, accepts the curated
    └── test_iot_z_distribution.py      # the symptom has a ceiling
```

---

## 📄 Inference contract

The `inference_contract_edge_iot.json` contract (v2) includes:

```text
feature_order · n_features
numeric_imputer · categorical_imputer
ordinal mappings + is_unknown flags
one-hot categories (get_feature_names_out order)
scaler mean/scale
best_threshold
xgboost base_score
iot_features · iot_bounds
forbidden_fields · consent_required
model_sha256
```

This prevents classic deployment pitfalls: unordered features, mishandled new
categories, inconsistent scaling, imputation differing between train and
serving, wrong threshold, misinterpreted base score.

The contract is **byte-deterministic**: `created_at` lives in
`export_manifest_edge_iot.json`, not here. Two consecutive exports produce the
same `sha256`.

> ⚠️ `model_sha256` travels in the contract; the engine does not verify it yet.
> Declared gap, with a closure plan in the B → C → D sequence.

---

## 🧰 Main commands

| Command | Description |
|---|---|
| `uv run python run_all.py` | Full Edge IoT chain |
| `uv run python scripts/curate_raw.py` | Forensic curation R1–R5 (explicit) |
| `uv run python scripts/calibrate_iot_reference.py` | Recalibrates IOT_REFERENCE on train (explicit) |
| `uv run python scripts/check_artifact_hashes.py` | Read-only integrity gate (13 checks) |
| `uv run python scripts/check_contract_posture.py` | Posture gate (4 armed controls) |
| `uv run pytest tests/ -v -s` | Data contract + z distribution |
| `cargo test` | Rust tests: live contract + IoT parity + firewall |
| `wasm-pack build --target web --release` | Builds WASM |
| `python -m http.server 8000` | Serves local demo |

---

## ✅ Tests

### Python

```bash
uv run pytest tests/ -v -s
```

| Test | What it certifies |
| --- | --- |
| `test_data_contract.py::test_gate_rejects_mutated_origin` | The mutated origin fails with the 4 literals. **It is the audit's control group** |
| `test_data_contract.py::test_gate_accepts_curated_ssot` | Positive control: without it, a `validate()` that rejects everything would pass the previous test |
| `test_iot_z_distribution.py` | Prints the per-feature z table and enforces a ceiling. The symptom cannot return |

Plus two read-only gates, invoked by CI before installing dependencies:

```bash
python scripts/check_artifact_hashes.py     # 13 checks: integrity · agreement · coupling
python scripts/check_contract_posture.py    # 4 controls present AND armed
```

### Rust

```bash
cd edge/churn-wasm && cargo test -- --nocapture
```

12 tests: 7 unit tests (`contract`, `inference`, `lib`) + `contract_live` (1) +
`parity_raw_iot` (4: end-to-end parity, privacy firewall, silent clipping characterization, sensitive golden).

Expected output of the IoT parity test (verbatim):

```text
🐍 Python (expected): 0.00150453
🦀 Rust   (got):      0.00150453
📉 Diff:              1.81e-11
🏁 Resultado:         ✅ PASS
```

The three consistency asserts (`threshold`, `n_features`, `n_trees`) **read
from the deployed artifacts**, not from test constants.

---

## 🔄 CI/CD

`on: [push, pull_request]` — coverage on all branches.

**Job `python-gates`:** integrity and posture with system Python (pure stdlib,
fail in 2 s), then `uv sync --frozen`, data contract gate, telemetry
regeneration, and z distribution gate with the table printed in the log.

**Job `rust-wasm`:** `cargo test` (all binaries) + `wasm-pack build`.

```yaml
name: CI

on: [push, pull_request]

jobs:
  python-gates:
    name: Python gates (integridad · postura · contrato · IoT z)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Gate de integridad (read-only, 13 verificaciones)
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

The legacy `python-parity` job was retired in ADR-0006 along with the Phase 1
scripts it invoked. It used Python 3.11 with unpinned `pip install` while the
rest of the repo honored `uv.lock` (3.10.11): two interpreters, one lockfile.

---

## 🧾 Generated artifacts

### Local, git-ignored (regenerable derivatives)

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

### Tracked — source of truth for Rust and for the auditor

```text
edge/
    xgb_model_edge_iot.json           34 features · 200 trees
    inference_contract_edge_iot.json  v2 · byte-deterministic
    metrics_edge_iot.json             AUC, matrix, F2 criterion, caveats
    export_manifest_edge_iot.json     8 hashes + provenance (run_fingerprint, created_at)
    parity_sample_edge_iot.json       100 × 34 f32
    raw_parity_case_iot.json          real golden case · double checkpoint
    raw_parity_case_iot_sensitive.json sensitive golden · p=47.44% · sens=0.2493

data/curated/                         SSOT + manifests + quarantine
data/synthetic/iot_features_metadata.json
```

**Criterion:** a tracked artifact that depends on ignored inputs is orphaned
evidence. The `.joblib` files are rebuilt; the JSON files in `edge/` and the
manifests are the official record. See ADR-0004.

---

## 💼 Business impact

| Area | Impact |
|---|---|
| Infrastructure | Reduces server-side inference cost |
| Privacy | Raw data never leaves the device |
| Latency | Immediate local prediction |
| Retention | Early detection of churn risk |
| Product | Interactive experience without a backend |
| Compliance | Favors data minimization |
| Costs | Fewer dependencies on APIs or cloud services |

---

## 🔐 Security and considerations

This project follows privacy-by-design principles, but there are important considerations:

- The model is distributed to the client.
- The model should not be considered a secret if published in the browser.
- Local inference protects user data, not necessarily the model's IP.
- It should not be used as the sole anti-fraud mechanism.
- If deployed to production, it is recommended to:
  - Sign artifacts.
  - Use Subresource Integrity for WASM/JS.
  - Version models.
  - Register contract hashes.
  - Monitor drift.
  - Maintain an audit trail of changes.

---

## 📡 Production and monitoring

To take this system to production, the following is recommended:

### Data monitoring

- Numeric feature drift.
- Categorical drift.
- Shifts in prediction distribution.
- Changes in actual churn rate.

### Model monitoring

- AUC/F2 over rolling windows.
- Threshold stability.
- Delta between prediction and actual outcome.

### MLOps

- Model versioning.
- Artifact hashing.
- Inference contract registry.
- Parity tests in CI.
- Reproducible training with fixed seeds.
- Canary deployments for new edge models.

---

## 🐞 Troubleshooting

### Parity fails by a tiny margin

Verify that Rust is using `f32` arithmetic for:

- Split comparisons.
- Margin accumulation.
- Tree evaluation.

### `UnicodeEncodeError` on Windows

If a script writes JSON with Unicode characters, always use:

```python
encoding="utf-8"
```

Example:

```python
out_path.write_text(payload, encoding="utf-8")
```

Or run with:

```bash
PYTHONUTF8=1
```

### The demo doesn't load WASM

Verify these exist:

```text
edge/churn-wasm/pkg/churn_wasm.js
edge/churn-wasm/pkg/churn_wasm_bg.wasm
```

Both must exist after:

```bash
wasm-pack build --target web --release
```

### The browser caches the WASM

Perform a hard reload:

```text
Ctrl + Shift + R
```

or clear the cache.

---

## 🗺️ Project evolution

This repository documents the construction of a deterministic edge inference engine. What began as a traditional banking churn model (Milestone 1) has evolved into a full IoT SDK with explicit privacy contracts for the financial industry.

### 🏁 Milestone 1 — Deterministic engine (retired, archived)
The Full 28f baseline and the Edge Lite 12f were retired in ADR-0006. They
served no traffic and descended from the mutated file. Recoverable from tag
`baseline-fase3`. Their historical parity (7.31e-09) remains as an archival
measurement.

### 🌱 Milestone 2 — Forensic curation and SSOT (completed ✅)
- [x] Fail-closed ingestion contract: dtype, units, algebraic identity
- [x] R1–R5 curation with domain preconditions re-verified on every run
- [x] MNAR quarantine without imputation (131 rows, z = −3.10)
- [x] Chain of custody: hashes of origin, script, contract, and output
- [x] IoT generator with support gate over the fixed reference

### 🛡️ Milestone 3 — Rust core (partial 🚧)
- [x] Privacy firewall: rejection of forbidden fields
- [x] Consent gate (`lib.rs:66`)
- [x] `init_engine` with `OnceLock`
- [x] RAW→proba parity with double checkpoint (1.81e-11)
- [x] The deployed contract deserializes in Rust (`contract_live.rs`)
- [ ] **SHA-256 integrity verification in the engine** — `sha2` is not in
      `Cargo.toml`. The hash travels in the contract; the engine does not check it yet
- [ ] **Clipping with a flag** — today it clips silently. An impossible input
      (`app_open = 9999`) produces a plausible prediction with no marker
- [ ] **Mandatory controls in the schema** — four `#[serde(default)]`
      attributes leave `forbidden_fields`, `iot_bounds`, `iot_features`, and
      `consent_required` off by default
- [ ] Rust test over the scaled samples
- [ ] Cross-platform compilation (Android `.so`, iOS `.a`)
- [ ] Formal benchmarking with `criterion.rs`

### 🔌 Milestone 4 — B2B ecosystem (partial 🚧)
- [x] Mock Bank API with consent management
- [x] Offline-first mode (embedded fallback + localStorage)
- [ ] Web SDK (NPM + Web Workers)
- [ ] Mobile wrappers

### 💼 Milestone 5 — Banking UX (partial 🚧)

- [x] Dual interface: Operator View + Client View
- [ ] **Retention policy outside the engine** — `offer_retention` is
      compiled into WASM (`lib.rs:84`). The business decision belongs to the
      application layer, not to the inference binary
- [ ] Explicit, visible, revocable consent — not disguised as a benefits banner
- [x] **Dynamic threshold from the contract** — the frontend reads
      `best_threshold` from `inference_contract_edge_iot.json`; no test or
      component hardcodes it. Evidence: "Business threshold" section and `edge/demo.js`

---

## 🎯 Current status

**Verified and in production:**
- Curated SSOT with fail-closed contract and hash-based chain of custody
- Rust/WASM engine with RAW→proba parity of 1.81e-11
- 12 integrity checks + 4 posture checks in CI
- 100% client-side inference, 0 bytes to the server
- Dual demo on GitHub Pages with offline fallback

**Known, declared gaps:**
1. The engine does not verify the model's SHA-256 (`sha2` absent from the crate)
2. Four contract controls are optional with a permissive default
3. IoT clipping emits no flag: impossible input → plausible prediction
4. The 14 **numeric** banking columns have no bounds at inference time
5. `best_round` is at the CV ceiling; early stopping is inert on a monotonic
post-SMOTE curve. **Declared, not closed:** `metrics_edge_iot.json` carries
`best_round_at_cap=true`, `cv_num_boost_round_cap=200`, and
`early_stopping_caveat`. Raising the ceiling does not find an optimum — it
moves the boundary. The real fix (intra-fold SMOTE) goes on a separate branch.

6. ~~The golden lives at p ≈ 0.15%: it tolerates a logit error ~166× larger
than one in the middle zone.~~ **Closed:** second golden at `p=47.4406%`
(sensitivity `0.2493`), logit tolerance `4.01e-6`, with its own Rust test
(`iot_sensitive_golden_parity`). The typical is kept as the common case.
7. SMOTE is applied before CV: validation AUC is optimistic.
   **The publishable number is the test-set one**
8. The retention policy is in the binary, not in the application

Gaps 1–4 have a closure plan in the B → C → D sequence.

---

## 📚 Tech stack

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
Vanilla JavaScript
GitHub Actions
uv
Pytest
```

Reproducibility:

```text
SEED = 44
```

---

## 📌 Final summary

**Churn Edge AI** turns a traditional banking churn model into a lightweight, private, zero-marginal-cost edge inference system.

It combines:

```text
Python for training
XGBoost for modeling
JSON as a portable format
Rust for secure, fast inference
WebAssembly for browser execution
Certified parity via testing
```

It is a relevant use case for:

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
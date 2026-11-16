"""
Mide IOT_REFERENCE sobre el split de TRAIN del curated. Congela con procedencia.
Salida: bloque para src/config.py + data/curated/iot_reference_provenance.json
"""
import hashlib, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import PROJECT_ROOT, SEED, TEST_SIZE, TARGET, IOT_REFERENCE

CUR = PROJECT_ROOT / "data" / "curated" / "churn_curated.csv"
COLS = list(IOT_REFERENCE.keys())

df = pd.read_csv(CUR)
y = df[TARGET].map({"Attrited Customer": 1, "Existing Customer": 0})
idx_tr, _ = train_test_split(df.index, test_size=TEST_SIZE, stratify=y,
                             random_state=SEED)
tr = df.loc[idx_tr, COLS]

new = {c: (round(float(tr[c].mean()), 4), round(float(tr[c].std(ddof=1)), 4))
       for c in COLS}

print("# IOT_REFERENCE — medido sobre TRAIN del curated, ddof=1")
print("IOT_REFERENCE = {")
for c, (m, s) in new.items():
    old = IOT_REFERENCE[c]
    print(f'    "{c}": ({m}, {s}),   # antes {old}')
print("}")

prov = {
    "measured_on": "train split del curated",
    "curated_sha256": hashlib.sha256(CUR.read_bytes()).hexdigest(),
    "n_train": len(tr), "seed": SEED, "test_size": TEST_SIZE,
    "stratified_by": TARGET, "std_ddof": 1,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "previous_values": {k: list(v) for k, v in IOT_REFERENCE.items()},
    "new_values": {k: list(v) for k, v in new.items()},
    "defect_corrected": "Total_Trans_Ct usaba (140,58): el máximo, no la media. "
                        "Independiente de la corrupción del origen.",
}
(CUR.parent / "iot_reference_provenance.json").write_text(
    json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
# ADR-0005 — El síntoma estaba registrado y normalizado

## Hallazgo
`src/config.py` contenía:
    NUMERIC_LIKE_COLUMNS = ["Total_Amt_Chng_Q4_Q1", "Total_Ct_Chng_Q4_Q1"]

Exactamente las dos columnas que llegaban como `object` — porque contenían
fechas 2024 donde debía ir un ratio. El pipeline ya había encontrado el síntoma
y lo resolvió con `pd.to_numeric(errors="coerce")` + imputación por la media.

## Por qué era un disfraz
Tres lavados silenciosos en serie: `coerce` convierte la fecha en NaN →
`fillna(mean)` la borra → los supervivientes ×1000 producen z de miles → el
clip del generador los convierte en un bound. Cada capa transformó una
violación de contrato en un valor plausible, sin dejar un flag.

`Avg_Utilization_Ratio` no disparó la lista porque sus valores mutados siguen
siendo numéricos (61.0, 0.76). Su corrupción de escala pasó invisible.

## Decisión
Eliminar `NUMERIC_LIKE_COLUMNS` y las dos funciones de coerción. El contrato de
ingestión (`src/data_contract.py`) garantiza dtype con `errors="raise"`: si el
dato no parsea, se rechaza. Una coerción posterior es redundante si el gate
funciona, y un disfraz si no.

## Nivel arquitectónico
El síntoma se había convertido en default. La capa sintomática dejó un registro
escrito en la configuración y nadie bajó a la causa.
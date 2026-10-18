// demo.js
// Carga el módulo WASM, el modelo y el contrato.
// Ejecuta la predicción cuando el usuario hace clic.

import init, { predict_churn, predict_from_scaled_features, verify_parity }
    from './churn-wasm/pkg/churn_wasm.js';

let wasmReady = false;
let modelJson = null;
let contractJson = null;

// ─── Helper para leer números aceptando coma decimal ───
function readNumber(id) {
  const raw = document.getElementById(id).value.trim();

  if (raw === '') {
    return null;
  }

  // Reemplaza coma decimal por punto decimal
  const normalized = raw.replace(',', '.');

  const parsed = Number(normalized);

  return Number.isFinite(parsed) ? parsed : null;
}

// ─── Inicialización ───
async function initialize() {
    try {
        // Cargar WASM
        await init();
        console.log('✅ WASM cargado');

        // Cargar modelo edge
        const modelResp = await fetch('xgb_model_edge.json');
        modelJson = await modelResp.text();
        console.log('✅ Modelo edge cargado:', modelJson.length, 'bytes');

        // Cargar contrato edge
        const contractResp = await fetch('inference_contract_edge.json');
        contractJson = await contractResp.text();
        console.log('✅ Contrato edge cargado:', contractJson.length, 'bytes');

        wasmReady = true;
        document.getElementById('status').textContent = '🟢 Motor WASM listo';
        document.getElementById('status').style.color = '#22c55e';

        // Ejecutar paridad automáticamente
        await runParityCheck();

    } catch (err) {
        console.error('❌ Error de inicialización:', err);
        document.getElementById('status').textContent = '🔴 Error: ' + err.message;
        document.getElementById('status').style.color = '#ef4444';
    }
}

// ─── Predicción desde formulario ───
function handlePredict() {
    if (!wasmReady) {
        alert('El motor WASM aún no está listo');
        return;
    }

    // Leer valores del formulario
    const input = {
        Customer_Age: readNumber('Customer_Age'),
        Gender: document.getElementById('Gender').value,
        Dependent_count: readNumber('Dependent_count'),
        Education_Level: document.getElementById('Education_Level').value,
        Marital_Status: document.getElementById('Marital_Status').value,
        Income_Category: document.getElementById('Income_Category').value,
        Card_Category: document.getElementById('Card_Category').value,
        Months_Inactive_12_mon: readNumber('Months_Inactive_12_mon'),
        Contacts_Count_12_mon: readNumber('Contacts_Count_12_mon'),
        Total_Revolving_Bal: readNumber('Total_Revolving_Bal'),
        Total_Amt_Chng_Q4_Q1: readNumber('Total_Amt_Chng_Q4_Q1'),
        Total_Trans_Amt: readNumber('Total_Trans_Amt'),
        Total_Trans_Ct: readNumber('Total_Trans_Ct'),
        Total_Ct_Chng_Q4_Q1: readNumber('Total_Ct_Chng_Q4_Q1'),
        Avg_Utilization_Ratio: readNumber('Avg_Utilization_Ratio'),
        Months_on_book: readNumber('Months_on_book'),
        Total_Relationship_Count: readNumber('Total_Relationship_Count'),
        Credit_Limit: readNumber('Credit_Limit'),
        Avg_Open_To_Buy: readNumber('Avg_Open_To_Buy'),
    };

    const inputJson = JSON.stringify(input);

    // Ejecutar predicción en WASM
    const startTime = performance.now();
    const resultJson = predict_churn(inputJson, modelJson, contractJson);
    const endTime = performance.now();

    const result = JSON.parse(resultJson);

    if (result.error) {
        document.getElementById('result').innerHTML =
            `<span style="color: #ef4444">❌ ${result.error}</span>`;
        return;
    }

    // Mostrar resultado
    const churnColor = result.is_churn ? '#ef4444' : '#22c55e';
    const churnLabel = result.is_churn ? '🔴 ALTO RIESGO DE CHURN' : '🟢 Cliente estable';

    document.getElementById('result').innerHTML = `
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 2em; margin-bottom: 10px;">${churnLabel}</div>
            <div style="font-size: 3em; font-weight: bold; color: ${churnColor};">
                ${result.probability_pct.toFixed(2)}%
            </div>
            <div style="color: #888; margin-top: 10px;">
                Probabilidad de abandono
            </div>
            <div style="color: #666; margin-top: 5px; font-size: 0.9em;">
                Margin: ${result.margin.toFixed(6)} |
                Threshold: ${result.threshold} |
                ⏱️ ${(endTime - startTime).toFixed(2)}ms
            </div>
        </div>
    `;
}

// ─── Verificación de paridad ───
async function runParityCheck() {
    try {
        const parityResp = await fetch('parity_sample_edge.json');
        const parity = await parityResp.json();

        // Tomar las primeras 5 muestras
        const nCheck = Math.min(5, parity.features.length);
        let allPassed = true;
        let results = [];

        for (let i = 0; i < nCheck; i++) {
            const featuresJson = JSON.stringify(parity.features[i]);
            const expectedProba = parity.expected_proba[i];

            const resultJson = verify_parity(
                featuresJson,
                expectedProba,
                modelJson,
                1e-4
            );
            const result = JSON.parse(resultJson);

            if (!result.passed) allPassed = false;
            results.push(result);
        }

        const parityDiv = document.getElementById('parity');
        if (allPassed) {
            parityDiv.innerHTML = `
                <span style="color: #22c55e;">
                    ✅ Paridad Python ↔ Rust verificada (${nCheck} muestras)
                </span>
                <br><small>Max diff: ${Math.max(...results.map(r => r.diff)).toExponential(2)}</small>
            `;
        } else {
            parityDiv.innerHTML = `
                <span style="color: #ef4444;">
                    ❌ Paridad FALLÓ
                </span>
            `;
            console.error('Parity results:', results);
        }

    } catch (err) {
        console.error('Error en parity check:', err);
        document.getElementById('parity').innerHTML =
            `<span style="color: #f59e0b;">⚠️ No se pudo verificar paridad: ${err.message}</span>`;
    }
}

// ─── Event listeners ───
document.getElementById('predictBtn').addEventListener('click', handlePredict);

// Inicializar al cargar
initialize();
// demo.js
// Carga el módulo WASM, el modelo IoT y el contrato enriquecido.
// Ejecuta la predicción con los 25 inputs (19 bank + 6 IoT) + consentimiento.

import init, { predict_churn, verify_parity } from './churn-wasm/pkg/churn_wasm.js';

let wasmReady = false;
let modelJson = null;
let contractJson = null;

// ─── Helper para leer números aceptando coma decimal ───
function readNumber(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    const raw = el.value.trim();
    if (raw === '') return null;
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

        // Cargar modelo IoT (34 features)
        const modelResp = await fetch('xgb_model_edge_iot.json');
        modelJson = await modelResp.text();
        console.log('✅ Modelo IoT cargado:', modelJson.length, 'bytes');

        // Cargar contrato IoT enriquecido
        const contractResp = await fetch('inference_contract_edge_iot.json');
        contractJson = await contractResp.text();
        console.log('✅ Contrato IoT cargado:', contractJson.length, 'bytes');

        wasmReady = true;
        document.getElementById('status').textContent = 
            '🟢 Motor WASM IoT listo (34 features)';
        document.getElementById('status').style.color = '#22c55e';

        // Ejecutar paridad automáticamente
        await runParityCheck();
    } catch (err) {
        console.error('❌ Error de inicialización:', err);
        document.getElementById('status').textContent = '🔴 Error: ' + err.message;
        document.getElementById('status').style.color = '#ef4444';
    }
}

// ─── Predicción desde formulario (25 inputs + consentimiento) ───
function handlePredict() {
    if (!wasmReady) {
        alert('El motor WASM aún no está listo');
        return;
    }

    // Leer consentimiento
    const consent = document.getElementById('consentCheckbox').checked;

    // Leer TODOS los inputs (19 bancarios + 6 IoT)
    const input = {
        // ─── Datos bancarios (19) ───
        Customer_Age: readNumber('Customer_Age'),
        Gender: document.getElementById('Gender').value,
        Dependent_count: readNumber('Dependent_count'),
        Education_Level: document.getElementById('Education_Level').value,
        Marital_Status: document.getElementById('Marital_Status').value,
        Income_Category: document.getElementById('Income_Category').value,
        Card_Category: document.getElementById('Card_Category').value,
        Months_on_book: readNumber('Months_on_book'),
        Total_Relationship_Count: readNumber('Total_Relationship_Count'),
        Months_Inactive_12_mon: readNumber('Months_Inactive_12_mon'),
        Contacts_Count_12_mon: readNumber('Contacts_Count_12_mon'),
        Credit_Limit: readNumber('Credit_Limit'),
        Total_Revolving_Bal: readNumber('Total_Revolving_Bal'),
        Avg_Open_To_Buy: readNumber('Avg_Open_To_Buy'),
        Total_Amt_Chng_Q4_Q1: readNumber('Total_Amt_Chng_Q4_Q1'),
        Total_Trans_Amt: readNumber('Total_Trans_Amt'),
        Total_Trans_Ct: readNumber('Total_Trans_Ct'),
        Total_Ct_Chng_Q4_Q1: readNumber('Total_Ct_Chng_Q4_Q1'),
        Avg_Utilization_Ratio: readNumber('Avg_Utilization_Ratio'),
        // ─── Datos IoT (6) ───
        app_open_count_30d: readNumber('app_open_count_30d'),
        contactless_usage_count_30d: readNumber('contactless_usage_count_30d'),
        payment_flow_abandon_count_30d: readNumber('payment_flow_abandon_count_30d'),
        nfc_failed_attempts_30d: readNumber('nfc_failed_attempts_30d'),
        app_crash_count_30d: readNumber('app_crash_count_30d'),
        pos_latency_p95_ms: readNumber('pos_latency_p95_ms'),
    };

    const inputJson = JSON.stringify(input);

    // Ejecutar predicción en WASM con consentimiento
    const startTime = performance.now();
    const resultJson = predict_churn(inputJson, modelJson, contractJson, consent);
    const endTime = performance.now();
    const result = JSON.parse(resultJson);

    // Manejar errores (incluye "Consent Required")
    if (result.error) {
        document.getElementById('result').innerHTML = `
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 2em; color: #f59e0b; margin-bottom: 10px;">
                    ⚠️ ${result.error}
                </div>
                <div style="color: #94a3b8; font-size: 0.9em;">
                    Marca el checkbox de consentimiento para usar datos IoT.
                </div>
            </div>
        `;
        return;
    }

    // ─── Renderizar resultado con regla de negocio ───
    const churnColor = result.is_churn ? '#ef4444' : '#22c55e';
    const churnLabel = result.is_churn 
        ? '🔴 ALTO RIESGO DE CHURN' 
        : '🟢 Cliente estable';
    
    // Tarjeta de cashback si aplica la regla de negocio
    const cashbackHtml = result.offer_retention ? `
        <div class="cashback-card">
            <h3>🎁 Oferta de Retención Activada</h3>
            <p>
                Cliente con alta probabilidad de abandono y consentimiento otorgado.
                <br><strong>Cashback sugerido: $50 USD</strong> para retener.
            </p>
        </div>
    ` : '';

    // Badge de consentimiento
    const consentBadge = result.consent_granted
        ? '<span style="color: #22c55e;">🔓 IoT activado</span>'
        : '<span style="color: #f59e0b;">🔒 IoT desactivado</span>';

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
                Features: ${result.n_features} |
                ${consentBadge} |
                ⏱️ ${(endTime - startTime).toFixed(2)}ms
            </div>
            ${cashbackHtml}
        </div>
    `;
}

// ─── Verificación de paridad (usa el parity sample IoT) ───
async function runParityCheck() {
    try {
        const parityResp = await fetch('parity_sample_edge_iot.json');
        if (!parityResp.ok) {
            document.getElementById('parity').innerHTML = 
                '<span style="color: #f59e0b;">⚠️ Parity sample IoT no encontrado</span>';
            return;
        }
        const parity = await parityResp.json();

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
                    ✅ Paridad Python ↔ Rust verificada (${nCheck} muestras IoT, 
                    ${parity.n_features} features)
                </span>
                <br><small>Max diff: ${Math.max(...results.map(r => r.diff)).toExponential(2)}</small>
            `;
        } else {
            parityDiv.innerHTML = `
                <span style="color: #ef4444;">❌ Paridad FALLÓ</span>
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
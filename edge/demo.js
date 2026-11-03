// demo.js
// Carga el módulo WASM, el modelo IoT y el contrato enriquecido.
// Consume datos del Mock Bank API y ejecuta predicción con consentimiento.

import init, { predict_churn, verify_parity } from './churn-wasm/pkg/churn_wasm.js';

// ─── Configuración del Mock Bank ───
const MOCK_BANK_URL = 'http://localhost:8000';
let currentClientId = 'CLIENT_001';

let wasmReady = false;
let modelJson = null;
let contractJson = null;

// ─── Helper: logger visual de llamadas a la API ───
function logApi(method, endpoint, status, detail = '') {
    const logDiv = document.getElementById('apiLog');
    if (!logDiv) return;
    const cls = method === 'GET' ? 'log-get' : 'log-post';
    const statusCls = status === 'ok' ? 'log-ok' : (status === 'err' ? 'log-err' : '');
    const time = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.innerHTML = `<span class="${cls}">[${method}]</span> ${endpoint} → <span class="${statusCls}">${status}</span> <span style="color:#64748b">${time} ${detail}</span>`;
    logDiv.appendChild(line);
    logDiv.scrollTop = logDiv.scrollHeight;
}

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
        // 1. Cargar WASM
        await init();
        console.log('✅ WASM cargado');

        // 2. Cargar modelo IoT (34 features)
        const modelResp = await fetch('xgb_model_edge_iot.json');
        modelJson = await modelResp.text();
        console.log('✅ Modelo IoT cargado:', modelJson.length, 'bytes');

        // 3. Cargar contrato IoT enriquecido
        const contractResp = await fetch('inference_contract_edge_iot.json');
        contractJson = await contractResp.text();
        console.log('✅ Contrato IoT cargado:', contractJson.length, 'bytes');

        wasmReady = true;
        document.getElementById('status').textContent = 
            '🟢 Motor WASM IoT listo (34 features)';
        document.getElementById('status').style.color = '#22c55e';

        // 4. Verificar paridad con muestras pre-escaladas
        await runParityCheck();

        // 5. NUEVO: Cargar datos iniciales desde Mock Bank
        await loadFeaturesFromBank(currentClientId);
        await checkPreviousConsent(currentClientId);

    } catch (err) {
        console.error('❌ Error de inicialización:', err);
        document.getElementById('status').textContent = '🔴 Error: ' + err.message;
        document.getElementById('status').style.color = '#ef4444';
    }
}

// ─── NUEVO: Cargar features desde Mock Bank ───
async function loadFeaturesFromBank(clientId) {
    const btn = document.getElementById('loadClientBtn');
    if (btn) btn.disabled = true;

    try {
        const resp = await fetch(`${MOCK_BANK_URL}/api/features/${clientId}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        
        const data = await resp.json();
        const features = data.features;

        // Rellenar el formulario con los datos del banco
        for (const [key, value] of Object.entries(features)) {
            const el = document.getElementById(key);
            if (el) {
                el.value = value;
            }
        }

        logApi('GET', `/api/features/${clientId}`, 'ok', 
            `— ${Object.keys(features).length} campos cargados`);
        
        // Actualizar tag de perfil de riesgo
        updateProfileTag(clientId);
        
        currentClientId = clientId;
    } catch (err) {
        logApi('GET', `/api/features/${clientId}`, 'err', `— ${err.message}`);
        console.error('Error cargando features del banco:', err);
    } finally {
        if (btn) btn.disabled = false;
    }
}

// ─── NUEVO: Verificar consentimiento previo ───
async function checkPreviousConsent(clientId) {
    try {
        const resp = await fetch(`${MOCK_BANK_URL}/api/consent/${clientId}`);
        if (!resp.ok) {
            logApi('GET', `/api/consent/${clientId}`, 'err', `— HTTP ${resp.status}`);
            return false;
        }
        
        const data = await resp.json();
        const checkbox = document.getElementById('consentCheckbox');
        if (checkbox) {
            checkbox.checked = data.consent === true;
        }

        if (data.consent) {
            logApi('GET', `/api/consent/${clientId}`, 'ok', 
                `— consentimiento previo (${data.granted_at || 'unknown'})`);
        } else {
            logApi('GET', `/api/consent/${clientId}`, 'ok', 
                `— sin consentimiento registrado`);
        }
        return data.consent;
    } catch (err) {
        logApi('GET', `/api/consent/${clientId}`, 'err', `— ${err.message}`);
        return false;
    }
}

// ─── NUEVO: Registrar consentimiento en el banco ───
async function registerConsentInBank(consent, clientId) {
    try {
        const resp = await fetch(`${MOCK_BANK_URL}/api/consent`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                client_id: clientId,
                consent: consent,
            }),
        });

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        
        const data = await resp.json();
        logApi('POST', '/api/consent', 'ok', 
            `— ${consent ? 'otorgado' : 'revocado'} para ${clientId}`);
        return true;
    } catch (err) {
        logApi('POST', '/api/consent', 'err', `— ${err.message}`);
        return false;
    }
}

// ─── NUEVO: Actualizar tag de perfil de riesgo ───
function updateProfileTag(clientId) {
    const tag = document.getElementById('clientProfileTag');
    if (!tag) return;
    
    if (clientId === 'CLIENT_001') {
        tag.className = 'client-profile-tag profile-high-risk';
        tag.textContent = '🔴 Alto riesgo';
    } else if (clientId === 'CLIENT_002') {
        tag.className = 'client-profile-tag profile-low-risk';
        tag.textContent = '🟢 Bajo riesgo';
    } else {
        tag.className = 'client-profile-tag';
        tag.textContent = clientId;
    }
}

// ─── Predicción desde formulario (25 inputs + consentimiento) ───
function handlePredict() {
    if (!wasmReady) {
        alert('El motor WASM aún no está listo');
        return;
    }

    const consent = document.getElementById('consentCheckbox').checked;

    const input = {
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
        app_open_count_30d: readNumber('app_open_count_30d'),
        contactless_usage_count_30d: readNumber('contactless_usage_count_30d'),
        payment_flow_abandon_count_30d: readNumber('payment_flow_abandon_count_30d'),
        nfc_failed_attempts_30d: readNumber('nfc_failed_attempts_30d'),
        app_crash_count_30d: readNumber('app_crash_count_30d'),
        pos_latency_p95_ms: readNumber('pos_latency_p95_ms'),
    };

    const inputJson = JSON.stringify(input);

    const startTime = performance.now();
    const resultJson = predict_churn(inputJson, modelJson, contractJson, consent);
    const endTime = performance.now();
    const result = JSON.parse(resultJson);

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

    const churnColor = result.is_churn ? '#ef4444' : '#22c55e';
    const churnLabel = result.is_churn 
        ? '🔴 ALTO RIESGO DE CHURN' 
        : '🟢 Cliente estable';
    
    const cashbackHtml = result.offer_retention ? `
        <div class="cashback-card">
            <h3>🎁 Oferta de Retención Activada</h3>
            <p>
                Cliente con alta probabilidad de abandono y consentimiento otorgado.
                <br><strong>Cashback sugerido: $50 USD</strong> para retener.
            </p>
        </div>
    ` : '';

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

// NUEVO: Botón "Cargar datos del banco"
document.getElementById('loadClientBtn')?.addEventListener('click', async () => {
    const selectedId = document.getElementById('clientSelect').value;
    currentClientId = selectedId;
    await loadFeaturesFromBank(selectedId);
    await checkPreviousConsent(selectedId);
});

// NUEVO: Cuando cambia el selector, cargar automáticamente
document.getElementById('clientSelect')?.addEventListener('change', async (e) => {
    currentClientId = e.target.value;
    await loadFeaturesFromBank(currentClientId);
    await checkPreviousConsent(currentClientId);
});

// NUEVO: Cuando cambia el checkbox, registrar en el Mock Bank
document.getElementById('consentCheckbox')?.addEventListener('change', async (e) => {
    const consent = e.target.checked;
    await registerConsentInBank(consent, currentClientId);
});

// Inicializar al cargar
initialize();
// demo.js
// Dos vistas:
//  - 👨‍💻 Operador: consola técnica (25 campos, log API, paridad, métricas)
//  - 📱 Cliente: simulador de app bancaria (consentimiento por beneficios,
//    oferta de retención; NUNCA muestra probabilidad ni features)

import init, { predict_churn, verify_parity } from './churn-wasm/pkg/churn_wasm.js';

// ─── Configuración del Mock Bank ───
const MOCK_BANK_URL = 'http://localhost:8000';
let currentClientId = 'CLIENT_001';

let wasmReady = false;
let modelJson = null;
let contractJson = null;

const sleep = ms => new Promise(r => setTimeout(r, ms));

// ─── Helper: logger visual de llamadas a la API (solo vista operador) ───
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
        await init();
        console.log('✅ WASM cargado');

        const modelResp = await fetch('xgb_model_edge_iot.json');
        modelJson = await modelResp.text();
        console.log('✅ Modelo IoT cargado:', modelJson.length, 'bytes');

        const contractResp = await fetch('inference_contract_edge_iot.json');
        contractJson = await contractResp.text();
        console.log('✅ Contrato IoT cargado:', contractJson.length, 'bytes');

        wasmReady = true;
        document.getElementById('status').textContent =
            '🟢 Motor WASM IoT listo (34 features)';
        document.getElementById('status').style.color = '#22c55e';

        await runParityCheck();
        await loadFeaturesFromBank(currentClientId);
        await checkPreviousConsent(currentClientId);
    } catch (err) {
        console.error('❌ Error de inicialización:', err);
        document.getElementById('status').textContent = '🔴 Error: ' + err.message;
        document.getElementById('status').style.color = '#ef4444';
    }
}

// ═══════════════════════════════════════════════════════════
// VISTA OPERADOR
// ═══════════════════════════════════════════════════════════

async function loadFeaturesFromBank(clientId) {
    const btn = document.getElementById('loadClientBtn');
    if (btn) btn.disabled = true;
    try {
        const resp = await fetch(`${MOCK_BANK_URL}/api/features/${clientId}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const features = data.features;
        for (const [key, value] of Object.entries(features)) {
            const el = document.getElementById(key);
            if (el) el.value = value;
        }
        logApi('GET', `/api/features/${clientId}`, 'ok',
            `— ${Object.keys(features).length} campos cargados`);
        updateProfileTag(clientId);
        currentClientId = clientId;
    } catch (err) {
        logApi('GET', `/api/features/${clientId}`, 'err', `— ${err.message}`);
        console.error('Error cargando features del banco:', err);
    } finally {
        if (btn) btn.disabled = false;
    }
}

async function checkPreviousConsent(clientId) {
    try {
        const resp = await fetch(`${MOCK_BANK_URL}/api/consent/${clientId}`);
        if (!resp.ok) {
            logApi('GET', `/api/consent/${clientId}`, 'err', `— HTTP ${resp.status}`);
            return false;
        }
        const data = await resp.json();
        const checkbox = document.getElementById('consentCheckbox');
        if (checkbox) checkbox.checked = data.consent === true;
        logApi('GET', `/api/consent/${clientId}`, 'ok',
            data.consent ? `— consentimiento previo (${data.granted_at})` : '— sin consentimiento registrado');
        return data.consent;
    } catch (err) {
        logApi('GET', `/api/consent/${clientId}`, 'err', `— ${err.message}`);
        return false;
    }
}

async function registerConsentInBank(consent, clientId) {
    try {
        const resp = await fetch(`${MOCK_BANK_URL}/api/consent`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ client_id: clientId, consent: consent }),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        logApi('POST', '/api/consent', 'ok',
            `— ${consent ? 'otorgado' : 'revocado'} para ${clientId}`);
        return true;
    } catch (err) {
        logApi('POST', '/api/consent', 'err', `— ${err.message}`);
        return false;
    }
}

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

function handlePredict() {
    if (!wasmReady) { alert('El motor WASM aún no está listo'); return; }

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

    const startTime = performance.now();
    const resultJson = predict_churn(JSON.stringify(input), modelJson, contractJson, consent);
    const endTime = performance.now();
    const result = JSON.parse(resultJson);

    if (result.error) {
        document.getElementById('result').innerHTML = `
            <div style="text-align: center; padding: 20px;">
                <div style="font-size: 2em; color: #f59e0b; margin-bottom: 10px;">⚠️ ${result.error}</div>
                <div style="color: #94a3b8; font-size: 0.9em;">
                    Marca el checkbox de consentimiento para usar datos IoT.
                </div>
            </div>`;
        return;
    }

    const churnColor = result.is_churn ? '#ef4444' : '#22c55e';
    const churnLabel = result.is_churn ? '🔴 ALTO RIESGO DE CHURN' : '🟢 Cliente estable';
    const cashbackHtml = result.offer_retention ? `
        <div class="cashback-card">
            <h3>🎁 Oferta de Retención Activada</h3>
            <p>Cliente con alta probabilidad de abandono y consentimiento otorgado.
               <br><strong>Cashback sugerido: $50 USD</strong> para retener.</p>
        </div>` : '';
    const consentBadge = result.consent_granted
        ? '<span style="color: #22c55e;">🔓 IoT activado</span>'
        : '<span style="color: #f59e0b;">🔒 IoT desactivado</span>';

    document.getElementById('result').innerHTML = `
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 2em; margin-bottom: 10px;">${churnLabel}</div>
            <div style="font-size: 3em; font-weight: bold; color: ${churnColor};">
                ${result.probability_pct.toFixed(2)}%
            </div>
            <div style="color: #888; margin-top: 10px;">Probabilidad de abandono</div>
            <div style="color: #666; margin-top: 5px; font-size: 0.9em;">
                Margin: ${result.margin.toFixed(6)} | Threshold: ${result.threshold} |
                Features: ${result.n_features} | ${consentBadge} |
                ⏱️ ${(endTime - startTime).toFixed(2)}ms
            </div>
            ${cashbackHtml}
        </div>`;
}

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
            const resultJson = verify_parity(
                JSON.stringify(parity.features[i]),
                parity.expected_proba[i],
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
                <br><small>Max diff: ${Math.max(...results.map(r => r.diff)).toExponential(2)}</small>`;
        } else {
            parityDiv.innerHTML = '<span style="color: #ef4444;">❌ Paridad FALLÓ</span>';
            console.error('Parity results:', results);
        }
    } catch (err) {
        console.error('Error en parity check:', err);
        document.getElementById('parity').innerHTML =
            `<span style="color: #f59e0b;">⚠️ No se pudo verificar paridad: ${err.message}</span>`;
    }
}

// ═══════════════════════════════════════════════════════════
// VISTA CLIENTE (simulador de app bancaria)
// ═══════════════════════════════════════════════════════════

const CUSTOMER_PROFILES = {
    CLIENT_001: {
        firstName: 'Ana',
        balance: '$12,450.00',
        card: 'Tarjeta Blue •••• 4821',
        transactions: [
            { label: 'Supermercado La Esquina', amount: -45.20, when: 'Hoy' },
            { label: 'Farmacia Del Ahorro', amount: -12.50, when: 'Ayer' },
            { label: 'Depósito de nómina', amount: 2300.00, when: '31 Oct' },
        ],
    },
    CLIENT_002: {
        firstName: 'Luis',
        balance: '$3,890.75',
        card: 'Tarjeta Silver •••• 7734',
        transactions: [
            { label: 'Netflix', amount: -15.99, when: 'Hoy' },
            { label: 'Uber', amount: -8.40, when: 'Ayer' },
            { label: 'Depósito freelance', amount: 1150.00, when: '30 Oct' },
        ],
    },
};

let customerConsent = false;
let customerInitialized = false;
const consentDeclined = new Set();

function renderCustomerProfile(clientId) {
    const profile = CUSTOMER_PROFILES[clientId];
    if (!profile) return;
    document.getElementById('customerName').textContent = profile.firstName;
    document.getElementById('balanceAmount').textContent = profile.balance;
    document.getElementById('balanceCard').textContent = profile.card;
    const txList = document.getElementById('txList');
    txList.innerHTML = profile.transactions.map(tx => `
        <div class="tx-row">
            <div>
                ${tx.label}
                <span class="tx-when">${tx.when}</span>
            </div>
            <div class="tx-amount ${tx.amount < 0 ? 'neg' : 'pos'}">
                ${tx.amount < 0 ? '-' : '+'}$${Math.abs(tx.amount).toFixed(2)}
            </div>
        </div>`).join('');
}

function lockedOffersHtml() {
    return `
        <div class="offer-locked">
            🔒 Beneficios personalizados desactivados.<br>
            <small>Actívalos en Ajustes (⚙️) para recibir beneficios
            acordes a tu uso.</small>
        </div>`;
}

function updateCustomerUI() {
    const clientId = document.getElementById('customerSelect').value;
    const modal = document.getElementById('consentModal');
    const simBtn = document.getElementById('simulateBtn');
    const zone = document.getElementById('offerZone');

    // El modal SOLO se muestra si:
    // 1. No hay consentimiento previo
    // 2. El cliente NO ha respondido al modal en esta sesión
    const shouldShowModal = !customerConsent && !consentDeclined.has(clientId);
    
    modal.hidden = !shouldShowModal;

    if (customerConsent) {
        simBtn.hidden = false;
    } else {
        simBtn.hidden = true;
        if (!shouldShowModal) {
            zone.innerHTML = lockedOffersHtml();
        }
    }
}

async function refreshCustomerConsent() {
    const clientId = document.getElementById('customerSelect').value;
    const consent = await checkPreviousConsent(clientId);
    customerConsent = consent === true;
    document.getElementById('settingsConsentToggle').checked = customerConsent;
    updateCustomerUI();
    
    // NUEVO: Si ya hay consentimiento previo, ejecutar simulación automáticamente
    if (customerConsent) {
        await simulateUsage();
    }
}

async function handleConsentDecision(accepted) {
    const clientId = document.getElementById('customerSelect').value;
    
    // 1. Ocultar modal INMEDIATAMENTE (sin esperar nada)
    document.getElementById('consentModal').hidden = true;
    
    // 2. Marcar que este cliente ya respondió al modal (para esta sesión)
    consentDeclined.add(clientId);
    
    try {
        if (accepted) {
            await registerConsentInBank(true, clientId);
            customerConsent = true;
            document.getElementById('settingsConsentToggle').checked = true;
            
            // Mostrar botón de simulación
            document.getElementById('simulateBtn').hidden = false;
            document.getElementById('offerZone').innerHTML = '';
            
            // Ejecutar simulación automáticamente
            await simulateUsage();
        } else {
            await registerConsentInBank(false, clientId);
            customerConsent = false;
            document.getElementById('settingsConsentToggle').checked = false;
            
            // Ocultar botón de simulación y mostrar mensaje de bloqueo
            document.getElementById('simulateBtn').hidden = true;
            document.getElementById('offerZone').innerHTML = lockedOffersHtml();
        }
    } catch (err) {
        console.error('Error registrando consentimiento:', err);
        // Aún así, el modal permanece oculto
    }
}

function renderOfferZone(result) {
    const zone = document.getElementById('offerZone');
    if (result.offer_retention) {
        zone.innerHTML = `
            <div class="offer-card">
                <div class="offer-emoji">🎁</div>
                <h4>¡Te extrañamos!</h4>
                <p>Notamos que has usado menos la app últimamente. Como gracias:
                   usa tu tarjeta este mes y recibe <strong>$50 de cashback</strong>.</p>
                <button class="btn-primary" id="claimOfferBtn">Quiero mi beneficio</button>
            </div>`;
        zone.querySelector('#claimOfferBtn')?.addEventListener('click', () => {
            zone.innerHTML = `
                <div class="offer-card neutral">
                    <div class="offer-emoji">✅</div>
                    <h4>Beneficio activado</h4>
                    <p>Tu cashback de $50 se acreditará al completar 5 compras
                       este mes. ¡Gracias por seguir con nosotros!</p>
                </div>`;
        });
    } else {
        zone.innerHTML = `
            <div class="offer-card neutral">
                <div class="offer-emoji">✨</div>
                <h4>Todo en orden</h4>
                <p>Tu cuenta está activa y sin novedades.
                   Gracias por ser parte de Mi Banco.</p>
            </div>`;
    }
}

async function simulateUsage() {
    const btn = document.getElementById('simulateBtn');
    const status = document.getElementById('simStatus');
    btn.disabled = true;
    const steps = [
        '📲 Registrando aperturas de app...',
        '💳 Revisando pagos contactless...',
        '📶 Midiendo tiempos de respuesta en POS...',
        '🔐 Analizando en tu dispositivo (los datos no salen)...',
    ];
    for (const s of steps) {
        status.textContent = s;
        await sleep(450);
    }
    try {
        const clientId = document.getElementById('customerSelect').value;
        const resp = await fetch(`${MOCK_BANK_URL}/api/features/${clientId}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const raw = { ...data.features };
        delete raw.client_id;

        const resultJson = predict_churn(JSON.stringify(raw), modelJson, contractJson, true);
        const result = JSON.parse(resultJson);
        if (result.error) throw new Error(result.error);

        // Solo consola: el cliente NUNCA ve la probabilidad
        console.log('[customer-view] inferencia interna (no mostrada):',
            result.probability_pct.toFixed(2) + '%');

        status.textContent = '';
        renderOfferZone(result);
    } catch (err) {
        status.textContent = '⚠️ No pudimos completar el análisis: ' + err.message;
    } finally {
        btn.disabled = false;
    }
}

// ─── Cambio de vistas ───
async function showCustomerView() {
    document.getElementById('operatorView').style.display = 'none';
    document.getElementById('customerView').hidden = false;
    document.getElementById('toggleOperator').classList.remove('active');
    document.getElementById('toggleCustomer').classList.add('active');
    if (!customerInitialized) {
        customerInitialized = true;
        renderCustomerProfile(document.getElementById('customerSelect').value);
    }
    await refreshCustomerConsent();
}

function showOperatorView() {
    document.getElementById('customerView').hidden = true;
    document.getElementById('operatorView').style.display = '';
    document.getElementById('toggleCustomer').classList.remove('active');
    document.getElementById('toggleOperator').classList.add('active');
    checkPreviousConsent(currentClientId);
}

// ─── Event listeners: Vista Operador ───
document.getElementById('predictBtn').addEventListener('click', handlePredict);

document.getElementById('loadClientBtn')?.addEventListener('click', async () => {
    const selectedId = document.getElementById('clientSelect').value;
    currentClientId = selectedId;
    await loadFeaturesFromBank(selectedId);
    await checkPreviousConsent(selectedId);
});

document.getElementById('clientSelect')?.addEventListener('change', async (e) => {
    currentClientId = e.target.value;
    await loadFeaturesFromBank(currentClientId);
    await checkPreviousConsent(currentClientId);
});

document.getElementById('consentCheckbox')?.addEventListener('change', async (e) => {
    await registerConsentInBank(e.target.checked, currentClientId);
});

// ─── Event listeners: Toggle de vistas ───
document.getElementById('toggleOperator').addEventListener('click', showOperatorView);
document.getElementById('toggleCustomer').addEventListener('click', showCustomerView);

// ─── Event listeners: Vista Cliente ───
document.getElementById('customerSelect').addEventListener('change', async (e) => {
    renderCustomerProfile(e.target.value);
    document.getElementById('offerZone').innerHTML = '';
    document.getElementById('simStatus').textContent = '';
    await refreshCustomerConsent();
});

document.getElementById('consentAccept').addEventListener('click', () => handleConsentDecision(true));
document.getElementById('consentDecline').addEventListener('click', () => handleConsentDecision(false));

document.getElementById('simulateBtn').addEventListener('click', simulateUsage);

document.getElementById('settingsBtn').addEventListener('click', () => {
    const panel = document.getElementById('settingsPanel');
    panel.hidden = !panel.hidden;
});

document.getElementById('closeSettings').addEventListener('click', () => {
    document.getElementById('settingsPanel').hidden = true;
});

document.getElementById('settingsConsentToggle').addEventListener('change', async (e) => {
    const clientId = document.getElementById('customerSelect').value;
    customerConsent = e.target.checked;
    await registerConsentInBank(customerConsent, clientId);
    if (customerConsent) consentDeclined.delete(clientId);
    updateCustomerUI();
});

// Inicializar al cargar
initialize();
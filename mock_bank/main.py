#!/usr/bin/env python
"""
mock_bank/main.py
─────────────────
FastAPI backend que simula la API del banco.
Provee datos crudos del cliente y gestiona consentimiento.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone  # ✅ Añadido timezone
from typing import Optional
# import json  # ❌ Eliminado (no se usa)

app = FastAPI(
    title="Mock Bank API",
    description="API simulada del banco para Churn Edge AI",
    version="1.0.0"
)

# CORS habilitado para el frontend (demo en localhost:8080)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: ["https://tu-dominio.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Base de datos en memoria (simulada) ───
# En producción esto sería PostgreSQL/MongoDB
CONSENT_DB: dict[str, dict] = {}

# ─── Datos de clientes (simulados) ───
CLIENTS_DATA = {
    "CLIENT_001": {
        "client_id": "CLIENT_001",
        "Customer_Age": 45,
        "Gender": "F",
        "Dependent_count": 3,
        "Education_Level": "Graduate",
        "Marital_Status": "Married",
        "Income_Category": "$60K - $80K",
        "Card_Category": "Blue",
        "Months_on_book": 39,
        "Total_Relationship_Count": 5,
        "Months_Inactive_12_mon": 2,
        "Contacts_Count_12_mon": 3,
        "Credit_Limit": 12691.0,
        "Total_Revolving_Bal": 777,
        "Avg_Open_To_Buy": 11914.0,
        "Total_Amt_Chng_Q4_Q1": 0.8,
        "Total_Trans_Amt": 3500,
        "Total_Trans_Ct": 45,
        "Total_Ct_Chng_Q4_Q1": 0.69,
        "Avg_Utilization_Ratio": 0.22,
        # ─── 6 features IoT ───
        "app_open_count_30d": 76,
        "contactless_usage_count_30d": 58,
        "payment_flow_abandon_count_30d": 5,
        "nfc_failed_attempts_30d": 3,
        "app_crash_count_30d": 2,
        "pos_latency_p95_ms": 1500,
    },
    "CLIENT_002": {
        "client_id": "CLIENT_002",
        "Customer_Age": 32,
        "Gender": "M",
        "Dependent_count": 1,
        "Education_Level": "High School",
        "Marital_Status": "Single",
        "Income_Category": "$40K - $60K",
        "Card_Category": "Silver",
        "Months_on_book": 24,
        "Total_Relationship_Count": 3,
        "Months_Inactive_12_mon": 4,
        "Contacts_Count_12_mon": 1,
        "Credit_Limit": 8500.0,
        "Total_Revolving_Bal": 1200,
        "Avg_Open_To_Buy": 7300.0,
        "Total_Amt_Chng_Q4_Q1": 1.2,
        "Total_Trans_Amt": 5200,
        "Total_Trans_Ct": 68,
        "Total_Ct_Chng_Q4_Q1": 0.85,
        "Avg_Utilization_Ratio": 0.35,
        # ─── 6 features IoT ───
        "app_open_count_30d": 120,
        "contactless_usage_count_30d": 95,
        "payment_flow_abandon_count_30d": 2,
        "nfc_failed_attempts_30d": 1,
        "app_crash_count_30d": 0,
        "pos_latency_p95_ms": 800,
    },
}


# ─── Modelos Pydantic ───
class ConsentRequest(BaseModel):
    client_id: str
    consent: bool
    timestamp: Optional[str] = None


class ConsentResponse(BaseModel):
    client_id: str
    consent: bool
    granted_at: str
    message: str


# ─── Endpoints ───

@app.get("/api/health")
async def health_check():
    """Health check del servicio."""
    return {
        "status": "healthy",
        "service": "Mock Bank API",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),  # ✅ Fix Python 3.12+
        "clients_available": len(CLIENTS_DATA),
    }


@app.get("/api/features/{client_id}")
async def get_client_features(client_id: str):
    """
    Retorna los datos crudos del cliente (19 bank + 6 IoT).
    El frontend envía este JSON al WASM para inferencia.
    """
    if client_id not in CLIENTS_DATA:
        raise HTTPException(
            status_code=404,
            detail=f"Cliente {client_id} no encontrado"
        )
    
    return {
        "client_id": client_id,
        "features": CLIENTS_DATA[client_id],
        "timestamp": datetime.now(timezone.utc).isoformat(),  # ✅ Fix Python 3.12+
    }


@app.post("/api/consent")
async def register_consent(request: ConsentRequest):
    """
    Registra o actualiza el consentimiento del cliente.
    """
    if request.client_id not in CLIENTS_DATA:
        raise HTTPException(
            status_code=404,
            detail=f"Cliente {request.client_id} no encontrado"
        )
    
    timestamp = request.timestamp or datetime.now(timezone.utc).isoformat()  # ✅ Fix Python 3.12+
    
    # Guardar/actualizar consentimiento
    CONSENT_DB[request.client_id] = {
        "client_id": request.client_id,
        "consent": request.consent,
        "granted_at": timestamp,
        "ip_address": "127.0.0.1",  # En producción: request.client.host
    }
    
    message = (
        f"Consentimiento {'otorgado' if request.consent else 'revocado'} "
        f"para {request.client_id}"
    )
    
    return ConsentResponse(
        client_id=request.client_id,
        consent=request.consent,
        granted_at=timestamp,
        message=message,
    )


@app.get("/api/consent/{client_id}")
async def check_consent(client_id: str):
    """
    Verifica si el cliente ha otorgado consentimiento.
    El frontend usa esto al cargar la página para restaurar el estado del checkbox.
    """
    if client_id not in CLIENTS_DATA:
        raise HTTPException(
            status_code=404,
            detail=f"Cliente {client_id} no encontrado"
        )
    
    if client_id not in CONSENT_DB:
        return {
            "client_id": client_id,
            "consent": False,
            "granted_at": None,
            "message": "Consentimiento no registrado",
        }
    
    consent_data = CONSENT_DB[client_id]
    return {
        "client_id": client_id,
        "consent": consent_data["consent"],
        "granted_at": consent_data["granted_at"],
        "message": "Consentimiento registrado",
    }


# ─── Endpoint adicional para debugging ───
@app.get("/api/clients")
async def list_clients():
    """Lista todos los clientes disponibles (solo para debugging)."""
    return {
        "clients": list(CLIENTS_DATA.keys()),
        "total": len(CLIENTS_DATA),
    }


if __name__ == "__main__":
    import uvicorn
    print("🏦 Iniciando Mock Bank API en http://localhost:8000")
    print("📖 Documentación: http://localhost:8000/docs")
    # ✅ String de importación en lugar del objeto app
    uvicorn.run("mock_bank.main:app", host="0.0.0.0", port=8000, reload=True)
    #uvicorn.run(app, host="0.0.0.0", port=8000, reload=False) 
# 🏦 Mock Bank API

Backend simulado del banco para Churn Edge AI.

## 📋 Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/health` | Health check del servicio |
| GET | `/api/features/{client_id}` | Datos crudos del cliente (19 bank + 6 IoT) |
| POST | `/api/consent` | Registra consentimiento del cliente |
| GET | `/api/consent/{client_id}` | Verifica consentimiento del cliente |
| GET | `/api/clients` | Lista clientes disponibles (debugging) |

## 🚀 Instalación

```bash
cd mock_bank
pip install -r requirements.txt
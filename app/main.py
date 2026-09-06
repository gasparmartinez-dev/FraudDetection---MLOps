from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib

from app.schemas import TransactionInput
#Importo la clase de validación de app/schemas.py

mis_modelos = {}
# Diccionario para guardar los modelos

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Carfar los modelos
        mis_modelos["scaler"] = joblib.load("models/scaler.joblib")
        mis_modelos["model"] = joblib.load("models/xgb_fraud_model.joblib")
        print("Modelo y escalador cargados en memoria RAM.")
    except Exception as e:
        print("Error al cargar el escalador y el modelo.")

    yield   # El servidor se queda encendido esperando peticiones

    mis_modelos.clear()
    print("Memoria RAM liberada.")

app = FastAPI(
    title = "API Detección Fraudes",
    description = "API REST para la detección de fraudes en transacciones financieras.",
    version = "1.0.0",
    lifespan = lifespan
)

# ENDPOINT DE ESTADO
@app.get("/health", tags=["Estado"])
def health_check():
    if "model" not in mis_modelos or "scaler" not in mis_modelos:
        raise HTTPException(
            status_code = 503,
            detail = "Servicio no disponible: los modelos no están cargados en memoria."
        )
    return {
        "status": "ok",
        "message": "Servicio de Detección de Fraudes activo y listo."
    }
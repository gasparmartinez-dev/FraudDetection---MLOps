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

import pandas as pd

@app.post("/predict", tags=["Prediccion"])
def predict_fraud(transaction: TransactionInput):
    if "model" not in mis_modelos or "scaler" not in mis_modelos:
        raise HTTPException(
            status_code = 503,
            detail = "Servicio no disponible: modelos no cargados."
        )

    #Convertir la entrada Pydantic a DataFrame de Pandas
    data_dict = transaction.model_dump()
    df_input = pd.DataFrame([data_dict])

    #Escalar Time y Amount
    scaler = mis_modelos["scaler"]
    df_input[["Time", "Scaler"]] = scaler.transform(df_input[["Time", "Amount"]])

    #Generar predicción
    model = mis_modelos["model"]
    prediccion = int(model.predict(df_input)[0])
    probabilidad = float(model.predict_proba(df_input)[0][1])

    decision = "BLOQUEADO" if prediccion == 1 else "APROVADO"

    #Devolver la respuesta en formato JSON
    return{
        "es_fraude": bool(prediccion == 1),
        "probabilidad_fraude": round(probabilidad, 4),
        "decision": decision
    }
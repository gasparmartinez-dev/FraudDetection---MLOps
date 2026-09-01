# 💳 Real-Time Credit Card Fraud Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

Este proyecto implementa una solución de **Machine Learning de extremo a extremo (End-to-End)** orientada al sector Financiero/Fintech. Permite clasificar transacciones con tarjeta de crédito como **Legítimas** o **Fraudulentas** en tiempo real mediante un microservicio REST empaquetado en Docker.

---

## 📄 Descripción del Problema

La detección de fraude financiero presenta un reto clave en el mundo real:
1. **Desbalanceo Extremo de Clases:** Menos del `0.2%` de las transacciones corresponden a eventos de fraude.
2. **Costo de Negocio Asimétrico:** Un *Falso Negativo* (no detectar un fraude) implica pérdida de dinero directa, mientras que un *Falso Positivo* (bloquear una compra legítima) genera fricción con el usuario.

El objetivo del proyecto es maximizar la detección de fraudes (Recall) manteniendo una precisión adecuada para reducir falsos positivos.

---

## 🛠️ Arquitectura del Proyecto

```text
fraud-detection-mlops/
├── data/                  # Datasets (Raw y Processed)
├── models/                # Artefactos entrenados (.joblib)
├── src/                   # Código modular de preprocesamiento y entrenamiento
├── app/                   # API REST servida con FastAPI y Pydantic
├── Dockerfile             # Contenedor para despliegue en producción
└── requirements.txt       # Dependencias del proyecto

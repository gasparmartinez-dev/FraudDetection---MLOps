import joblib
import os
from xgboost import XGBClassifier

from preprocessing import cargar_datos, separar_escalar_datos

df = cargar_datos("data/creditcard.csv")

X_train, X_test, y_train, y_test = separar_escalar_datos(df, ruta_scaler="models/scaler.joblib")

print(X_train[:5])
import joblib
import os
from xgboost import XGBClassifier

from preprocessing import cargar_datos, separar_escalar_datos

df = cargar_datos("data/creditcard.csv")

X_train, X_test, y_train, y_test = separar_escalar_datos(df, ruta_scaler="models/scaler.joblib")

# print(X_train[:5])

# Cálculo del factor de balanceo

num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
factor_balanceo = num_neg / num_pos

print(f"El factor de balanceo es: {factor_balanceo}")

model = XGBClassifier(
    scale_pos_weight=factor_balanceo,
    random_state=42,
    eval_metric="logloss"
)
print("Ajustando el modelo con los datos de entrenamiento...")
model.fit(X_train, y_train)
print("Modelo entrenado.")

# Guardar el modelo entrenado

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgboost_model.joblib")

print("Modelo guardado.")

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# Imprimir métricas de evaluación
print("\nEVALUACIÓN DEL MODELO")
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# La matriz de confusion indica que hay 56852 verdaderos negativos, 
# 81 verdaderos positivos, 12 falsos positivos y 17 falsos negativos
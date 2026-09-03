import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def cargar_datos(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    if "Class" not in df.columns:
        raise ValueError("La columna Class no se encuentra en el dataframe")

    return df

def separar_escalar_datos(df, ruta_scaler = "models/scaler.joblib", test_size = 0.2, random_state = 42):
    # Separo los datos en X e y
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    #Crear el scaler
    scaler = RobustScaler()

    X_train[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
    X_test[["Amount", "Time"]] = scaler.transform(X_test[["Amount", "Time"]])

    os.makedirs(os.path.dirname(ruta_scaler), exist_ok=True)
    joblib.dump(scaler, ruta_scaler)

    return X_train, X_test, y_train, y_test
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    if "Class" not in df.columns:
        raise ValueError("La columna Class no se encuentra en el dataframe")

    return df
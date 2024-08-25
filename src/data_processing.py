import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data():
    # Cargar los datos
    clientes = pd.read_csv("../data/raw/online_shoppers_intention.csv")
    
    # Preprocesamiento de datos
    meses_dict = {"Feb": 1, "Mar": 2, "May": 3, "June": 4, "Jul": 5, "Aug": 6, "Sep": 7, "Oct": 8, "Nov": 9, "Dec": 10}
    clientes["Mes_Num"] = clientes["Month"].map(meses_dict)
    clientes.drop(columns=['Month'], inplace=True)
    clientes['Weekend'] = clientes['Weekend'].astype(int)
    clientes['Revenue'] = clientes['Revenue'].astype(int)
    clientes_new = pd.get_dummies(clientes, columns=['VisitorType'], dtype=int)
    
    return clientes_new

def prepare_features(data):
    features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 
                'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
    X = data[features]
    y = data['Revenue']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    return X_resampled, y_resampled

if __name__ == "__main__":
    data = load_and_preprocess_data()
    X, y = prepare_features(data)
    
    # Guardar los datos procesados
    pd.concat([X, y], axis=1).to_csv("../data/processed/processed_data.csv", index=False)
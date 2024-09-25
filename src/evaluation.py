import pandas as pd
import pickle
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def load_test_data():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "test", "test_data.csv")
    data = pd.read_csv(data_path)
    return data.drop('Revenue', axis=1), data['Revenue']

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    try:
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
            if X_test.shape[1] > n_features:
                print(f"Advertencia: Recortando características de {X_test.shape[1]} a {n_features}")
                X_test = X_test.iloc[:, :n_features]
            elif X_test.shape[1] < n_features:
                print(f"Error: El modelo espera {n_features} características, pero los datos tienen {X_test.shape[1]}")
                return

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{model_name} Evaluation:")
        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    except Exception as e:
        print(f"Error al evaluar el modelo {model_name}: {str(e)}")

if __name__ == "__main__":
    X_test, y_test = load_test_data()
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    
    model_names = ["Random Forest", "LightGBM", "XGBoost", "Gradient Boosting", "SVM", "Voting Classifier"]
    
    for i, name in enumerate(model_names, 1):
        model_path = os.path.join(model_dir, f"model_{i}.pkl")
        print(f"Cargando {name}...")
        try:
            model = load_model(model_path)
            evaluate_model(model, X_test, y_test, name)
        except Exception as e:
            print(f"Error al cargar el modelo {name}: {str(e)}")
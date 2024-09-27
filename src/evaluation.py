import pandas as pd
import pickle
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.base import BaseEstimator

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
        if not isinstance(model, BaseEstimator):
            print(f"Advertencia: {model_name} no es un modelo de scikit-learn válido.")
            print(f"Tipo del objeto: {type(model)}")
            if isinstance(model, np.ndarray):
                print(f"Forma del array: {model.shape}")
                print(f"Primeros elementos: {model[:5]}")
            elif hasattr(model, '__dict__'):
                print(f"Atributos del objeto: {model.__dict__.keys()}")
            return

        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
            if X_test.shape[1] > n_features:
                print(f"Advertencia: Recortando características de {X_test.shape[1]} a {n_features}")
                X_test = X_test.iloc[:, :n_features]
            elif X_test.shape[1] < n_features:
                print(f"Error: El modelo espera {n_features} características, pero los datos tienen {X_test.shape[1]}")
                return

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{model_name} Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        if y_pred_proba is not None:
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            print(f"AUC-ROC: {auc_roc:.4f}")
        
    except Exception as e:
        print(f"Error al evaluar el modelo {model_name}: {str(e)}")

if __name__ == "__main__":
    X_test, y_test = load_test_data()
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    
    model_names = ["Random Forest", "LightGBM", "XGBoost", "Gradient Boosting", "SVM", "Voting Classifier"]
    
    for i, name in enumerate(model_names, 1):
        model_path = os.path.join(model_dir, f"model_{i}.pkl")
        print(f"\nCargando {name}...")
        try:
            model = load_model(model_path)
            print(f"Tipo de objeto cargado para {name}: {type(model)}")
            if isinstance(model, BaseEstimator):
                print(f"Características del modelo:")
                if hasattr(model, 'n_features_in_'):
                    print(f"  - Número de características esperadas: {model.n_features_in_}")
                if hasattr(model, 'feature_importances_'):
                    print(f"  - Las 5 características más importantes: {model.feature_importances_.argsort()[-5:][::-1]}")
                if hasattr(model, 'classes_'):
                    print(f"  - Clases: {model.classes_}")
            evaluate_model(model, X_test, y_test, name)
        except Exception as e:
            print(f"Error al cargar el modelo {name}: {str(e)}")

    print("\nResumen de las características de los datos de prueba:")
    print(f"Número de muestras: {X_test.shape[0]}")
    print(f"Número de características: {X_test.shape[1]}")
    print(f"Nombres de las características: {X_test.columns.tolist()}")
    print(f"Distribución de clases en y_test: {y_test.value_counts(normalize=True)}")
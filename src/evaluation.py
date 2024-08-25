import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def load_test_data():
    data = pd.read_csv("../data/test/test_data.csv")
    X_test = data.drop('Revenue', axis=1)
    y_test = data['Revenue']
    return X_test, y_test

def load_models():
    models = []
    for i in range(1, 7):  # 6 modelos supervisados
        models.append(joblib.load(f"../models/model_{i}.pkl"))
    return models

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"ROC-AUC Score: {roc_auc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    X_test, y_test = load_test_data()
    models = load_models()
    model_names = ["Random Forest", "LightGBM", "XGBoost", "Gradient Boosting", "SVM", "Voting Classifier"]
    
    for model, name in zip(models, model_names):
        evaluate_model(model, X_test, y_test, name)
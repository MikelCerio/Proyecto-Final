import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import KMeans
import joblib

def load_data():
    data = pd.read_csv("../data/processed/processed_data.csv")
    X = data.drop('Revenue', axis=1)
    y = data['Revenue']
    return X, y

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modelo 1: Random Forest
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Modelo 2: LightGBM
    lgbm = lgb.LGBMClassifier(class_weight='balanced', random_state=42)
    lgbm.fit(X_train, y_train)
    
    # Modelo 3: XGBoost
    xgb_model = xgb.XGBClassifier(scale_pos_weight=sum(y==0)/sum(y==1), random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Modelo 4: Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    
    # Modelo 5: SVM
    svm = SVC(class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    # Modelo 6: Voting Classifier (Ensamble de modelos supervisados)
    voting_clf = VotingClassifier(estimators=[
        ('rf', rf),
        ('lgbm', lgbm),
        ('xgb', xgb_model),
        ('gb', gb),
        ('svm', svm)
    ], voting='soft')
    voting_clf.fit(X_train, y_train)
    
    # Modelo 7: K-Means (No supervisado)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    
    return rf, lgbm, xgb_model, gb, svm, voting_clf, kmeans, X_test, y_test

if __name__ == "__main__":
    X, y = load_data()
    models = train_models(X, y)
    
    # Guardar los modelos
    for i, model in enumerate(models[:-2]):  # Excluimos X_test y y_test
        joblib.dump(model, f"../models/model_{i+1}.pkl")
    
    # Guardar datos de prueba
    pd.concat([models[-2], models[-1]], axis=1).to_csv("../data/test/test_data.csv", index=False)

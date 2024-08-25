import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo
model = joblib.load("../models/model_6.pkl")  # Usamos el Voting Classifier

st.title('Predictor de Intención de Compra')

# Crear inputs para las características
administrative = st.number_input('Número de páginas administrativas visitadas', min_value=0)
administrative_duration = st.number_input('Duración total en páginas administrativas', min_value=0.0)
informational = st.number_input('Número de páginas informativas visitadas', min_value=0)
informational_duration = st.number_input('Duración total en páginas informativas', min_value=0.0)
product_related = st.number_input('Número de páginas de productos visitadas', min_value=0)
product_related_duration = st.number_input('Duración total en páginas de productos', min_value=0.0)
bounce_rates = st.number_input('Tasa de rebote', min_value=0.0, max_value=1.0)
exit_rates = st.number_input('Tasa de salida', min_value=0.0, max_value=1.0)
page_values = st.number_input('Valor de la página', min_value=0.0)

# Crear un DataFrame con los inputs
input_data = pd.DataFrame({
    'Administrative': [administrative],
    'Administrative_Duration': [administrative_duration],
    'Informational': [informational],
    'Informational_Duration': [informational_duration],
    'ProductRelated': [product_related],
    'ProductRelated_Duration': [product_related_duration],
    'BounceRates': [bounce_rates],
    'ExitRates': [exit_rates],
    'PageValues': [page_values]
})

# Hacer la predicción
if st.button('Predecir'):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    st.write(f"Predicción: {'Compra' if prediction[0] == 1 else 'No Compra'}")
    st.write(f"Probabilidad de compra: {probability:.2f}")
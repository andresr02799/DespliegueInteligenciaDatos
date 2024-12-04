import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Cargar los modelos entrenados
random_forest_model = joblib.load('random_forest_model.pkl')
ridge_model = joblib.load('ridge_model.pkl')
lasso_model = joblib.load('lasso_model.pkl')

# Cargar el dataset
data_path = "student_spending (1).csv"
student_spending = pd.read_csv(data_path)

# Interfaz de usuario
st.title('Recomendaciones para la Gestión Financiera de Estudiantes')

# Variables de entrada
monthly_income = st.number_input("Ingrese su ingreso mensual", min_value=0, max_value=10000)
entertainment = st.number_input("Gasto en entretenimiento", min_value=0, max_value=1000)
personal_care = st.number_input("Gasto en cuidado personal", min_value=0, max_value=1000)
technology = st.number_input("Gasto en tecnología", min_value=0, max_value=1000)
miscellaneous = st.number_input("Gasto en misceláneos", min_value=0, max_value=1000)

# Preprocesamiento de las entradas
user_data = pd.DataFrame({
    'monthly_income': [monthly_income],
    'entertainment': [entertainment],
    'personal_care': [personal_care],
    'technology': [technology],
    'miscellaneous': [miscellaneous]
})

# Escalar las variables (usamos el mismo escalador que usaste al entrenar los modelos)
scaler = MinMaxScaler()
user_data_scaled = scaler.fit_transform(user_data)

# Hacer predicciones con los modelos entrenados
rf_prediction = random_forest_model.predict(user_data_scaled)
ridge_prediction = ridge_model.predict(user_data_scaled)
lasso_prediction = lasso_model.predict(user_data_scaled)

# Mostrar las recomendaciones
st.write(f"Predicción de gasto no esencial usando Random Forest: {rf_prediction[0]:.2f}")
st.write(f"Predicción de gasto no esencial usando Ridge: {ridge_prediction[0]:.2f}")
st.write(f"Predicción de gasto no esencial usando Lasso: {lasso_prediction[0]:.2f}")

# Recomendaciones personalizadas
if rf_prediction[0] > 500:
    st.write("¡Tu gasto no esencial es alto! Considera reducir el gasto en entretenimiento, tecnología o misceláneos.")
else:
    st.write("¡Buen trabajo! Tu gasto no esencial está en un rango adecuado. Sigue así.")

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Cargar los modelos entrenados
random_forest_model = joblib.load('random_forest_model.pkl')
ridge_model = joblib.load('ridge_model.pkl')
lasso_model = joblib.load('lasso_model.pkl')
scaler = joblib.load('scaler.pkl')  # Asegúrate de haber guardado el scaler

# Título de la app
st.title("Recomendaciones Financieras para Estudiantes")

# Entrada de datos del usuario
age = st.slider("Edad", 18, 25, 20)
monthly_income = st.number_input("Ingreso mensual", min_value=0, value=1000)
tuition = st.number_input("Gastos de matrícula", min_value=0, value=1000)
housing = st.number_input("Gastos de alojamiento", min_value=0, value=500)
food = st.number_input("Gastos en comida", min_value=0, value=200)
transportation = st.number_input("Gastos en transporte", min_value=0, value=100)
entertainment = st.number_input("Gastos en entretenimiento", min_value=0, value=50)
technology = st.number_input("Gastos en tecnología", min_value=0, value=100)
miscellaneous = st.number_input("Gastos misceláneos", min_value=0, value=30)

# Crear un dataframe con los datos de entrada
input_data = pd.DataFrame({
    'age': [age],
    'monthly_income': [monthly_income],
    'tuition': [tuition],
    'housing': [housing],
    'food': [food],
    'transportation': [transportation],
    'entertainment': [entertainment],
    'technology': [technology],
    'miscellaneous': [miscellaneous]
})

# Excluir las mismas columnas que fueron excluidas en el entrenamiento
# Las columnas que el modelo no usó durante el entrenamiento
input_data_filtered = input_data.drop(columns=['gender', 'year_in_school', 'major', 'preferred_payment_method'], errors='ignore')

# Escalar las variables de entrada usando el mismo escalador que se usó para el entrenamiento
input_data_scaled = scaler.transform(input_data_filtered)  # Usa el transform() aquí, no fit_transform()

# Realizar las predicciones
rf_prediction = random_forest_model.predict(input_data_scaled)
ridge_prediction = ridge_model.predict(input_data_scaled)
lasso_prediction = lasso_model.predict(input_data_scaled)

# Mostrar los resultados
st.write(f"**Random Forest** - Gasto No Esencial estimado: {rf_prediction[0]:.2f} unidades monetarias")
st.write(f"**Ridge Regression** - Gasto No Esencial estimado: {ridge_prediction[0]:.2f} unidades monetarias")
st.write(f"**Lasso Regression** - Gasto No Esencial estimado: {lasso_prediction[0]:.2f} unidades monetarias")

# Recomendaciones basadas en los modelos
if rf_prediction[0] > 200:
    st.write("¡Recomendación! Considera reducir tus gastos en entretenimiento, tecnología y misceláneos.")
else:
    st.write("¡Buen trabajo! Tus gastos no esenciales están dentro de un rango razonable.")

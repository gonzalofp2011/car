import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Car Evaluation - Machine Learning",
    page_icon="🚗",
    layout="centered"
)

st.title("Prediccion de Evaluacion de Autos")
st.write("**Nombre:** gonzalo fernandez")
st.write("**Codigo ISIL:** 75929809")
st.write("**Cuaderno COLAB:**https://colab.research.google.com/drive/1nkRJdN6O6xxKTQ9XlSHvobV7THTxXs7o?usp=sharing")

st.markdown("""
Esta aplicación utiliza modelos de machine learning para predecir la evaluación de un auto
segun sus caracterasticas: precio de compra, costo de mantenimiento, numero de puertas,
capacidad de personas, tama?o de maletera y nivel de seguridad.

Las clases posibles son:

- **unacc**: inaceptable
- **acc**: aceptable
- **good**: bueno
- **vgood**: muy bueno
""")

model_option = st.selectbox(
    "Seleccione el modelo:",
    ["Árbol de Decisión", "Random Forest"]
)

if model_option == "Árbol de Decisión":
    model = joblib.load("modelos/modelo_arbol_decision.pkl")
else:
    model = joblib.load("modelos/modelo_random_forest.pkl")

st.subheader("Ingrese las características del auto")

buying = st.selectbox("Precio de compra", ["vhigh", "high", "med", "low"])
maint = st.selectbox("Costo de mantenimiento", ["vhigh", "high", "med", "low"])
doors = st.selectbox("Número de puertas", ["2", "3", "4", "5more"])
persons = st.selectbox("Capacidad de personas", ["2", "4", "more"])
lug_boot = st.selectbox("Tamaño de maletera", ["small", "med", "big"])
safety = st.selectbox("Nivel de seguridad", ["low", "med", "high"])

input_df = pd.DataFrame([{
    "buying": buying,
    "maint": maint,
    "doors": doors,
    "persons": persons,
    "lug_boot": lug_boot,
    "safety": safety
}])

st.subheader("Datos ingresados")
st.dataframe(input_df)

if st.button("Realizar predicción"):
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    st.success(f"Resultado de la evaluación: **{prediction}**")

    st.subheader("Probabilidades por clase")
    prob_df = pd.DataFrame({
        "Clase": classes,
        "Probabilidad": probabilities
    }).sort_values("Probabilidad", ascending=False)

    st.dataframe(prob_df)

st.markdown("---")
st.caption("Proyecto desarrollado para la Evaluación PA2 - Machine Learning y despliegue web con Streamlit.")
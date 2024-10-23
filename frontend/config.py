import sys
import os
# Ajustar el sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import joblib
import numpy as np
import pandas as pd
import gradio as gr
from database.db_operations import insert_prediction

# Cargar el modelo
model = joblib.load(r'C:\Users\busin\Desktop\mode_ic\models\best_ensemble_model.joblib')

# Definir el orden correcto de las columnas
column_order = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'smoking_status']

# Mapeo manual para variables categóricas
gender_map = {'Male': 0, 'Female': 1}
smoking_status_map = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2}

def predict_stroke(gender, age, hypertension, heart_disease, avg_glucose_level, smoking_status):
    # Codificar variables categóricas
    gender_encoded = gender_map.get(gender, -1)  # -1 para valores desconocidos
    smoking_status_encoded = smoking_status_map.get(smoking_status, -1)  # -1 para valores desconocidos

    # Crear un diccionario con los datos de entrada
    input_data = {
        'gender': gender_encoded,
        'age': age,
        'hypertension': int(hypertension),
        'heart_disease': int(heart_disease),
        'avg_glucose_level': avg_glucose_level,
        'smoking_status': smoking_status_encoded
    }

    # Crear un DataFrame con el orden correcto de las columnas
    input_df = pd.DataFrame([input_data])[column_order]

    # Añadir características ingenieradas
    input_df['age_squared'] = input_df['age'] ** 2
    input_df['glucose_age_interaction'] = input_df['age'] * input_df['avg_glucose_level']

    # Realizar la predicción
    prediction = model.predict_proba(input_df)[0]
    stroke_probability = prediction[1]

    # Redondear stroke_probability a dos decimales
    stroke_probability = round(stroke_probability, 2)

    result = f"La probabilidad de ictus es: {stroke_probability:.2%}\n"
    if stroke_probability > 0.5:
        result += "Se recomienda consultar a un médico."
    else:
        result += "El riesgo parece ser bajo, pero siempre es bueno mantener hábitos saludables."

    try: 
        # Preparar los datos para insertar en la base de datos
        data_values = (
            gender,
            age,
            int(hypertension),
            int(heart_disease),
            avg_glucose_level,
            smoking_status,
            stroke_probability 
        )

        # Insertar los datos en la base de datos
        print("Antes de insertar en la base de datos")
        insert_prediction(data_values)
        print("Después de insertar en la base de datos")
    except Exception as e:
        print(f"Error al insertar en la base de datos: {e}")

    return result

# Crear la interfaz de Gradio
iface = gr.Interface(
    fn=predict_stroke,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Género"),
        gr.Slider(0, 100, label="Edad"),
        gr.Checkbox(label="Hipertensión"),
        gr.Checkbox(label="Enfermedad cardíaca"),
        gr.Number(label="Nivel promedio de glucosa"),
        gr.Dropdown(["formerly smoked", "never smoked", "smokes"], label="Estado de fumador")
    ],
    outputs="text",
    title="Predictor de Riesgo de Ictus",
    description="Introduce los datos del paciente para evaluar el riesgo de ictus."
)

# Lanzar la interfaz
if __name__=="__main__":
    iface.launch()
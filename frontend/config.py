import joblib
import numpy as np
import pandas as pd
import gradio as gr

# Cargar el modelo
model = joblib.load('best_ensemble_model.joblib')

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

    result = f"La probabilidad de ictus es: {stroke_probability:.2%}\n"
    if stroke_probability > 0.5:
        result += "Se recomienda consultar a un médico."
    else:
        result += "El riesgo parece ser bajo, pero siempre es bueno mantener hábitos saludables."

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
iface.launch()
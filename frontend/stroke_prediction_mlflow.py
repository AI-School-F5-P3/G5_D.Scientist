import joblib
import numpy as np
import pandas as pd
import gradio as gr
import mlflow
import mlflow.sklearn
from datetime import datetime

# Configurar MLFlow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("stroke-prediction-monitoring")

# Cargar el modelo
model = joblib.load('best_ensemble_model.joblib')

# Definir el orden correcto de las columnas
column_order = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'smoking_status']

# Mapeo manual para variables categóricas
gender_map = {'Male': 0, 'Female': 1}
smoking_status_map = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2}

def get_key_by_value(dictionary, value):
    """Función auxiliar para obtener la llave a partir del valor de forma segura"""
    for key, val in dictionary.items():
        if val == value:
            return key
    return "Unknown"  # Valor por defecto si no se encuentra

def log_prediction(input_data, prediction, prediction_probability):
    """Registra la predicción en MLFlow"""
    with mlflow.start_run(run_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        # Registrar parámetros de entrada
        mlflow.log_params({
            "gender": get_key_by_value(gender_map, input_data['gender']),
            "age": input_data['age'],
            "hypertension": input_data['hypertension'],
            "heart_disease": input_data['heart_disease'],
            "avg_glucose_level": input_data['avg_glucose_level'],
            "smoking_status": get_key_by_value(smoking_status_map, input_data['smoking_status'])
        })
        
        # Registrar métricas
        mlflow.log_metrics({
            "prediction_probability": prediction_probability,
            "prediction": 1 if prediction_probability > 0.5 else 0
        })
        
        # Registrar feature engineering
        mlflow.log_metrics({
            "age_squared": input_data['age'] ** 2,
            "glucose_age_interaction": input_data['age'] * input_data['avg_glucose_level'],
        })

def predict_stroke(gender, age, hypertension, heart_disease, avg_glucose_level, smoking_status):
    # Codificar variables categóricas
    gender_encoded = gender_map.get(gender, -1)
    smoking_status_encoded = smoking_status_map.get(smoking_status, -1)
    
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
    
    # Registrar la predicción en MLFlow
    log_prediction(
        input_data=input_data,
        prediction=1 if stroke_probability > 0.5 else 0,
        prediction_probability=stroke_probability
    )
    
    result = f"La probabilidad de ictus es: {stroke_probability:.2%}\n"
    if stroke_probability > 0.5:
        result += "Se recomienda realizar más pruebas."
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

if __name__ == "__main__":
    # Lanzar la interfaz
    iface.launch()
import joblib
import pandas as pd
import gradio as gr
from prometheus_client import Counter, Histogram, start_http_server
import time

# Cargar el modelo
model = joblib.load('../models/best_ensemble_model.joblib')

# Definir el orden correcto de las columnas
column_order = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'smoking_status']

# Función para cargar el informe guardado
def load_model_report():
    with open('../models/model_report.txt', 'r') as f:
        report = f.read()
    return report

# Mapeo manual para variables categóricas
gender_map = {'Male': 0, 'Female': 1}
smoking_status_map = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2}

# Iniciar servidor de Prometheus en el puerto 8000 para recolectar métricas
start_http_server(8000)

# Definir métricas para Prometheus
PREDICTION_COUNTER = Counter('stroke_predictions_total', 'Número total de predicciones realizadas')
ERROR_COUNTER = Counter('stroke_prediction_errors_total', 'Número total de errores en las predicciones')
REQUEST_LATENCY = Histogram('stroke_prediction_latency_seconds', 'Latencia del tiempo de respuesta de las predicciones')

# Función para predecir el riesgo de ictus y monitorizar las métricas
@REQUEST_LATENCY.time()  # Mide el tiempo que tarda en hacer una predicción
def predict_stroke(gender, age, hypertension, heart_disease, avg_glucose_level, smoking_status):
    try:
        # Incrementar el contador de predicciones
        PREDICTION_COUNTER.inc()

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

        # Formatear el resultado de la predicción
        result = f"La probabilidad de ictus es: {stroke_probability:.2%}\n"
        if stroke_probability > 0.5:
            result += "Se recomienda consultar a un médico."
        else:
            result += "El riesgo parece ser bajo, pero siempre es bueno mantener hábitos saludables."

        # Cargar el informe del modelo
        report = load_model_report()

        # Mostrar ambos: predicción e informe
        return result + "\n\n" + report

    except Exception as e:
        # Incrementar el contador de errores en caso de fallo
        ERROR_COUNTER.inc()
        raise e

# Crear la interfaz en Gradio
iface = gr.Interface(
    fn=predict_stroke,  # Función que incluye la monitorización
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Género"),
        gr.Slider(18, 100, label="Edad"),
        gr.Checkbox(label="Hipertensión"),
        gr.Checkbox(label="Enfermedad cardíaca"),
        gr.Number(label="Nivel promedio de glucosa"),
        gr.Dropdown(["formerly smoked", "never smoked", "smokes"], label="Estado de fumador")
    ],
    outputs="text",  # Mostrar el resultado en formato de texto
    title="Predictor de Riesgo de Ictus con Monitorización",
    description="Introduce los datos del paciente para evaluar el riesgo de ictus y monitorizar el rendimiento del modelo."
)

# Lanzar la interfaz
iface.launch()

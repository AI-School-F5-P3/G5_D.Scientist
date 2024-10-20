import joblib
import pandas as pd
import gradio as gr
from prometheus_client import Counter, Histogram, start_http_server
import time
import mlflow
import mlflow.sklearn

# Configurar el experimento de MLflow
mlflow.set_experiment("Stroke Prediction Experiment")

# Cargar el modelo
model = joblib.load('../models/best_ensemble_model.joblib')

# Definir el orden correcto de las columnas
column_order = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'smoking_status']

# Función para cargar el informe guardado
def load_model_report():
    with open('../models/model_report.txt', 'r') as f:
        report = f.read()
    return report

# Función para cargar las gráficas (ROC y Feature Importance)
def load_model_graphs():
    feature_importance_path = '../models/feature_importance.png'
    roc_curve_path = '../models/roc_curve.png'
    return feature_importance_path, roc_curve_path

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
@REQUEST_LATENCY.time()
def predict_stroke(gender, age, hypertension, heart_disease, avg_glucose_level, smoking_status):
    try:
        with mlflow.start_run():
            # Incrementar el contador de predicciones
            PREDICTION_COUNTER.inc()

            # Codificar variables categóricas
            gender_encoded = gender_map.get(gender, -1)
            smoking_status_encoded = smoking_status_map.get(smoking_status, -1)

            # Convertir edad a entero y asegurarse de que esté entre 18 y 100
            age = max(18, min(100, int(age)))

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

            # Registrar parámetros y métricas en MLflow
            mlflow.log_params(input_data)
            mlflow.log_metric("stroke_probability", stroke_probability)

            # Formatear el resultado de la predicción
            result = f"La probabilidad de ictus es: {stroke_probability:.2%}\n"
            if stroke_probability > 0.5:
                result += "Se recomienda consultar a un médico."
            else:
                result += "El riesgo parece ser bajo, pero siempre es bueno mantener hábitos saludables."

            return result

    except Exception as e:
        # Incrementar el contador de errores en caso de fallo
        ERROR_COUNTER.inc()
        raise e

# Función combinada para mostrar el informe y las gráficas
def show_report_and_graphs():
    report = load_model_report()
    feature_importance_path, roc_curve_path = load_model_graphs()
    return report, feature_importance_path, roc_curve_path

# Crear la interfaz en Gradio
with gr.Blocks() as demo:
    with gr.Tabs():
        # Pestaña de predicción
        with gr.Tab("Predicción de Ictus"):
            with gr.Row():
                # Columna izquierda: Formulario de entrada
                with gr.Column():
                    gender = gr.Dropdown(["Male", "Female"], label="Género")
                    # Acotar la edad entre 18 y 100
                    age = gr.Number(label="Edad", value=18, precision=0, step=1, minimum=18, maximum=100)
                    hypertension = gr.Checkbox(label="Hipertensión")
                    heart_disease = gr.Checkbox(label="Enfermedad cardíaca")
                    avg_glucose_level = gr.Number(label="Nivel promedio de glucosa")
                    smoking_status = gr.Dropdown(["formerly smoked", "never smoked", "smokes"], label="Estado de fumador")
                    
                    predict_button = gr.Button("Realizar Predicción")

                # Columna derecha: Resultados de la predicción
                with gr.Column():
                    pred_output = gr.Textbox(label="Resultado de la Predicción", interactive=False)
        
        # Pestaña para mostrar el informe y las gráficas
        with gr.Tab("Informe del Modelo y Gráficas"):
            # Botón para cargar y mostrar el informe y las gráficas
            show_report_button = gr.Button("Mostrar Informe y Gráficas")

            # Dividir el área de informe y gráficas en dos columnas
            with gr.Row():
                # Columna para el informe
                with gr.Column():
                    report_output = gr.Textbox(label="Informe del Modelo", interactive=False)
                
                # Columna para las gráficas
                with gr.Column():
                    feature_img = gr.Image(label="Importancia de las Características", type="filepath")
                    roc_img = gr.Image(label="Curva ROC", type="filepath")

            show_report_button.click(show_report_and_graphs, inputs=[], outputs=[report_output, feature_img, roc_img])

    # Configurar las acciones de los botones
    predict_button.click(predict_stroke, inputs=[gender, age, hypertension, heart_disease, avg_glucose_level, smoking_status], outputs=pred_output)

# Lanzar la interfaz con el directorio ../models permitido
demo.launch(allowed_paths=["../models"])




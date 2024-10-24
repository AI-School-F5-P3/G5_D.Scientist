import joblib
import pandas as pd
import gradio as gr
import mlflow
import time

from database.db_operations import insert_prediction
import sys
import os

# Ajustar el sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Cargar el modelo
model = joblib.load('../models/best_ensemble_model.joblib')

# Definir el orden correcto de las columnas
column_order = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'smoking_status']

# Mapeo manual para variables categóricas
gender_map = {'Male': 0, 'Female': 1}
smoking_status_map = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2}

# Función para predecir el riesgo de ictus y registrar métricas con MLflow
def predict_stroke(gender, age, hypertension, heart_disease, avg_glucose_level, smoking_status):
    try:
        start_time = time.time()
        
        gender_encoded = gender_map.get(gender, -1)
        smoking_status_encoded = smoking_status_map.get(smoking_status, -1)
        age = max(18, min(100, int(age)))

        input_data = {
            'gender': gender_encoded,
            'age': age,
            'hypertension': int(hypertension),
            'heart_disease': int(heart_disease),
            'avg_glucose_level': avg_glucose_level,
            'smoking_status': smoking_status_encoded
        }

        input_df = pd.DataFrame([input_data])[column_order]
        input_df['age_squared'] = input_df['age'] ** 2
        input_df['glucose_age_interaction'] = input_df['age'] * input_df['avg_glucose_level']

        prediction = model.predict_proba(input_df)[0]
        stroke_probability = prediction[1]

        result = f"La probabilidad de ictus es: {stroke_probability:.2%}\n"
        if stroke_probability > 0.5:
            result += "Se recomienda consultar a un médico."
        else:
            result += "El riesgo parece ser bajo, pero siempre es bueno mantener hábitos saludables."
        
        latency = time.time() - start_time

        # Registrar las métricas en MLflow
        with mlflow.start_run():
            mlflow.log_param("gender", gender)
            mlflow.log_param("age", age)
            mlflow.log_param("hypertension", hypertension)
            mlflow.log_param("heart_disease", heart_disease)
            mlflow.log_param("avg_glucose_level", avg_glucose_level)
            mlflow.log_param("smoking_status", smoking_status)
            mlflow.log_metric("stroke_probability", stroke_probability)
            mlflow.log_metric("latency", latency)

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

    except Exception as e:
        with mlflow.start_run():
            mlflow.log_param("error", str(e))
        raise e

# Crear la interfaz en Gradio
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Predicción de Ictus"):
            with gr.Row():
                with gr.Column():
                    gender = gr.Dropdown(["Male", "Female"], label="Género")
                    age = gr.Number(label="Edad", value=18, precision=0, step=1, minimum=18, maximum=100)
                    hypertension = gr.Checkbox(label="Hipertensión")
                    heart_disease = gr.Checkbox(label="Enfermedad cardíaca")
                    avg_glucose_level = gr.Number(label="Nivel promedio de glucosa")
                    smoking_status = gr.Dropdown(["formerly smoked", "never smoked", "smokes"], label="Estado de fumador")
                    
                    predict_button = gr.Button("Realizar Predicción")

                with gr.Column():
                    pred_output = gr.Textbox(label="Resultado de la Predicción", interactive=False)

    predict_button.click(predict_stroke, inputs=[gender, age, hypertension, heart_disease, avg_glucose_level, smoking_status], outputs=pred_output)

# Lanzar la interfaz
demo.launch()
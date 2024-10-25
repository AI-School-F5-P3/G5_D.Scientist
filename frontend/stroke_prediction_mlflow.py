import joblib
import numpy as np
import pandas as pd
import gradio as gr
import mlflow
import mlflow.sklearn
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

# Cargar variables de entorno (crea un archivo .env con estas variables)
load_dotenv()

# Configuración de la base de datos MySQL
MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'user': os.getenv('MYSQL_USER', 'tu_usuario'),
    'password': os.getenv('MYSQL_PASSWORD', 'tu_password'),
    'database': os.getenv('MYSQL_DATABASE', 'stroke_prediction')
}

# Configurar MLFlow con MySQL
mlflow.set_tracking_uri(f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}")
mlflow.set_experiment("stroke-prediction-monitoring")

def init_database():
    """Inicializa la base de datos y crea las tablas necesarias si no existen"""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        
        # Crear tabla para predicciones
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME,
                gender VARCHAR(10),
                age INT,
                hypertension BOOLEAN,
                heart_disease BOOLEAN,
                avg_glucose_level FLOAT,
                smoking_status VARCHAR(20),
                prediction_probability FLOAT,
                prediction BOOLEAN,
                mlflow_run_id VARCHAR(50)
            )
        """)
        
        conn.commit()
        print("Base de datos inicializada correctamente")
        
    except Error as e:
        print(f"Error al inicializar la base de datos: {e}")
        
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def save_to_mysql(input_data, prediction, prediction_probability, mlflow_run_id):
    """Guarda la predicción en MySQL"""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        
        query = """
            INSERT INTO predictions 
            (timestamp, gender, age, hypertension, heart_disease, 
             avg_glucose_level, smoking_status, prediction_probability, 
             prediction, mlflow_run_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        values = (
            datetime.now(),
            list(gender_map.keys())[list(gender_map.values()).index(input_data['gender'])],
            input_data['age'],
            input_data['hypertension'],
            input_data['heart_disease'],
            input_data['avg_glucose_level'],
            list(smoking_status_map.keys())[list(smoking_status_map.values()).index(input_data['smoking_status'])],
            prediction_probability,
            prediction > 0.5,
            mlflow_run_id
        )
        
        cursor.execute(query, values)
        conn.commit()
        
    except Error as e:
        print(f"Error al guardar en MySQL: {e}")
        
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Cargar el modelo
model = joblib.load('best_ensemble_model.joblib')

# Definir el orden correcto de las columnas
column_order = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'smoking_status']

# Mapeo manual para variables categóricas
gender_map = {'Male': 0, 'Female': 1}
smoking_status_map = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2}

def log_prediction(input_data, prediction, prediction_probability):
    """Registra la predicción en MLFlow y MySQL"""
    with mlflow.start_run(run_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        # Registrar parámetros de entrada
        mlflow.log_params({
            "gender": input_data['gender'],
            "age": input_data['age'],
            "hypertension": input_data['hypertension'],
            "heart_disease": input_data['heart_disease'],
            "avg_glucose_level": input_data['avg_glucose_level'],
            "smoking_status": input_data['smoking_status']
        })
        
        # Registrar métricas
        mlflow.log_metrics({
            "prediction_probability": prediction_probability,
            "prediction": 1 if prediction_probability > 0.5 else 0
        })
        
        # Registrar feature engineering
        mlflow.log_metrics({
            "age_squared": input_data['age'] ** 2,
            "glucose_age_interaction": input_data['age'] * input_data['avg_glucose_level']
        })
        
        # Guardar en MySQL
        save_to_mysql(input_data, prediction, prediction_probability, run.info.run_id)

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
    
    # Registrar la predicción en MLFlow y MySQL
    log_prediction(
        input_data=input_data,
        prediction=1 if stroke_probability > 0.5 else 0,
        prediction_probability=stroke_probability
    )
    
    result = f"La probabilidad de ictus es: {stroke_probability:.2%}\n"
    if stroke_probability > 0.5:
        result += "Se recomienda consultar a un médico."
    else:
        result += "El riesgo parece ser bajo, pero siempre es bueno mantener hábitos saludables."
    
    return result

# Inicializar la base de datos
init_database()

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
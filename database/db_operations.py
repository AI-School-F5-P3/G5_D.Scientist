import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_DATABASE'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )

def generate_unique_id():
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT nextval('predicciones_usuario_id_seq')")
    usuario_id = cursor.fetchone()[0]
    cursor.close()
    connection.close()
    return usuario_id

def insert_prediction(gender, age, hypertension, heart_disease, avg_glucose_level, smoking_status, prediction_result):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        usuario_id = generate_unique_id()
        
        query = sql.SQL("""
            INSERT INTO predicciones (usuario_id, gender, age, hypertension, heart_disease,
                                      avg_glucose_level, smoking_status, prediction_result, fecha)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW());
        """)
        
        cursor.execute(query, (usuario_id, gender, age, bool(hypertension), bool(heart_disease),
                               avg_glucose_level, smoking_status, prediction_result))
        
        connection.commit()
        cursor.close()
        connection.close()
        print(f"Predicción insertada correctamente en la base de datos con ID: {usuario_id}")
        return usuario_id
    except Exception as e:
        print(f"Error en insert_prediction: {e}")
        return None
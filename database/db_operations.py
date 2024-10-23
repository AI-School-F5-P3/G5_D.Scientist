# database/db_operations.py

from database.db_connection import get_db_connection

def insert_prediction(data):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        sql_query = """
            INSERT INTO predicciones (
                gender, age, hypertension, heart_disease, avg_glucose_level,
                smoking_status, prediction_result
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        print("Consulta SQL:", sql_query)
        print("Datos a insertar:", data)
        cursor.execute(sql_query, data)
        connection.commit()
        cursor.close()
        connection.close()
        print("Inserción exitosa en la base de datos")
    except Exception as e:
        print(f"Error en insert_prediction: {e}")


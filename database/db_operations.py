from .db_connection import get_db_connection

def insert_user(user_data):
    """
    Inserta un nuevo usuario en la tabla `datos_sensibles`.
    
    Args:
        user_data (tuple): Datos del usuario a insertar (nombre, apellido, email, contraseña, telefono, poblacion, edad, gender).
        
    Returns:
        int: ID del usuario insertado, o None si ocurre un error.
    """
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        query = """
            INSERT INTO datos_sensibles (nombre, apellido, email, contrasena, telefono, poblacion, edad, gender)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
        """
        cursor.execute(query, user_data)
        user_id = cursor.fetchone()[0]
        connection.commit()
        cursor.close()
        connection.close()
        return user_id
    except Exception as e:
        print(f"Error en insert_user: {e}")
        return None

from .db_connection import get_db_connection

def insert_prediction(prediction_data):
    """
    Inserta una nueva predicción en la tabla `predicciones`.
    
    Args:
        prediction_data (tuple): Datos de la predicción a insertar (usuario_id, gender, age, hypertension, heart_disease, avg_glucose_level, smoking_status, prediction_result).
        
    Returns:
        None
    """
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        query = """
            INSERT INTO predicciones (usuario_id, gender, age, hypertension, heart_disease,
                                      avg_glucose_level, smoking_status, prediction_result, fecha)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW());
        """
        cursor.execute(query, prediction_data)
        connection.commit()
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"Error en insert_prediction: {e}")

def get_user_by_email(email):
    """
    Recupera un usuario de la base de datos a través de su email.
    
    Args:
        email (str): Email del usuario a buscar.
        
    Returns:
        dict: Datos del usuario en formato de diccionario, o None si no existe o si ocurre un error.
    """
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        query = "SELECT * FROM datos_sensibles WHERE email = %s;"
        cursor.execute(query, (email,))
        user = cursor.fetchone()
        cursor.close()
        connection.close()
        if user:
            return {
                "id": user[0],
                "nombre": user[1],
                "apellido": user[2],
                "email": user[3],
                "contrasena": user[4],
                "telefono": user[5],
                "poblacion": user[6],
                "edad": user[7],
                "gender": user[8]
            }
        return None
    except Exception as e:
        print(f"Error en get_user_by_email: {e}")
        return None

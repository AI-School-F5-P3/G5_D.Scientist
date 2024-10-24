# database/db_connection.py

import mysql.connector
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

def get_db_connection():
    try:
        # Obtener las variables de entorno necesarias
        db_host = os.environ.get("DB_HOST")
        db_user = os.environ.get("DB_USER")
        db_password = os.environ.get("DB_PASSWORD")
        db_name = os.environ.get("DB_NAME")

        # Verificar que las variables de entorno están definidas
        if not db_user or not db_password or not db_name:
            raise ValueError("Faltan variables de entorno para la conexión a la base de datos.")

        # Imprimir información de depuración
        print(f"Intentando conectar a la base de datos '{db_name}' en '{db_host}' con el usuario '{db_user}'.")

        # Establecer la conexión a la base de datos
        connection = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name,

        )

        # Verificar si la conexión es exitosa
        if connection.is_connected():
            print("Conexión exitosa a la base de datos.")
        else:
            print("No se pudo conectar a la base de datos.")

        return connection

    except mysql.connector.Error as err:
        print(f"Error al conectar a la base de datos: {err}")
        raise

    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        raise

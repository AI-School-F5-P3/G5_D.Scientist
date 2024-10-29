import psycopg2
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def get_db_connection():
    """
    Establece y retorna una conexión a la base de datos PostgreSQL.
    """
    try:
        connection = psycopg2.connect(
           host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_DATABASE"),
            port=os.getenv("DB_PORT", 5432)
        )
        print("Conexión exitosa a la base de datos ictus_db.")
        return connection

    except psycopg2.Error as err:
        print(f"Error al conectar a la base de datos: {err}")
        raise
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        raise
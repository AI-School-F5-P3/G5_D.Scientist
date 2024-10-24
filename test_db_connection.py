from database.db_connection import get_db_connection

try:
    connection = get_db_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT DATABASE();")
    database_name = cursor.fetchone()
    print(f"Conectado a la base de datos: {database_name[0]}")
    cursor.close()
    connection.close()
except Exception as e:
    print(f"Error al conectar a la base de datos: {e}")

# tests/test_db_connection.py

import pytest
from database.db_connection import get_db_connection
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_db_connection():
    """
    Verifica que la conexión a la base de datos sea exitosa.
    """
    try:
        connection = get_db_connection()
        assert connection is not None, "La conexión a la base de datos falló."
        assert connection.closed == 0, "La conexión a la base de datos debería estar abierta."
    finally:
        if connection:
            connection.close()

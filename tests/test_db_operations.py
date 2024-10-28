# tests/test_db_operations.py

import pytest
from database.db_operations import insert_prediction

@pytest.fixture
def mock_data():
    """
    Datos de prueba para la inserción en la base de datos.
    """
    return (
        'Female', 85, True, False, 130.5, 'never smoked', 0.45
    )

def test_insert_prediction(mock_data):
    """
    Verifica que los datos se inserten en la base de datos sin errores.
    """
    try:
        insert_prediction(mock_data)
        assert True  # Si la inserción es exitosa
    except Exception as e:
        assert False, f"Error en la inserción de la base de datos: {e}"

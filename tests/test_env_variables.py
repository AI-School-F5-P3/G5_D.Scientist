# tests/test_env_variables.py

import os

def test_env_variables():
    """
    Verifica que las variables de entorno necesarias estén configuradas.
    """
    required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME", "DB_PORT"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    assert not missing_vars, f"Variables de entorno faltantes: {missing_vars}"

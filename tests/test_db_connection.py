import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Añadir el directorio raíz al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_connection import get_db_connection

class TestDBConnection(unittest.TestCase):
    @patch('database.db_connection.psycopg2.connect')
    def test_get_db_connection(self, mock_connect):
        # Configurar el mock
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection

        # Llamar a la función
        connection = get_db_connection()

        # Verificar que se llamó a connect con los argumentos correctos
        mock_connect.assert_called_once_with(
            host='localhost',
            database='ictus_db',
            user='postgres',
            password='tu_contraseña'
        )

        # Verificar que la función devuelve la conexión
        self.assertEqual(connection, mock_connection)

if __name__ == '__main__':
    unittest.main()
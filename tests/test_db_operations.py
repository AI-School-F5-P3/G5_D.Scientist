import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Añadir el directorio raíz al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_operations import generate_unique_id, insert_prediction

class TestDBOperations(unittest.TestCase):
    @patch('database.db_operations.get_db_connection')
    def test_generate_unique_id(self, mock_get_db_connection):
        # Configurar los mocks
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_connection.cursor.return_value = mock_cursor
        mock_get_db_connection.return_value = mock_connection

        # Llamar a la función
        result = generate_unique_id()

        # Verificar el resultado
        self.assertEqual(result, 1)

        # Verificar que se llamaron los métodos correctos
        mock_cursor.execute.assert_called_once_with("SELECT nextval('predicciones_usuario_id_seq')")
        mock_cursor.close.assert_called_once()
        mock_connection.close.assert_called_once()

    @patch('database.db_operations.get_db_connection')
    @patch('database.db_operations.generate_unique_id')
    def test_insert_prediction(self, mock_generate_unique_id, mock_get_db_connection):
        # Configurar los mocks
        mock_connection = MagicMock()
        mock_cursor = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_get_db_connection.return_value = mock_connection
        mock_generate_unique_id.return_value = 1

        # Llamar a la función
        result = insert_prediction('Male', 65, True, False, 100, 'formerly smoked', '30')

        # Verificar el resultado
        self.assertEqual(result, 1)

        # Verificar que se llamaron los métodos correctos
        mock_cursor.execute.assert_called_once()
        mock_connection.commit.assert_called_once()
        mock_cursor.close.assert_called_once()
        mock_connection.close.assert_called_once()

if __name__ == '__main__':
    unittest.main()
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

# Añadir el directorio raíz al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import predict_stroke, gender_map, smoking_status_map, column_order, insert_prediction

class TestMain(unittest.TestCase):
    @patch('main.model')
    @patch('main.insert_prediction')
    def test_predict_stroke(self, mock_insert_prediction, mock_model):
        # Configurar el mock del modelo
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
        
        # Configurar el mock de insert_prediction
        mock_insert_prediction.return_value = 1

        # Llamar a la función
        result = predict_stroke('Male', 65, True, False, 100, 'formerly smoked')

        # Verificar el resultado
        self.assertEqual(result, "La probabilidad de sufrir un ictus es: 30%")

        # Verificar que se llamó a insert_prediction con los argumentos correctos
        mock_insert_prediction.assert_called_once_with('Male', 65, True, False, 100, 'formerly smoked', '30')

    def test_gender_map(self):
        self.assertEqual(gender_map['Male'], 0)
        self.assertEqual(gender_map['Female'], 1)

    def test_smoking_status_map(self):
        self.assertEqual(smoking_status_map['formerly smoked'], 0)
        self.assertEqual(smoking_status_map['never smoked'], 1)
        self.assertEqual(smoking_status_map['smokes'], 2)

    def test_column_order(self):
        expected_order = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'smoking_status', 'age_squared', 'glucose_age_interaction']
        self.assertEqual(column_order, expected_order)

def insert_prediction(gender, age, hypertension, heart_disease, avg_glucose_level, smoking_status, prediction_result):
    # Implementación mock de la función insert_prediction
    return 1

if __name__ == '__main__':
    unittest.main()
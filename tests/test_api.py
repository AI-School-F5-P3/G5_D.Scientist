import unittest
from fastapi.testclient import TestClient
import sys
import os

# Añadir el directorio raíz al path de Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Bienvenido a la API de predicción de ictus"})

    def test_predict_stroke(self):
        test_data = {
            "gender": "Male",
            "age": 65,
            "hypertension": True,
            "heart_disease": False,
            "avg_glucose_level": 100,
            "smoking_status": "formerly smoked"
        }
        response = self.client.post("/predict", json=test_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json())
        self.assertIn("usuario_id", response.json())

if __name__ == '__main__':
    unittest.main()
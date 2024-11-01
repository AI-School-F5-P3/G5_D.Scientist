from locust import HttpUser, task, between
import json

class GradioUser(HttpUser):
    wait_time = between(1, 5)  # Espera entre 1 y 5 segundos entre tareas

    @task
    def predict_stroke(self):
        # Datos de ejemplo para la predicción
        payload = {
            "data": [
                "Male",  # gender
                65,      # age
                True,    # hypertension
                False,   # heart_disease
                100,     # avg_glucose_level
                "formerly smoked"  # smoking_status
            ]
        }

        # Realiza una solicitud POST a la ruta de predicción de Gradio
        with self.client.post("/api/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = json.loads(response.text)
                    if "data" in result and isinstance(result["data"], list) and len(result["data"]) > 0:
                        response.success()
                    else:
                        response.failure("Respuesta inesperada: " + response.text)
                except json.JSONDecodeError:
                    response.failure("Respuesta no es JSON válido: " + response.text)
            else:
                response.failure("Código de estado inesperado: " + str(response.status_code))

    @task
    def visit_homepage(self):
        # Simula una visita a la página principal
        self.client.get("/")
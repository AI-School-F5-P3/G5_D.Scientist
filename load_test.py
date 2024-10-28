from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 5)  # Espera entre 1 y 5 segundos entre tareas

    @task
    def predict(self):
        token_response = self.client.post("/token", data={"username": "admin", "password": "password"})
        access_token = token_response.json().get("access_token")

        if access_token:
            headers = {"Authorization": f"Bearer {access_token}"}
            self.client.post("/predict", json={
                "gender": "Male",
                "age": 45,
                "hypertension": True,
                "heart_disease": False,
                "avg_glucose_level": 105.5,
                "smoking_status": "smokes"
            }, headers=headers)

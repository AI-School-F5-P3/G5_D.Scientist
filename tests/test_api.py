import pytest
from httpx import AsyncClient
from api import app  # Asegúrate de que `app` se importe correctamente desde `api.py`
from httpx import ASGITransport

@pytest.mark.asyncio
async def test_root():
    """
    Verifica que la ruta raíz (/) devuelva un código de estado 200.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API de predicción de ictus. Visita /docs para la documentación."}

@pytest.mark.asyncio
async def test_predict_endpoint():
    """
    Verifica que la ruta /predict procesa los datos y devuelve una predicción.
    Primero, se autentica para obtener un token de acceso válido.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        auth_data = {
            "username": "admin@test.com",  # Email del usuario administrador
            "password": "password"         # Contraseña correspondiente
        }
        auth_response = await ac.post("/token", data=auth_data)

        assert auth_response.status_code == 200
        access_token = auth_response.json().get("access_token")
        assert access_token is not None

        test_data = {
            "gender": "Male",
            "age": 70,
            "hypertension": True,
            "heart_disease": False,
            "avg_glucose_level": 100.5,
            "smoking_status": "never smoked"
        }

        headers = {"Authorization": f"Bearer {access_token}"}
        response = await ac.post("/predict", json=test_data, headers=headers)

        assert response.status_code == 200
        assert "prediction" in response.json()

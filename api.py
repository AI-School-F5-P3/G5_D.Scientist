from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from auth.auth_bearer import get_current_user
from auth.auth_handler import create_access_token
from database.db_operations import insert_prediction, get_user_by_email  # Asume que esta función existe aquí
from passlib.context import CryptContext
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from typing import Optional
import logging
import os

# Inicializa el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configura el contexto de encriptación para contraseñas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI()

# Endpoint raíz para verificar el estado de la API
@app.get("/")
async def root():
    return {"message": "API de predicción de ictus. Visita /docs para la documentación."}

# Modelo de solicitud para la predicción
class PredictionRequest(BaseModel):
    gender: str
    age: int
    hypertension: bool
    heart_disease: bool
    avg_glucose_level: float
    smoking_status: str

# Definición de la función para generar PDF
def generate_pdf(data, prediction, user_data=None):
    pdf_path = "prediction_report.pdf"  # Cambiado a un directorio válido en Windows
    c = canvas.Canvas(pdf_path, pagesize=letter)
    textobject = c.beginText()
    textobject.setTextOrigin(50, 750)
    textobject.setFont("Helvetica", 12)

    lines = ["Resultado de Predicción de Ictus", "-------------------------------"]
    if user_data:
        lines += [
            f"ID: {user_data.get('id', '')}",
            f"Nombre: {user_data.get('nombre', '')}",
            f"Apellido: {user_data.get('apellido', '')}",
            f"Email: {user_data.get('email', '')}",
            f"Género: {user_data.get('gender', '')}",
            f"Edad: {user_data.get('edad', '')}",
            f"Población: {user_data.get('poblacion', '')}",
            f"Teléfono: {user_data.get('telefono', '')}",
            ""
        ]
    lines += [
        f"Género: {data.gender}",
        f"Edad: {data.age}",
        f"Hipertensión: {'Sí' if data.hypertension else 'No'}",
        f"Enfermedad cardíaca: {'Sí' if data.heart_disease else 'No'}",
        f"Nivel promedio de glucosa: {data.avg_glucose_level}",
        f"Estatus de fumador: {data.smoking_status}",
        "",
        f"Resultado de la predicción: {prediction}"
    ]

    for line in lines:
        textobject.textLine(line)

    c.drawText(textobject)
    c.save()
    return pdf_path


# Definición de la función de envío de email con PDF
def send_email_with_pdf(email, pdf_path):
    # Implementa la lógica para enviar el PDF adjunto por correo
    pass  # Esta función debe implementarse según tus necesidades

# Endpoint para generar el token de autenticación
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user_by_email(form_data.username)  # Asume que existe esta función para obtener el usuario
    if not user or not pwd_context.verify(form_data.password, user["contrasena"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciales inválidas")
    access_token = create_access_token(data={"sub": user["email"], "id": user["id"]})
    return {"access_token": access_token, "token_type": "bearer"}

# Endpoint de predicción
@app.post("/predict")
async def predict(data: PredictionRequest, current_user: Optional[dict] = Depends(get_current_user)):
    prediction = "Alto riesgo"
    logger.info(f"Prediction result: {prediction}")

    # Usar "usuario_id" en lugar de "datos_sensibles_id"
    prediction_data = (
        current_user["id"] if current_user else None,
        data.gender, data.age, data.hypertension,
        data.heart_disease, data.avg_glucose_level, data.smoking_status, prediction
    )
    insert_prediction(prediction_data)

    pdf_path = generate_pdf(data, prediction, user_data=current_user)
    if current_user:
        send_email_with_pdf(current_user["email"], pdf_path)
        os.remove(pdf_path)

    return {"message": "Predicción completada", "prediction": prediction}


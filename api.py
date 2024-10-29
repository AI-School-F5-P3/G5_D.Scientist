from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from database.db_operations import insert_prediction

app = FastAPI()

# Cargar el modelo
model = joblib.load('model/best_ensemble_model.joblib')

# Definir el orden correcto de las columnas
column_order = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'smoking_status', 'age_squared', 'glucose_age_interaction']

# Mapeo manual para variables categóricas
gender_map = {'Male': 0, 'Female': 1}
smoking_status_map = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2}

class StrokeData(BaseModel):
    gender: str
    age: float
    hypertension: bool
    heart_disease: bool
    avg_glucose_level: float
    smoking_status: str

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de predicción de ictus, entra en /docs ."}

@app.post("/predict")
async def predict_stroke(data: StrokeData):
    # Crear un DataFrame con los datos de entrada
    input_data = pd.DataFrame([[
        data.gender,
        data.age,
        data.hypertension,
        data.heart_disease,
        data.avg_glucose_level,
        data.smoking_status
    ]], columns=['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'smoking_status'])

    # Aplicar mapeo manual
    input_data['gender'] = input_data['gender'].map(gender_map)
    input_data['smoking_status'] = input_data['smoking_status'].map(smoking_status_map)

    # Convertir valores booleanos a enteros
    input_data['hypertension'] = input_data['hypertension'].astype(int)
    input_data['heart_disease'] = input_data['heart_disease'].astype(int)

    # Crear las características adicionales
    input_data['age_squared'] = input_data['age'] ** 2
    input_data['glucose_age_interaction'] = input_data['age'] * input_data['avg_glucose_level']

    # Asegurar que no hay valores NaN
    input_data = input_data.fillna(0)

    # Asegurar el orden correcto de las columnas
    input_data = input_data.reindex(columns=column_order)

    # Realizar la predicción
    probability = model.predict_proba(input_data)[0][1]
    prediction_percentage = int(round(probability * 100))

    # Eliminar los últimos ceros
    prediction_str = str(prediction_percentage)
    prediction_str = prediction_str.rstrip('0')
    if prediction_str.endswith('.'):
        prediction_str = prediction_str[:-1]
    
    # Guardar la predicción en la base de datos
    usuario_id = insert_prediction(
        data.gender,
        data.age,
        data.hypertension,
        data.heart_disease,
        data.avg_glucose_level,
        data.smoking_status,
        prediction_str
    )

    return {"prediction": prediction_str, "usuario_id": usuario_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
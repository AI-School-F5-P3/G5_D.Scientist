import gradio as gr
import pandas as pd
import joblib
import numpy as np
import os

# Debugging: Imprimir información sobre el directorio y archivos
print("Directorio actual:", os.getcwd())
print("Archivos en el directorio:", os.listdir())

try:
    # Cargar el modelo guardado
    model = joblib.load('../models/best_model_CatBoost_hiper.joblib')
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")

def predict_stroke(age, hypertension, heart_disease, avg_glucose_level, bmi,
                  gender, smoking_status):
    try:
        # Imprimir los datos de entrada para debugging
        print("Datos recibidos:", {
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'gender': gender,
            'smoking_status': smoking_status
        })

        # Crear un DataFrame con los datos de entrada
        input_data = pd.DataFrame({
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'gender': [gender],
            'smoking_status': [smoking_status]
        })
        
        print("DataFrame creado exitosamente")
        
        # Realizar la predicción
        prediction_proba = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        
        print(f"Predicción exitosa: {prediction}, Probabilidad: {prediction_proba}")
        
        # Preparar el resultado
        risk_level = "Alto" if prediction_proba > 0.5 else "Bajo"
        result = f"""
        Probabilidad de accidente cerebrovascular: {prediction_proba:.2%}
        Nivel de riesgo: {risk_level}
        
        Factores de riesgo principales:
        - Edad: {'Alto' if age > 60 else 'Moderado' if age > 40 else 'Bajo'}
        - Glucosa: {'Alto' if avg_glucose_level > 200 else 'Moderado' if avg_glucose_level > 140 else 'Bajo'}
        - Hipertensión: {'Presente' if hypertension else 'No presente'}
        - Enfermedad cardíaca: {'Presente' if heart_disease else 'No presente'}
        """
        
        return result
    
    except Exception as e:
        print(f"Error en la predicción: {str(e)}")
        return f"Error en la predicción: {str(e)}"


# Crear la interfaz
with gr.Blocks(title="Predictor de Accidente Cerebrovascular") as iface:
    gr.Markdown("# Predictor de Riesgo de Accidente Cerebrovascular")
    gr.Markdown("Ingrese los datos del paciente para evaluar el riesgo.")
    
    with gr.Row():
        with gr.Column():
            # Datos numéricos
            age = gr.Slider(minimum=0, maximum=100, value=50, label="Edad")
            hypertension = gr.Checkbox(label="¿Tiene hipertensión?")
            heart_disease = gr.Checkbox(label="¿Tiene enfermedad cardíaca?")
            avg_glucose_level = gr.Number(label="Nivel promedio de glucosa", value=100)
            bmi = gr.Number(label="Índice de masa corporal (BMI)", value=25)
        
        with gr.Column():
            # Datos categóricos
            gender = gr.Radio(choices=["Male", "Female"], label="Género")
            smoking_status = gr.Radio(
                choices=["formerly smoked", "never smoked", "smokes"],
                label="Estado de fumador"
            )
    
    # Botón de predicción y salida
    submit_btn = gr.Button("Predecir Riesgo", variant="primary")
    output = gr.Textbox(label="Resultado")
    
    # Conectar la función
    submit_btn.click(
        predict_stroke,
        inputs=[age, hypertension, heart_disease, avg_glucose_level, bmi,
                gender, smoking_status],
        outputs=output
    )
    
    gr.Markdown("""
    ### Información importante
    - Este es un modelo predictivo basado en datos históricos
    - Los resultados son orientativos y no sustituyen el diagnóstico médico profesional
    - Consulte siempre con un profesional de la salud para una evaluación completa
    """)

if __name__ == "__main__":
    iface.launch()
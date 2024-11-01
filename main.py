import joblib
import numpy as np
import pandas as pd
import gradio as gr
from pathlib import Path
import os
from dotenv import load_dotenv
from database.db_operations import insert_prediction
import itertools
import logging
import warnings

# Suprimir advertencias
warnings.filterwarnings("ignore")

# Establezca el nivel de registro en ERROR para suprimir los mensajes de información y advertencia
logging.getLogger().setLevel(logging.ERROR)

# Cargar variables de entorno
load_dotenv()

# Cargar el modelo
model = joblib.load('model/best_ensemble_model.joblib')

# Definir el orden correcto de las columnas
column_order = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'smoking_status', 'age_squared', 'glucose_age_interaction']

# Mapeo manual para variables categóricas
gender_map = {'Male': 0, 'Female': 1}
smoking_status_map = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2}

# Función que realiza la predicción de ictus basada en los datos de entrada.
def predict_stroke(gender, age, hypertension, heart_disease, avg_glucose_level, smoking_status):
    # Crear un DataFrame con los datos de entrada
    input_data = pd.DataFrame([[gender, age, hypertension, heart_disease, avg_glucose_level, smoking_status]],
                              columns=['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'smoking_status'])

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

    # Insertar la predicción en la base de datos
    insert_prediction(gender, age, hypertension, heart_disease, avg_glucose_level, smoking_status, prediction_percentage)

    # Devolver el resultado
    return f"La probabilidad de sufrir un ictus es: {prediction_percentage}%"

    # Crea el encabezado de la página con el logo y el título del proyecto.
def create_header():
    
    # Un objeto `gr.Row` que contiene el logo y el título.
    return gr.Row([   

        # Imagen del logo      
        gr.Image("media/neuro.png", show_label=False, width=50, height=50, interactive=False),

        # Título del proyecto
        gr.Markdown("# NeuroPredict", elem_classes=["logo-text"])
    ], elem_classes=["header"])

    # Crea la página de inicio con un video, imágenes y texto descriptivo.
def create_home_page():

    # Rutas de las imágenes a mostrar
    image_paths = ["media/ictus_4.jpeg", "media/ictus_2.jpg", "media/ictus_3.jpeg", "media/ictus_5.jpeg", "media/ictus_6.jpeg"]
    
    # Textos descriptivos para las imágenes
    captions = [
        "Prevención del ictus: Mantén un estilo de vida saludable",
        "Factores de riesgo: Conoce lo que puede aumentar tus probabilidades",
        "Síntomas del ictus: Reconoce las señales de advertencia",
        "Alimentación: Recuerda la importancia de la buena alimentación",
        "Alegria: Sobre todo no te olvides de SONREIR y dejar el estrés a un lado"
    ]

    # Ciclo infinito de imágenes y textos descriptivos
    image_caption_cycle = itertools.cycle(zip(image_paths, captions))

    def update_image_and_caption():

        """
        Actualiza la imagen y el texto descriptivo cada 5 segundos.
        """ 
        return next(image_caption_cycle)

    with gr.Blocks() as home:
        
        # Agregar el encabezado
        create_header()

        # Agregar un video que se reproduce automáticamente
        gr.Video("media/ictus_video.mp4", autoplay=True, loop=True, show_label=False, width=800, elem_classes=["main-video"])
        
        # Título de bienvenida
        gr.Markdown("# Bienvenido al Predictor de Riesgo de Ictus", elem_classes=["center-text"])
        
        with gr.Row():

            # Imagen inicial y texto descriptivo
            image = gr.Image(value=image_paths[0], show_label=False, interactive=False, elem_classes=["info-image"])
            caption = gr.Markdown(value=captions[0], elem_classes=["image-caption"])
        
        # Texto que indica deslizar para ver más información
        gr.Markdown("Desliza para ver más información sobre el ictus", elem_classes=["center-text"])

        # Actualizar la imagen y el texto descriptivo cada 5 segundos
        home.load(update_image_and_caption, outputs=[image, caption], every=5)

    return home

    # Crea la interfaz para realizar la predicción del riesgo de ictus.
def create_prediction_interface():

    with gr.Blocks() as prediction:

        # Agregar el encabezado
        create_header()

        # Título de la sección de predicción
        gr.Markdown("## Predicción de Riesgo de Ictus")
        with gr.Row():
            with gr.Column():
                gender = gr.Radio(["Male", "Female"], label="Género")
                age = gr.Slider(minimum=0, maximum=120, step=1, label="Edad")
                hypertension = gr.Checkbox(label="Hipertensión")
                heart_disease = gr.Checkbox(label="Enfermedad cardíaca")
                avg_glucose_level = gr.Slider(minimum=50, maximum=300, step=1, label="Nivel promedio de glucosa")
                smoking_status = gr.Radio(["formerly smoked", "never smoked", "smokes"], label="Estado de fumador")
                predict_button = gr.Button("Predecir")
            with gr.Column():
                output = gr.Textbox(label="Resultado de la predicción")
        
        # Configurar el botón para realizar la predicción
        predict_button.click(
            predict_stroke,
            inputs=[gender, age, hypertension, heart_disease, avg_glucose_level, smoking_status],
            outputs=output
        )
    return prediction

# Crea la página con información detallada sobre el ictus.
def create_info_page():
    with gr.Blocks() as info:
        create_header()
        gr.Markdown("## Información sobre el Ictus")
        gr.Markdown("""
        El ictus, también conocido como accidente cerebrovascular, ocurre cuando el suministro de sangre a una parte del cerebro se interrumpe o se reduce, lo que impide que el tejido cerebral reciba oxígeno y nutrientes. Las células cerebrales comienzan a morir en minutos.

        ### Tipos de Ictus
        1. **Ictus isquémico**: Causado por un coágulo que bloquea un vaso sanguíneo en el cerebro.
        2. **Ictus hemorrágico**: Causado por una hemorragia en el cerebro cuando un vaso sanguíneo se rompe.

        ### Factores de Riesgo
        - Hipertensión
        - Tabaquismo
        - Diabetes
        - Obesidad
        - Enfermedades cardíacas
        - Edad avanzada
        - Antecedentes familiares

        ### Síntomas
        - Debilidad o entumecimiento repentino en la cara, brazo o pierna, especialmente en un lado del cuerpo
        - Confusión repentina, dificultad para hablar o entender
        - Problemas repentinos de visión en uno o ambos ojos
        - Dificultad repentina para caminar, mareos, pérdida de equilibrio o coordinación
        - Dolor de cabeza severo repentino sin causa conocida

        ### Prevención
        - Controlar la presión arterial
        - Dejar de fumar
        - Mantener una dieta saludable
        - Hacer ejercicio regularmente
        - Mantener un peso saludable
        - Limitar el consumo de alcohol
        """)
    return info

def create_resources_page():
    with gr.Blocks() as resources:
        create_header()
        gr.Markdown("## Recursos Adicionales")
        gr.Markdown("""
        ### Organizaciones de Apoyo
        - [Federación Española de Ictus](https://ictusfederacion.es/)
        - [Asociación Española contra el Ictus](https://www.ictusasociacion.org/)
        
        ### Información Médica
        - [Sociedad Española de Neurología](https://www.sen.es/)
        - [American Stroke Association](https://www.stroke.org/)
        
        ### Líneas de Ayuda
        - Emergencias: 112
        - Teléfono de la Esperanza: 717 003 717
        
        ### Aplicaciones Móviles
        - Stroke Riskometer
        - FAST Test
        
        ### Libros Recomendados
        - "Después del Ictus" por Julio Agredano
        - "Vivir después de un Ictus" por José Álvarez Sabín
        
        ### Grupos de Apoyo
        Consulta con tu hospital local o centro de salud para información sobre grupos de apoyo en tu área.
        """)
    return resources

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css="custom.css") as demo:
    with gr.Tabs() as tabs:
        with gr.TabItem("Inicio"):
            create_home_page()
        with gr.TabItem("Predicción"):
            create_prediction_interface()
        with gr.TabItem("Información"):
            create_info_page()
        with gr.TabItem("Recursos"):
            create_resources_page()

if __name__ == "__main__":
    demo.launch()
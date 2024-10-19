# **Stroke Prediction System**
Este proyecto es un sistema de predicción de ictus que utiliza Machine Learning y una interfaz gráfica basada en Gradio. El sistema permite a dos tipos de usuarios (administradores y usuarios regulares) interactuar con el modelo, registrar encuestas y ver los resultados almacenados en una base de datos MySQL.

Contenido
Características del Proyecto
Requisitos Previos
Estructura del Proyecto
Instalación
Uso del Proyecto
Detalles Técnicos
Dockerización
Contribuciones
Características del Proyecto
Predicción de ictus: Utiliza un modelo de ensamble que combina varios algoritmos de ML como RandomForest, Naive Bayes y SVM para predecir la probabilidad de que una persona sufra un ictus.
Base de datos MySQL: Los datos de las encuestas completadas por los usuarios se guardan en una base de datos MySQL.
Roles de usuario:
Administrador: Puede acceder a informes detallados de las encuestas y gestionar modelos.
Usuario regular: Puede completar una encuesta y obtener la predicción de riesgo de ictus.
Interfaz gráfica con Gradio: Interfaz amigable para los usuarios y administradores.
Automatización del pipeline: La aplicación incluye scripts para entrenar modelos, monitorizar su rendimiento, y registrar experimentos en MLFlow.
Requisitos Previos
Antes de empezar, asegúrate de tener instalado lo siguiente:

Python 3.9+
MySQL
Docker (opcional para contenerización)
Dependencias de Python (ver Instalación)
Estructura del Proyecto
bash
Copiar código
project/
│
├── data/                    # Datos para entrenamiento y pruebas
│   ├── raw/                 # Datos crudos
│   └── processed/           # Datos procesados
│
├── notebooks/               # Notebooks para análisis exploratorio
│   └── EDA_extreme.ipynb
│
├── models/                  # Modelos entrenados
│   ├── model_v1.pkl         # Modelo de RandomForest
│   ├── model_v2.h5          # Modelo de Redes Neuronales
│   └── ensemble_model.pkl   # Modelo de Ensamble
│
├── src/                     # Código fuente del proyecto
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train_model.py       # Entrenamiento de modelos
│   ├── monitor_model.py     # Monitorización del modelo
│   ├── mlflow_tracking.py   # Registro de experimentos en MLFlow
│   └── gradio_app.py        # Interfaz gráfica de Gradio
│
├── tests/                   # Test unitarios
│   └── test_train.py        # Test para entrenamiento
│
├── requirements.txt         # Dependencias
├── Dockerfile               # Contenerización
├── Jenkinsfile              # CI/CD
├── mlflow.yaml              # Configuración para MLFlow
└── README.md                # Documentación
Instalación
Clonar el repositorio
bash
Copiar código
git clone https://github.com/tu-usuario/stroke-prediction-system.git
cd stroke-prediction-system
Instalar dependencias
bash
Copiar código
pip install -r requirements.txt
Configurar MySQL
Asegúrate de tener una instancia de MySQL corriendo y crea la base de datos y tablas necesarias con los siguientes comandos:

sql
Copiar código
CREATE DATABASE stroke_prediction_db;

USE stroke_prediction_db;

CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(100) NOT NULL,
    role ENUM('admin', 'user') NOT NULL DEFAULT 'user'
);

CREATE TABLE surveys (
    survey_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    gender VARCHAR(10),
    age INT,
    hypertension BOOLEAN,
    heart_disease BOOLEAN,
    avg_glucose_level FLOAT,
    smoking_status VARCHAR(20),
    stroke_probability FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
Configurar las credenciales de MySQL
En los scripts de conexión a MySQL (por ejemplo, src/gradio_app.py), asegúrate de actualizar las credenciales de MySQL con tu usuario, contraseña y host de MySQL.

Uso del Proyecto
1. Ejecutar la aplicación con Gradio
Para ejecutar la interfaz gráfica, usa el siguiente comando:

bash
Copiar código
python src/gradio_app.py
2. Interfaz de Usuario
Usuario regular: Puede completar una encuesta ingresando datos como edad, género, hipertensión, entre otros, y recibirá una predicción de la probabilidad de sufrir un ictus.

Administrador: Puede iniciar sesión como administrador y acceder a un informe que muestra todas las encuestas completadas, junto con sus resultados.

3. Entrenar modelos de ML
Si deseas entrenar los modelos, puedes ejecutar los siguientes scripts de Python:

bash
Copiar código
# Entrenar modelos de Machine Learning
python src/train_model.py
Detalles Técnicos
Modelos de Machine Learning: El proyecto utiliza un modelo de ensamble que combina algoritmos como RandomForest, Naive Bayes y SVM.
MLFlow: Se usa para rastrear los experimentos de Machine Learning, como las métricas de rendimiento y los modelos entrenados.
MySQL: Utilizado para almacenar los datos de las encuestas completadas por los usuarios.
Gradio: Framework para construir interfaces gráficas en Python de manera rápida y sencilla.
Dockerización
Para ejecutar la aplicación en un contenedor Docker, sigue estos pasos:

Construir la imagen Docker:
bash
Copiar código
docker build -t stroke-prediction-app .
Ejecutar el contenedor:
bash
Copiar código
docker run -p 7860:7860 stroke-prediction-app
Esto expondrá la aplicación en http://localhost:7860.

Contribuciones
¡Las contribuciones son bienvenidas! Si deseas contribuir a este proyecto, abre un issue o un pull request con tus sugerencias.

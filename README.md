# **Sistema de predicción de ICTUS**

## Descripción del Proyecto
Este proyecto tiene como objetivo desarrollar un prototipo de sistema de predicción de riesgo de ictus para el hospital F5, basado en algoritmos de aprendizaje supervisado. El modelo recibe datos de los pacientes y evalúa el riesgo de ictus mediante una predicción automatizada que servirá como criba inicial antes de la consulta médica. Este enfoque permitirá al personal sanitario identificar pacientes en riesgo y optimizar la gestión de consultas.

## Planteamiento
El hospital F5 ha recopilado un conjunto de datos con diversos indicadores de salud de los pacientes, incluyendo información sobre el historial médico y los hábitos de vida. Estos datos se utilizan para entrenar un modelo de inteligencia artificial que, de manera autónoma y a través de una línea de comandos, solicita la entrada de información del paciente y devuelve una predicción sobre el riesgo de ictus.

### Datos
El dataset incluye las siguientes características:
- Género
- Edad
- Hipertensión
- Enfermedad cardíaca
- Nivel promedio de glucosa
- Índice de masa corporal (IMC)
- Estado de fumador

### Modelo
Se ha implementado un modelo de aprendizaje supervisado para clasificar el riesgo de ictus. El modelo es entrenado con técnicas de preprocesamiento como el balanceo de clases y la ingeniería de características (interacciones y transformaciones de variables) para mejorar la precisión en la predicción de eventos de ictus.

## Arquitectura
El proyecto está dividido en los siguientes módulos:
1. **Preprocesamiento y EDA**: análisis exploratorio de los datos, tratamiento de valores faltantes y codificación de variables categóricas.
2. **Modelado**: entrenamiento y ajuste del modelo usando algoritmos como Regresión Logística, Random Forest, Naves Bayes y SVM. También se ha entrenado un modelo de red neuronal para compararlo con los modelos tradicionales.
3. **API de Predicción**: implementación de una API con FastAPI que recibe los datos de entrada y devuelve la predicción.
4. **Almacenamiento de Predicciones**: integración con una base de datos PostgreSQL para almacenar las predicciones y auditorías.
5. **Monitorización**: registros de las predicciones con MLFlow para seguimiento y análisis de rendimiento.
6. **Autenticación y Autorización**: el directorio `auth` gestiona la autenticación mediante JWT, garantizando acceso seguro a la API.

## Estructura del Proyecto
- `data/`: contiene el script de preprocesamiento y análisis de datos.
- `models/`: almacenamiento de modelos entrenados.
- `api.py`: definición de la API de predicción.
- `database/`: módulos para la conexión y operaciones en la base de datos PostgreSQL.
- `auth/`: gestiona la autenticación y autorización mediante tokens JWT.
- `tests/`: pruebas unitarias y de integración para asegurar el correcto funcionamiento de cada módulo.

## Configuración de la Base de Datos
El proyecto utiliza una base de datos PostgreSQL para almacenar datos de usuarios y las predicciones realizadas. A continuación, se muestra la configuración de las tablas necesarias.

### Creación de las Tablas
Ejecuta las siguientes consultas SQL para crear las tablas `datos_sensibles` y `predicciones`:

```sql
CREATE TABLE datos_sensibles (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(100),
    apellido VARCHAR(100),
    email VARCHAR(100) UNIQUE NOT NULL,
    contrasena VARCHAR(255) NOT NULL,
    telefono VARCHAR(20),
    poblacion VARCHAR(100),
    edad INT,
    gender VARCHAR(10)
);

CREATE TABLE predicciones (
    id SERIAL PRIMARY KEY,
    usuario_id INT REFERENCES datos_sensibles(id),
    gender VARCHAR(10),
    age INT,
    hypertension BOOLEAN,
    heart_disease BOOLEAN,
    avg_glucose_level FLOAT,
    smoking_status VARCHAR(20),
    prediction_result FLOAT,
    fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Configuración de Variables de Entorno
Especifica las credenciales de conexión a la base de datos y la configuración de JWT en el archivo `.env`:

## Instalación
1. Clona el repositorio y navega al directorio del proyecto.
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd PROYECTODATASCIENTIST
   ```
2. Instala los paquetes requeridos.
   ```bash
   pip install -r requirements.txt
   ```
3. Configura las variables de entorno necesarias en un archivo `.env` (ver ejemplo en `.env.example`).

## Ejecución
1. **Iniciar la API de predicción**:
   ```bash
   uvicorn api:app --reload
   ```
2. **Lanzar la interfaz de usuario**:
   ```bash
   python frontend/app.py
   ```
3. **Probar la predicción**: Accede a la interfaz en Gradio y utiliza el modelo para realizar predicciones.

## Evaluación del Rendimiento
Se evaluará el rendimiento del modelo mediante métricas de clasificación y validación cruzada. Un informe adicional se generará para documentar el rendimiento del modelo en el conjunto de validación.

## Contribuciones
Las contribuciones son bienvenidas. Por favor, abre una discusión antes de enviar cambios importantes y asegúrate de realizar pruebas adecuadas.

## Licencia
Este proyecto está bajo la Licencia MIT.

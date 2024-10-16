import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import Memory
import os

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Columnas en el DataFrame:", df.columns)
    print("Primeras filas del DataFrame:")
    print(df.head())
    print(f"Forma del DataFrame: {df.shape}")
    return df

def create_preprocessing_pipeline(df):
    # Identificar columnas numéricas y categóricas
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features = [col for col in numeric_features if col != 'stroke']  # Excluir la variable objetivo
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    print(f"Características numéricas: {numeric_features}")
    print(f"Características categóricas: {categorical_features}")

    # Crear un directorio para el caché si no existe
    cache_dir = os.path.join('data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Crear un objeto Memory para el caché
    memory = Memory(location=cache_dir, verbose=0)

    # Crear los transformadores numéricos y categóricos con memory
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ], memory=memory)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ], memory=memory)

    # Crear el preprocesador con los transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def main():
    try:
        # Cargar datos
        input_file = os.path.join('data', 'clean_stroke_data.csv')
        print(f"Cargando datos desde: {input_file}")
        df = load_data(input_file)

        # Crear y ajustar el pipeline de preprocesamiento
        preprocess_pipeline = create_preprocessing_pipeline(df)
        X = df.drop('stroke', axis=1)  # Variable objetivo es 'stroke'
        y = df['stroke']

        print(f"Forma de X antes del preprocesamiento: {X.shape}")
        print(f"Forma de y: {y.shape}")

        # Aplicar el pipeline de preprocesamiento
        print("Aplicando preprocesamiento...")
        x_preprocessed = preprocess_pipeline.fit_transform(X)

        print(f"Forma de x_preprocessed: {x_preprocessed.shape}")
        print(f"Forma de y: {y.shape}")

        # Asegurarse de que X_preprocessed y y tengan el mismo número de muestras
        if x_preprocessed.shape[0] != y.shape[0]:
            print("Error: El número de muestras en X e y no coincide.")
            return

        # Guardar los datos preprocesados
        processed_dir = os.path.join('data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        x_file = os.path.join(processed_dir, 'x_preprocessed.npy')
        y_file = os.path.join(processed_dir, 'y.npy')

        print(f"Guardando x_preprocessed en: {x_file}")
        np.save(x_file, x_preprocessed)
        print(f"Guardando y en: {y_file}")
        np.save(y_file, y)

        # Verificar que los archivos se hayan guardado
        if os.path.exists(x_file) and os.path.exists(y_file):
            print("Datos preprocesados guardados correctamente.")
            print(f"Tamaño de x_preprocessed.npy: {os.path.getsize(x_file)} bytes")
            print(f"Tamaño de y.npy: {os.path.getsize(y_file)} bytes")
        else:
            print("Error: No se pudieron guardar los archivos.")

    except Exception as e:
        print(f"Error durante el preprocesamiento: {e}")

if __name__ == "__main__":
    main()
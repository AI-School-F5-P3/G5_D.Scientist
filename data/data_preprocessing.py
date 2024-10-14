import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import Memory
import os
import joblib

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Columnas en el DataFrame:", df.columns)
    print("Primeras filas del DataFrame:")
    print(df.head())
    return df

def create_preprocessing_pipeline():
    numeric_features = ['age', 'avg_glucose_level', 'bmi']
    categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

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

    # Crear el pipeline final con el argumento memory especificado
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)], memory=memory)

    return pipeline

def main():
    try:
        # Cargar datos
        input_file = os.path.join('data', 'clean_stroke_data.csv')
        df = load_data(input_file)

        # Crear y ajustar el pipeline de preprocesamiento
        preprocess_pipeline = create_preprocessing_pipeline()
        X = df.drop('stroke', axis=1)
        y = df['stroke']
        preprocess_pipeline.fit(X)

        # Crear directorio para datos procesados si no existe
        processed_dir = os.path.join('data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)

        # Guardar el pipeline ajustado
        joblib.dump(preprocess_pipeline, os.path.join(processed_dir, 'preprocess_pipeline.joblib'))

        # Aplicar el preprocesamiento y guardar los datos preprocesados
        x_preprocessed = preprocess_pipeline.transform(X)
        np.save(os.path.join(processed_dir, 'X_preprocessed.npy'), x_preprocessed)
        np.save(os.path.join(processed_dir, 'y.npy'), y)

        print("Preprocesamiento completado y datos guardados.")
    except Exception as e:
        print(f"Error durante el preprocesamiento: {str(e)}")

if __name__ == "__main__":
    main()


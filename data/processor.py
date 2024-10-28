import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, show
from bokeh.layouts import column
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTETomek
from joblib import load

df_final = load('df_final.joblib')

df = pd.read_csv('stroke_dataset.csv')

columns_to_keep = ['gender', 'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'smoking_status', 'stroke']

df_final = df_final[columns_to_keep]

df = df_final.copy()

def preprocess_data(df):
    """
    Preprocesa los datos usando solo LabelEncoder para variables categóricas
    y StandardScaler para variables numéricas
    """
    # Separar features y target
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    # Definir columnas
    categorical_columns = ['gender', 'smoking_status']
    numeric_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level']
    
    # Crear copia de X para no modificar el original
    X_processed = X.copy()
    
    # Aplicar LabelEncoder a columnas categóricas
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        X_processed[col] = label_encoders[col].fit_transform(X_processed[col])
    
    # Aplicar StandardScaler a columnas numéricas
    scaler = StandardScaler()
    X_processed[numeric_columns] = scaler.fit_transform(X_processed[numeric_columns])
    
    return X_processed, y, label_encoders, scaler

def balance_data(X, y, random_state=42):
    """
    Aplica SMOTE-Tomek para balance sin dividir los datos
    """
    # Aplicar SMOTE-Tomek
    smote_tomek = SMOTETomek(random_state=random_state)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    
    # Convertir a DataFrame manteniendo los nombres de las columnas
    X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    
    return X_resampled_df, y_resampled

def main(df):
    """
    Función principal que ejecuta todo el pipeline de preprocesamiento y balanceo
    """
    # Preprocesar datos
    X_processed, y, label_encoders, scaler = preprocess_data(df)
    
    # Balancear datos
    X_resampled, y_resampled = balance_data(X_processed, y)
    
    # Imprimir información sobre el balanceo
    print("\nDistribución original de clases:")
    print(y.value_counts(normalize=True))
    
    print("\nDistribución de clases después de SMOTE-Tomek:")
    print(pd.Series(y_resampled).value_counts(normalize=True))
    
    print("\nFormas de los conjuntos de datos:")
    print(f"X original: {X_processed.shape}")
    print(f"X resampled: {X_resampled.shape}")
    
    # Combinar X_resampled y y_resampled en un solo DataFrame
    df_resampled = X_resampled.copy()
    df_resampled['stroke'] = y_resampled
    
    # Guardar el DataFrame procesado y balanceado como CSV
    df_resampled.to_csv('stroke_dataset_processed.csv', index=False)
    print("\nDataset procesado y balanceado guardado como 'stroke_dataset_processed.csv'")
    
    return X_resampled, y_resampled, label_encoders, scaler

if __name__ == "__main__":
    # Cargar y ejecutar
    # df = pd.read_csv('tu_archivo.csv')
    X_resampled, y_resampled, label_encoders, scaler = main(df)
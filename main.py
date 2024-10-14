import numpy as np
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
import os
import time
import pandas as pd

# Definir constantes para evitar duplicación de cadenas
BALANCING_METHOD = 'Balancing Method'
F1_SCORE = 'F1 Score'

def load_data():
    """
    Carga los datos de entrenamiento y prueba desde archivos .npy.
    """
    x_train = np.load(os.path.join('data', 'interim', 'x_train.npy'))
    x_test = np.load(os.path.join('data', 'interim', 'x_test.npy'))
    y_train = np.load(os.path.join('data', 'interim', 'y_train.npy'))
    y_test = np.load(os.path.join('data', 'interim', 'y_test.npy'))
    return x_train, x_test, y_train, y_test

def main():
    """
    Función principal para entrenar y evaluar varios modelos de machine learning.
    """
    # Cargar los datos
    x_train, x_test, y_train, y_test = load_data()

    # Inicializamos los modelos a entrenar
    models = [
        LogisticRegressionModel(balance_classes=True),  # Regresión logística con balanceo interno
        RandomForestModel(n_estimators=100, max_depth=10),  # Random Forest con hiperparámetros ajustados
        XGBoostModel()  # XGBoost con búsqueda aleatoria de hiperparámetros
    ]

    # Almacenar los resultados
    results = []

    # Iteramos sobre cada modelo, entrenamos, evaluamos y guardamos los resultados
    for model in models:
        print(f"\nTraining and evaluating {model.model_name}...")

        # Medimos el tiempo de entrenamiento
        start_time = time.time()
        model.train(x_train, y_train)  # Entrenamos el modelo
        training_duration = time.time() - start_time  # Tiempo de entrenamiento

        # Evaluamos el modelo en el conjunto de prueba
        accuracy, f1, auc_roc = model.evaluate(x_test, y_test)

        # Agregar los resultados a la lista
        results.append({
            BALANCING_METHOD: 'None',  # Puedes personalizar este valor si tienes métodos de balanceo
            'Model': model.model_name,
            'Accuracy': accuracy,
            F1_SCORE: f1,
            'AUC-ROC': auc_roc,
            'Training Time (s)': training_duration
        })

        # Guardar el modelo entrenado (sin especificar path, usa el por defecto)
        model.save_model()

    # Guardar resultados en un archivo CSV
    results_df = pd.DataFrame(results)
    output_file = os.path.join('results', 'model_performance.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Crear la carpeta si no existe
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Graficar la importancia de las características para el modelo XGBoost
    xgb_model = [m for m in models if isinstance(m, XGBoostModel)][0]
    feature_names = [f"feature_{i}" for i in range(x_train.shape[1])]  # Nombres ficticios
    xgb_model.plot_feature_importance(feature_names)

if __name__ == "__main__":
    main()


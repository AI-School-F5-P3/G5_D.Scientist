import numpy as np
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel  # Corregido el nombre
from models.xgboost_model import XGBoostModel
import os
import time

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
    x_train, x_test, y_train, y_test = load_data()

    # Inicializamos los modelos
    models = [
        LogisticRegressionModel(balance_classes=True),
        RandomForestModel(n_estimators=100, max_depth=10),  # Hiperparámetros adicionales
        XGBoostModel(learning_rate=0.1, n_estimators=200)
    ]

    # Entrenamos y evaluamos cada modelo
    for model in models:
        print(f"\nTraining and evaluating {model.model_name}:")
        
        # Medir el tiempo de entrenamiento
        start_time = time.time()
        model.train(x_train, y_train)
        training_duration = time.time() - start_time

        # Evaluamos el modelo
        accuracy, report, auc_roc = model.evaluate(x_test, y_test)
        
        # Imprimimos los resultados
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"Training Time: {training_duration:.2f} seconds")
        print(f"Classification Report:\n{report}")
        
        # Guardar el modelo entrenado
        model.save_model()

    # Gráfico de importancia de características para XGBoost
    xgb_model = [m for m in models if isinstance(m, XGBoostModel)][0]
    feature_names = [f"feature_{i}" for i in range(x_train.shape[1])]
    xgb_model.plot_feature_importance(feature_names)

if __name__ == "__main__":
    main()

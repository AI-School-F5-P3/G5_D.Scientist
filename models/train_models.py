import numpy as np
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestClassifier
from models.xgboost_model import XGBoostModel
import os

def load_data():
    x_train = np.load(os.path.join('data', 'interim', 'x_train.npy'))
    x_test = np.load(os.path.join('data', 'interim', 'x_test.npy'))
    y_train = np.load(os.path.join('data', 'interim', 'y_train.npy'))
    y_test = np.load(os.path.join('data', 'interim', 'y_test.npy'))
    return x_train, x_test, y_train, y_test

def main():
    x_train, x_test, y_train, y_test = load_data()

    models = [
        LogisticRegressionModel(),
        RandomForestClassifier(),
        XGBoostModel()
    ]

    for model in models:
        print(f"\nTraining and evaluating {model.model_name}:")
        model.train(x_train, y_train)
        accuracy, f1, auc_roc = model.evaluate(x_test, y_test)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        model.save_model()

    # Generar gráfico de importancia de características para XGBoost
    xgb_model = [m for m in models if isinstance(m, XGBoostModel)][0]
    feature_names = [f"feature_{i}" for i in range(x_train.shape[1])]  # Asume que no tenemos nombres reales de características
    xgb_model.plot_feature_importance(feature_names)

if __name__ == "__main__":
    main()
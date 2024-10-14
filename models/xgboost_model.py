from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__('xgboost')
        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )

    def train(self, x_train, y_train):
        # Definir el espacio de búsqueda de hiperparámetros
        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'min_child_weight': [1, 2, 3, 4, 5]
        }

        # Realizar búsqueda aleatoria de hiperparámetros con validación cruzada
        random_search = RandomizedSearchCV(
            self.model,
            param_distributions=param_dist,
            n_iter=100,
            scoring='f1',
            n_jobs=-1,
            cv=5,
            verbose=1,
            random_state=42
        )

        # Calcular la escala de peso para las clases
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

        # Ajustar el modelo
        random_search.fit(
            x_train, 
            y_train,
            sample_weight=np.where(y_train == 1, scale_pos_weight, 1)
        )

        # Actualizar el modelo con los mejores parámetros
        self.model = random_search.best_estimator_

        print("Mejores parámetros encontrados:")
        print(random_search.best_params_)

        # Evaluar el modelo usando validación cruzada
        cv_scores = cross_val_score(self.model, x_train, y_train, cv=5, scoring='f1')
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

    def evaluate(self, x_test, y_test, x_train=None, y_train=None):
        if x_train is not None and y_train is not None:
            # Evaluación en conjunto de entrenamiento
            y_train_pred = self.model.predict(x_train)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            print(f"Train Accuracy: {train_accuracy:.4f}")

        # Evaluación en conjunto de prueba
        y_test_pred = self.model.predict(x_test)

        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Test Accuracy: {test_accuracy:.4f}")

    def plot_feature_importance(self, feature_names):
        import matplotlib.pyplot as plt
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Importancia de características")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

    def save_model(self):
        import joblib
        joblib.dump(self.model, f'{self.model_name}_model.joblib')
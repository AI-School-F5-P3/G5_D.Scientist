from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self):
        """
        Inicializa el modelo XGBoost con parámetros básicos y desactiva el uso de la codificación de etiquetas.
        """
        super().__init__('xgboost')
        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',  # Métrica para evaluar el rendimiento durante el entrenamiento
            random_state=42
        )

    def train(self, x_train, y_train):
        """
        Entrena el modelo de XGBoost utilizando una búsqueda aleatoria de hiperparámetros con validación cruzada.
        """
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

        # Crear el objeto de búsqueda aleatoria con validación cruzada
        random_search = RandomizedSearchCV(
            self.model,
            param_distributions=param_dist,
            n_iter=100,  # Número de iteraciones para la búsqueda
            scoring='f1',  # Métrica de evaluación
            n_jobs=-1,  # Usar todos los núcleos disponibles
            cv=5,  # Validación cruzada de 5 particiones
            verbose=1,  # Verbo alto para mostrar el progreso
            random_state=42  # Fijar la semilla para reproducibilidad
        )

        # Calcular el peso de la clase positiva para manejar el desbalance de clases
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

        # Ajustar el modelo usando el conjunto de entrenamiento
        random_search.fit(
            x_train, 
            y_train,
            sample_weight=np.where(y_train == 1, scale_pos_weight, 1)  # Asignar pesos para clases desbalanceadas
        )

        # Actualizar el modelo con los mejores hiperparámetros encontrados
        self.model = random_search.best_estimator_

        print("Mejores parámetros encontrados:")
        print(random_search.best_params_)

        # Evaluación mediante validación cruzada
        cv_scores = cross_val_score(self.model, x_train, y_train, cv=5, scoring='f1')
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

    def evaluate(self, x_test, y_test, x_train=None, y_train=None):
        """
        Evalúa el modelo tanto en los datos de prueba como opcionalmente en los de entrenamiento.
        """
        if x_train is not None and y_train is not None:
            # Evaluación en conjunto de entrenamiento
            y_train_pred = self.model.predict(x_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            print(f"Train Accuracy: {train_accuracy:.4f}")

        # Evaluación en conjunto de prueba
        y_test_pred = self.model.predict(x_test)

        # Métricas para el conjunto de prueba
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc_roc = roc_auc_score(y_test, self.model.predict_proba(x_test)[:, 1])

        # Mostrar resultados
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test AUC-ROC: {test_auc_roc:.4f}")
        
        # Imprimir reporte de clasificación completo
        print(f"Classification Report:\n{classification_report(y_test, y_test_pred)}")

        return test_accuracy, test_f1, test_auc_roc

    def plot_feature_importance(self, feature_names):
        """
        Grafica la importancia de las características usando la importancia calculada por XGBoost.
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]  # Ordenar las características por importancia descendente

        # Gráfico de barras de la importancia de características
        plt.figure(figsize=(10, 6))
        plt.title("Importancia de características")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

    def save_model(self, path=None):
    """
    Guarda el modelo entrenado en un archivo .joblib.
    Si no se proporciona una ruta, usa un valor por defecto basado en model_name.
    """
    if not path:
        path = f'{self.model_name}_model.joblib'
    
    # Guardar el modelo usando joblib
    import joblib
    joblib.dump(self.model, path)


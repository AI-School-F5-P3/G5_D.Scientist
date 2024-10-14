from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os

class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None  # Inicializamos el modelo como None

    def train(self, x_train, y_train):
        """
        Método abstracto que debe ser implementado por las clases hijas.
        """
        raise NotImplementedError("Subclasses must implement the train method")

    def evaluate(self, x_test, y_test):
        """
        Evalúa el modelo usando precisión, reporte de clasificación y AUC-ROC si es aplicable.
        """
        y_pred = self.model.predict(x_test)  # Predicciones
        accuracy = accuracy_score(y_test, y_pred)  # Precisión
        report = classification_report(y_test, y_pred)  # Reporte de clasificación
        
        # Si el modelo soporta predict_proba, calculamos el AUC-ROC
        auc_roc = roc_auc_score(y_test, self.model.predict_proba(x_test)[:, 1]) if hasattr(self.model, "predict_proba") else None
        
        return accuracy, report, auc_roc

    def save_model(self, path=None):
        """
        Guarda el modelo entrenado en la carpeta 'models'.
        """
        if not path:
            path = os.path.join('models', f'{self.model_name}_model.joblib')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load_model(self, path):
        """
        Carga un modelo previamente guardado.
        """
        if os.path.exists(path):
            self.model = joblib.load(path)
        else:
            raise FileNotFoundError(f"El archivo {path} no existe.")

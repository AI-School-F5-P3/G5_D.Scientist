from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self, balance_classes=False, **kwargs):
        """
        Inicializa el modelo LogisticRegression.
        Puede aceptar argumentos adicionales (kwargs) y balancear las clases internamente.
        """
        super().__init__('logistic_regression')
        # Si balance_classes es True, activamos el balanceo interno de clases
        if balance_classes:
            kwargs['class_weight'] = 'balanced'
        self.model = LogisticRegression(random_state=42, **kwargs)

    def train(self, x_train, y_train):
        """
        Entrena el modelo con los datos de entrenamiento.
        """
        self.model.fit(x_train, y_train)


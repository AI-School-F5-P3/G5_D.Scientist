from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, balance_classes=False, **kwargs):
        """
        Inicializa el modelo RandomForestClassifier.
        Puede aceptar argumentos adicionales (kwargs) como hiperparámetros y el balanceo de clases.
        """
        super().__init__('random_forest')
        # Si balance_classes es True, activamos el balanceo interno de clases
        if balance_classes:
            kwargs['class_weight'] = 'balanced'
        self.model = RandomForestClassifier(random_state=42, **kwargs)

    def train(self, x_train, y_train):
        """
        Entrena el modelo con los datos de entrenamiento.
        """
        self.model.fit(x_train, y_train)

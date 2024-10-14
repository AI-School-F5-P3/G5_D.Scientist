from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__('random_forest')
        self.model = RandomForestClassifier(random_state=42)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
from sklearn.linear_model import LogisticRegression
from base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__('logistic_regression')
        self.model = LogisticRegression()

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class BaseModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def train(self, X_train, y_train):
        raise NotImplementedError("Subclasses must implement train method")

    def evaluate(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    def save_model(self):
        joblib.dump(self.model, os.path.join('models', f'{self.model_name}_model.joblib'))
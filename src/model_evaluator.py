import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mlflow
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    classification_report, roc_curve, f1_score
)
from utils.logger_config import Logger
from src.model_config import ModelConfig

class ModelEvaluator:
    """Class for model evaluation and metric calculation."""
    
    def __init__(self):
        self.logger = Logger().get_logger()
        self.logger.info("ModelEvaluator inicializado")
        
        # Crear directorios necesarios
        self.model_dir = 'models/saved_models'
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    
    @staticmethod
    def find_optimal_threshold(y_true, y_pred_proba):
        try:
            thresholds = np.linspace(0, 1, 100)
            f1_scores = [f1_score(y_true, y_pred_proba >= threshold) for threshold in thresholds]
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            Logger().get_logger().info(f"Optimal threshold found: {optimal_threshold}")
            return optimal_threshold
        except Exception as e:
            Logger().get_logger().error(f"Error finding optimal threshold: {str(e)}", exc_info=True)
            raise
    
    def evaluate_model(self, model, X, y, dataset_name):
        try:
            self.logger.info(f"Evaluating model on {dataset_name} set")
            y_pred_proba = model.predict_proba(X)[:, 1]
            optimal_threshold = self.find_optimal_threshold(y, y_pred_proba)
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'auc_roc': roc_auc_score(y, y_pred_proba),
                'brier_score': brier_score_loss(y, y_pred_proba)
            }
            
            self.logger.info(f"{dataset_name} Set Performance:")
            for metric_name, value in metrics.items():
                self.logger.info(f"{metric_name}: {value:.4f}")
                mlflow.log_metric(f"{dataset_name.lower()}_{metric_name}", value)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}", exc_info=True)
            raise
    
    def plot_feature_importance(self, ensemble, feature_names):
        try:
            self.logger.info("Generating feature importance plot")
            rf_model = ensemble.named_estimators_['rf']
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(f'{self.model_dir}/feature_importance.png')
            plt.close()
            
            self.logger.info("Feature importance plot saved")
            
        except Exception as e:
            self.logger.error(f"Error plotting feature importance: {str(e)}", exc_info=True)
            raise
    
    def plot_roc_curve(self, y_true, y_pred_proba, auc_score):
        try:
            self.logger.info("Generando gráfico de curva ROC")
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            # Asegurarse de que auc_score sea un número, no un diccionario
            if isinstance(auc_score, dict):
                auc_value = auc_score.get('auc_roc', 0.0)
            else:
                auc_value = auc_score
                
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_value:.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig(f'{self.model_dir}/roc_curve.png')
            plt.close()
            
            self.logger.info("Gráfico de curva ROC guardado")
        except Exception as e:
            self.logger.error(f"Error plotting ROC curve: {str(e)}")
            raise

    def save_classification_report(self, model, X_test, y_test):
        try:
            self.logger.info("Generando reporte de clasificación")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            report = classification_report(y_test, y_pred, output_dict=True)
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            
            report_text = f"Informe del modelo:\n\n"
            report_text += f"Precisión: {report['accuracy']:.2f}\n"
            report_text += f"Recall (clase 1): {report['1']['recall']:.2f}\n"
            report_text += f"F1-Score (clase 1): {report['1']['f1-score']:.2f}\n"
            report_text += f"AUC-ROC: {auc_roc:.2f}\n"
            
            # Usar la ruta correcta
            report_path = f'{self.model_dir}/model_report.txt'
            with open(report_path, 'w') as f:
                f.write(report_text)
                
            self.logger.info("Reporte de clasificación guardado")
            
        except Exception as e:
            self.logger.error(f"Error generando reporte de clasificación: {str(e)}")
            raise
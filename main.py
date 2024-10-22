# main.py
import mlflow
import joblib
from pathlib import Path
import numpy as np
from sklearn.model_selection import cross_val_score
from src.model_config import ModelConfig
from src.data_processor import DataProcessor
from src.model_builder import ModelBuilder
from src.model_evaluator import ModelEvaluator
from sklearn.metrics import classification_report, roc_auc_score
from utils.logger_config import Logger


def main():
    """Main training script."""
    logger = Logger().get_logger()
    logger.info("Inicio del proceso de predicción de ICTUS")
    
    try:
        # Initialize configuration
        config = ModelConfig()
        logger.info(f"Experiment name: {config.experiment_name}")
    
        # Set up MLflow experiment
        mlflow.set_experiment(config.experiment_name)
        logger.info("MLflow configuración del experimento")
        
        with mlflow.start_run():
            # Process data
            logger.info("Iniciar el procesamiento de datos")
            data_processor = DataProcessor(config)
            X_train, X_val, X_test, y_train, y_val, y_test = data_processor.load_and_preprocess(
                Path("data/stroke_dataset_processed.csv")
            )
            #G5_D.Scientist\data\processed\stroke_dataset_processed.csv
            
            # Train and evaluate model
            logger.info("Comenzando la construcción del modelo")
            model_builder = ModelBuilder(config)
            grid_search = model_builder.perform_grid_search(X_train, y_train)
            ensemble = model_builder.create_ensemble(grid_search.best_estimator_)
            
            # Train ensemble
            logger.info("Modelo ensamble de entrenamiento")
            ensemble.fit(X_train, y_train)
            
            # Evaluate modelo
            logger.info("Empezando evaluación del modelos")
            evaluator = ModelEvaluator()
            # Obtener métricas como diccionarios
            train_metrics = evaluator.evaluate_model(ensemble, X_train, y_train, "Training")
            val_metrics = evaluator.evaluate_model(ensemble, X_val, y_val, "Validation")
            test_metrics = evaluator.evaluate_model(ensemble, X_test, y_test, "Test")

            # Calcular el overfitting (entre entrenamiento y prueba)
            print("\nOverfitting Metrics:")
            print(f"Accuracy Overfitting (Train - Test): {train_metrics['accuracy'] - test_metrics['accuracy']:.4f}")
            print(f"AUC Overfitting (Train - Test): {train_metrics['auc_roc'] - test_metrics['auc_roc']:.4f}")
            print(f"Brier Score Difference (Train - Test): {train_metrics['brier_score'] - test_metrics['brier_score']:.4f}")

            # Validación cruzada del ensemble
            logger.info("Comenzando validación cruzada")
            cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='roc_auc')
            print(f"\nCross-validation ROC-AUC scores: {cv_scores}")
            print(f"Mean CV ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

            # Registrar parámetros en MLflow
            mlflow.log_params(grid_search.best_params_)

            # Registrar el modelo en MLflow
            mlflow.sklearn.log_model(ensemble, "ensemble_model")

            # Visualizar la importancia de las características usando Random Forest
            logger.info("Generar gráfico de importancia de características")
            evaluator.plot_feature_importance(ensemble, X_train.columns)

            # Visualizar la curva ROC
            logger.info("Generar gráfico de curva ROC")
            y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
            evaluator.plot_roc_curve(y_test, y_pred_proba, test_metrics)
            
            # Registrar artefactos en MLflow
            mlflow.log_artifact("models/saved_models/feature_importance.png")
            mlflow.log_artifact("models/saved_models/roc_curve.png")

            # Generar y guardar el informe de clasificación
            logger.info("Generar informe de clasificación y guardarlo")
            evaluator.save_classification_report(ensemble, X_test, y_test)

            
            # Generate plots
            logger.info("Generando gráficos de visualización")
            evaluator.plot_feature_importance(ensemble, X_train.columns)
            evaluator.plot_roc_curve(
                y_test, 
                ensemble.predict_proba(X_test)[:, 1],
                test_metrics['auc_roc']
            )
            
            # Log artifacts and parameters
            logger.info("Registro de artefactos y parámetros de MLflow")

            mlflow.log_params(grid_search.best_params_)
            mlflow.log_artifact("models/saved_models/feature_importance.png")
            mlflow.log_artifact("models/saved_models/roc_curve.png")
            mlflow.sklearn.log_model(ensemble, "ensemble_model")
            
            # Save models
            logger.info("Guardando modelos")
            joblib.dump(ensemble, 'models/saved_models/best_ensemble_model.joblib')
            #joblib.dump(data_processor.label_encoder, 'models/saved_models/label_encoder.joblib')

            logger.info("Pipeline completado con éxito")
            
    except Exception as e:
        logger.error("Pipeline falló con error", exc_info=True)
        raise
    
    finally:
        logger.info("Pipeline ejecución terminada")
if __name__ == "__main__":
    main()
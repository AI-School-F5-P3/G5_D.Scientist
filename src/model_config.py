from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from utils.logger_config import Logger

@dataclass
class ModelConfig:
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.25
    cv_folds: int = 5
    experiment_name: str = "Stroke Prediction Experiment"
    logger = Logger().get_logger()
    
    # Model specific parameters
    lr_param_grid: Dict[str, Any] = None
    rf_params: Dict[str, Any] = None
    svm_params: Dict[str, Any] = None
    
    def __post_init__(self):
        self.logger.info("Initializing ModelConfig with parameters")
        self.lr_param_grid = {
            'classifier__C': np.logspace(-3, -1, 20),
            'classifier__penalty': ['l1', 'l2', 'elasticnet'],
            'classifier__solver': ['saga'],
            'classifier__l1_ratio': np.linspace(0, 1, 5)
        }
        
        self.rf_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_leaf': 5,
            'random_state': self.random_state
        }
        
        self.svm_params = {
            'kernel': 'linear',
            'probability': True,
            'random_state': self.random_state,
            'C': 0.1
        }
        self.logger.debug(f"Model parameters initialized: LR grid: {self.lr_param_grid}, "
                         f"RF params: {self.rf_params}, SVM params: {self.svm_params}")

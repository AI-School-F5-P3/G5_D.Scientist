from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from utils.logger_config import Logger
from src.model_config import ModelConfig

class ModelBuilder:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = Logger().get_logger()
        self.logger.info("ModelBuilder initialized")
        
    def create_base_pipeline(self):
        try:
            self.logger.info("Creating base pipeline")
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('sampler', ADASYN(random_state=self.config.random_state)),
                ('classifier', LogisticRegression(random_state=self.config.random_state, max_iter=1000))
            ])
            self.logger.debug("Base pipeline created successfully")
            return pipeline
        except Exception as e:
            self.logger.error(f"Error creating base pipeline: {str(e)}", exc_info=True)
            raise
    
    def perform_grid_search(self, X_train, y_train):
        try:
            self.logger.info("Starting grid search")
            pipeline = self.create_base_pipeline()
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds, 
                shuffle=True, 
                random_state=self.config.random_state
            )
            
            grid_search = GridSearchCV(
                pipeline, 
                self.config.lr_param_grid, 
                cv=cv, 
                scoring='roc_auc', 
                n_jobs=-1
            )
            
            self.logger.info("Fitting grid search")
            grid_search.fit(X_train, y_train)
            
            self.logger.info(f"Best parameters found: {grid_search.best_params_}")
            self.logger.debug(f"Best score: {grid_search.best_score_}")
            
            return grid_search
            
        except Exception as e:
            self.logger.error(f"Error in grid search: {str(e)}", exc_info=True)
            raise
    
    def create_ensemble(self, best_lr):
        try:
            self.logger.info("Creating ensemble model")
            ensemble = VotingClassifier(
                estimators=[
                    ('lr', best_lr),
                    ('nb', GaussianNB()),
                    ('svm', SVC(**self.config.svm_params)),
                    ('rf', RandomForestClassifier(**self.config.rf_params))
                ],
                voting='soft'
            )
            self.logger.debug("Ensemble model created successfully")
            return ensemble
        except Exception as e:
            self.logger.error(f"Error creating ensemble: {str(e)}", exc_info=True)
            raise
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.model_config import ModelConfig
from sklearn.model_selection import train_test_split
from utils.logger_config import Logger

class DataProcessor:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = Logger().get_logger()
        self.gender_encoder = LabelEncoder()
        self.smoking_encoder = LabelEncoder()
        self.logger.info("DataProcessor initialized")
        
        # Definir las categorías conocidas
        self.gender_categories = ['Male', 'Female']
        self.smoking_categories = ['never smoked', 'formerly smoked', 'smokes']
        
        # Pre-entrenar los encoders con todas las categorías posibles
        self.gender_encoder.fit(self.gender_categories)
        self.smoking_encoder.fit(self.smoking_categories)
        
        self.logger.info("Label encoders pre-entrenados con categorías conocidas")
    
    def load_and_preprocess(self, filepath: str):
        try:
            self.logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            self.logger.debug(f"Data loaded with shape: {df.shape}")
            
            X = df.drop('stroke', axis=1)
            y = df['stroke']
            
            self.logger.info("Starting feature preprocessing")
            X = self.preprocess_features(X)
            
            self.logger.info("Feature preprocessing completed")
            self.logger.debug(f"Final feature set: {list(X.columns)}")
            
            return self.split_data(X, y)
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {str(e)}", exc_info=True)
            raise
    
    def preprocess_features(self, X):
        """Preprocesa las características aplicando encoding y feature engineering."""
        try:
            # Encode categorical variables
            X['gender'] = self.gender_encoder.transform(X['gender'])
            X['smoking_status'] = self.smoking_encoder.transform(X['smoking_status'])
            
            # Feature engineering
            X['age_squared'] = X['age'] ** 2
            X['glucose_age_interaction'] = X['age'] * X['avg_glucose_level']
            
            return X
        except Exception as e:
            self.logger.error(f"Error in feature preprocessing: {str(e)}", exc_info=True)
            raise
    
    def preprocess_single_sample(self, input_data: dict) -> pd.DataFrame:
        """Preprocesa un único ejemplo para predicción."""
        try:
            self.logger.info("Preprocesando datos de entrada individual")
            
            # Convertir input_data a DataFrame
            df = pd.DataFrame([input_data])
            
            # Aplicar las mismas transformaciones que en el entrenamiento
            df['gender'] = self.gender_encoder.transform([input_data['gender']])[0]
            df['smoking_status'] = self.smoking_encoder.transform([input_data['smoking_status']])[0]
            
            # Feature engineering
            df['age_squared'] = df['age'] ** 2
            df['glucose_age_interaction'] = df['age'] * df['avg_glucose_level']
            
            self.logger.debug(f"Datos preprocesados exitosamente: {df.to_dict('records')[0]}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error en el preprocesamiento de datos individuales: {str(e)}", exc_info=True)
            raise
        
    def split_data(self, X, y):
        """Divide los datos en conjuntos de entrenamiento, validación y prueba."""
        try:
            self.logger.info("Dividiendo los datos en conjuntos de entrenamiento, validación y prueba")
            
            # Primero dividimos en conjunto de entrenamiento y conjunto de prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config.test_size, random_state=self.config.random_state, stratify=y)
            
            # Luego dividimos el conjunto de entrenamiento en entrenamiento y validación
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.config.validation_size, random_state=self.config.random_state, stratify=y_train)
            
            self.logger.info("Datos divididos correctamente")
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        except Exception as e:
            self.logger.error(f"Error dividiendo los datos: {str(e)}", exc_info=True)
            raise

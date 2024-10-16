import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from models import RandomForestModel, LogisticRegressionModel, XGBoostModel

# Definir constantes para evitar duplicación de cadenas
BALANCING_METHOD = 'Balancing Method'
F1_SCORE = 'F1 Score'



def balance_data(x, y, method):
    balancing_methods = {
        'none': lambda x, y: (x, y),
        'smote': SMOTE(random_state=42),
        'adasyn': ADASYN(random_state=42),
        'random_under': RandomUnderSampler(random_state=42),
        'tomek': TomekLinks(),
        'smote_tomek': SMOTETomek(random_state=42),
        'smoteenn': SMOTEENN(random_state=42)
    }
    
    if method not in balancing_methods:
        raise ValueError(f"Método de balanceo '{method}' no reconocido")
    
    balancer = balancing_methods[method]
    return balancer.fit_resample(x, y) if method != 'none' else balancer(x, y)

def train_and_evaluate(x_train, x_test, y_train, y_test, model):
    start_time = time.time()
    model.train(x_train, y_train)
    y_pred = model.model.predict(x_test)
    
    # Métricas de evaluación
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Verificar si el modelo soporta predict_proba (para calcular AUC-ROC)
    if hasattr(model.model, "predict_proba"):
        auc_roc = roc_auc_score(y_test, model.model.predict_proba(x_test)[:, 1])
    else:
        auc_roc = None
    
    duration = time.time() - start_time
    return accuracy, f1, recall, auc_roc, duration

def main():
    # Cargar datos preprocesados
    X = np.load(os.path.join('data', 'processed', 'x_preprocessed.npy'))
    y = np.load(os.path.join('data', 'processed', 'y_train.npy'))  # Asegúrate de que y_train sea la variable objetivo

    
    # Dividir en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Definir métodos de balanceo externo a probar
    balancing_methods = ['none', 'smote', 'adasyn', 'random_under', 'tomek', 'smote_tomek', 'smoteenn']
    
    # Definir modelos con y sin balanceo interno
    models = {
    'Logistic Regression (no balance)': LogisticRegressionModel(balance_classes=False),
    'Logistic Regression (balanced)': LogisticRegressionModel(balance_classes=True),
    'Random Forest (no balance)': RandomForestModel(balance_classes=False),
    'Random Forest (balanced)': RandomForestModel(balance_classes=True),
    'XGBoost (default)': XGBClassifier(),
    'XGBoost (scale_pos_weight)': XGBClassifier(scale_pos_weight=1)  # Ajusta este valor según tu relación de clases
}
    
    
    # Almacenar resultados
    results = []
    
    # Probar balanceo externo (aparte) y modelos
    for method in balancing_methods:
        print(f"Balancing method: {method}")
        x_train_balanced, y_train_balanced = balance_data(x_train, y_train, method)
        
        for model_name, model in models.items():
            print(f"  Training {model_name} with external balancing...")
            accuracy, f1, recall, auc_roc, duration = train_and_evaluate(x_train_balanced, x_test, y_train_balanced, y_test, model)
            results.append({
                BALANCING_METHOD: method,
                'Model': model_name,
                'Accuracy': accuracy,
                F1_SCORE: f1,
                'Recall': recall,
                'AUC-ROC': auc_roc,
                'Training Time (s)': duration
            })
    
    # Probar los modelos con balanceo interno
    for model_name, model in models.items():
        print(f"  Training {model_name} with internal balancing...")
        accuracy, f1, recall, auc_roc, duration = train_and_evaluate(x_train, x_test, y_train, y_test, model)
        results.append({
            BALANCING_METHOD: 'internal',
            'Model': model_name,
            'Accuracy': accuracy,
            F1_SCORE: f1,
            'Recall': recall,
            'AUC-ROC': auc_roc,
            'Training Time (s)': duration
        })
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame(results)
    
    # Guardar resultados
    output_file = os.path.join('results', 'balancing_experiments.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Crear el directorio si no existe
    results_df.to_csv(output_file, index=False)
    
    print(f"Results saved to {output_file}")
    print(results_df)

    # Visualización de resultados 
    plt.figure(figsize=(12, 6))
    sns.barplot(x=BALANCING_METHOD, y=F1_SCORE, hue='Model', data=results_df)
    plt.title('Comparación de F1 Score por Método de Balanceo y Modelo')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


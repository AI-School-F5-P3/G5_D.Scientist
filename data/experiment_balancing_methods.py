import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
import xgboost as xgb
import os
import pandas as pd

def balance_data(x, y, method):
    if method == 'none':
        return x, y
    elif method == 'smote':
        return SMOTE(random_state=42).fit_resample(x, y)
    elif method == 'adasyn':
        return ADASYN(random_state=42).fit_resample(x, y)
    elif method == 'random_under':
        return RandomUnderSampler(random_state=42).fit_resample(x, y)
    elif method == 'tomek':
        return TomekLinks().fit_resample(x, y)
    elif method == 'smote_tomek':
        return SMOTETomek(random_state=42).fit_resample(x, y)
    elif method == 'smoteenn':
        return SMOTEENN(random_state=42).fit_resample(x, y)
    else:
        raise ValueError("Método de balanceo no reconocido")

def train_and_evaluate(x_train, x_test, y_train, y_test, model):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
    return accuracy, f1, auc_roc

def main():
    # Cargar datos preprocesados
    X = np.load(os.path.join('data', 'processed', 'X_preprocessed.npy'))
    y = np.load(os.path.join('data', 'processed', 'y.npy'))
    
    # Dividir en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Definir métodos de balanceo a probar
    balancing_methods = ['none', 'smote', 'adasyn', 'random_under', 'tomek', 'smote_tomek', 'smoteenn']
    
    # Definir modelos a probar
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42)
    }
    
    # Almacenar resultados
    results = []
    
    # Iterar sobre métodos de balanceo y modelos
    for method in balancing_methods:
        print(f"Balancing method: {method}")
        x_train_balanced, y_train_balanced = balance_data(x_train, y_train, method)
        
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            accuracy, f1, auc_roc = train_and_evaluate(x_train_balanced, x_test, y_train_balanced, y_test, model)
            results.append({
                'Balancing Method': method,
                'Model': model_name,
                'Accuracy': accuracy,
                'F1 Score': f1,
                'AUC-ROC': auc_roc
            })
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame(results)
    
    # Guardar resultados
    output_file = os.path.join('results', 'balancing_experiments.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Crear el directorio si no existe
    results_df.to_csv(output_file, index=False)
    
    print(f"Results saved to {output_file}")
    print(results_df)

if __name__ == "__main__":
    main()
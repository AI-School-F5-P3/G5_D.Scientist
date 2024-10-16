import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
from models import RandomForestModel, LogisticRegressionModel, XGBoostModel

def load_data():
    X = np.load(os.path.join('data', 'processed', 'X_preprocessed.npy'))
    y = np.load(os.path.join('data', 'processed', 'y.npy'))
    return X, y

def train_and_evaluate(x_train, x_test, y_train, y_test, model):
    model.train(x_train, y_train)
    y_pred = model.model.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    if hasattr(model.model, "predict_proba"):
        auc_roc = roc_auc_score(y_test, model.model.predict_proba(x_test)[:, 1])
    else:
        auc_roc = None
    
    return accuracy, f1, recall, auc_roc

def save_results_to_csv(results_df, filename):
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    results_df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

def plot_results(results_df):
    metrics = ['Accuracy', 'F1 Score', 'Recall', 'AUC-ROC']
    n_metrics = len(metrics)
    n_models = len(results_df)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    bar_width = 0.25
    index = np.arange(n_models)
    
    for i, metric in enumerate(metrics):
        axes[i].bar(index, results_df[metric], bar_width, label=metric)
        axes[i].set_xlabel('Models')
        axes[i].set_ylabel(metric)
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_xticks(index)
        axes[i].set_xticklabels(results_df['Model'], rotation=45)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(os.path.join('results', 'model_comparison.png'))
    print("Comparison plot saved to results/model_comparison.png")

def main():
    # Cargar los datos
    X, y = load_data()
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Datos cargados y divididos exitosamente.")
    print(f"Forma de x_train: {x_train.shape}")
    print(f"Forma de x_test: {x_test.shape}")
    print(f"Forma de y_train: {y_train.shape}")
    print(f"Forma de y_test: {y_test.shape}")
    
    # Definir modelos
    models = {
        'Logistic Regression': LogisticRegressionModel(balance_classes=True),
        'Random Forest': RandomForestModel(balance_classes=True),
        'XGBoost': XGBoostModel(balance_classes=True)
    }
    
    # Entrenar y evaluar modelos
    results = []
    for model_name, model in models.items():
        print(f"\nEntrenando y evaluando {model_name}...")
        accuracy, f1, recall, auc_roc = train_and_evaluate(x_train, x_test, y_train, y_test, model)
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'F1 Score': f1,
            'Recall': recall,
            'AUC-ROC': auc_roc
        })
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame(results)
    
    # Mostrar resultados en la terminal
    print("\nResultados de los modelos:")
    print(results_df)
    
    # Guardar resultados en CSV
    save_results_to_csv(results_df, 'model_results.csv')
    
    # Crear y guardar visualización
    plot_results(results_df)

if __name__ == "__main__":
    main()


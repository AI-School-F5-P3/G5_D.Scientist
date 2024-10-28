
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import csv
import os
import joblib
from datetime import datetime
from tabulate import tabulate

# Configurar las rutas y cargar datos
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', '..', 'data', 'stroke_dataset_processed.csv')
    df = pd.read_csv(data_path)
    print("Datos cargados exitosamente")
except FileNotFoundError:
    print("Error: No se puede encontrar el archivo CSV")
    sys.exit(1)

# Crear un directorio para almacenar los resultados con marca de tiempo
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(current_dir, f'results_{timestamp}')
os.makedirs(results_dir, exist_ok=True)
print(f"Directorio de resultados creado en: {results_dir}")

# Preprocesamiento de variables categóricas
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Separar características (X) y variable objetivo (y)
X = df.drop('stroke', axis=1)
y = df['stroke']

# Ingeniería de características adicionales
X['age_squared'] = X['age'] ** 2
X['glucose_age_interaction'] = X['age'] * X['avg_glucose_level']

# Normalización de datos numéricos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar los label encoders y scaler
joblib.dump(label_encoders, os.path.join(results_dir, 'label_encoders.joblib'))
joblib.dump(scaler, os.path.join(results_dir, 'scaler.joblib'))

# Definición del modelo de red neuronal
class RedNeuronal(nn.Module):
    def __init__(self, input_size):
        super(RedNeuronal, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Clase personalizada para manejar el conjunto de datos
class StrokeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values.reshape(-1, 1))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def evaluate_model(model, data_loader, device, dataset_name, results_dir):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (outputs >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_preds)
    auc_roc = roc_auc_score(all_labels, all_probs)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    # Crear y guardar la matriz de confusión
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix - {dataset_name} Set")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{dataset_name.lower()}_confusion_matrix.png'))
    plt.close()
    
    return accuracy, auc_roc, recall, f1

# Preparar el conjunto de datos para entrenamiento, validación y prueba
dataset = StrokeDataset(X_scaled, y)
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Configuración de DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Configuración del dispositivo y modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = X.shape[1]
model = RedNeuronal(input_size).to(device)

# Configuración de entrenamiento
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 10
patience_counter = 0

# Entrenamiento del modelo con Early Stopping
print("\nIniciando entrenamiento...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Validación del modelo
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
    
    train_losses.append(epoch_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Training Loss: {train_losses[-1]:.4f}")
    print(f"Validation Loss: {val_losses[-1]:.4f}")
    
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(results_dir, 'best_stroke_model.pth'))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

print("\nEntrenamiento completado")

# Función para crear una tabla HTML con los resultados
def create_results_table(metrics_data):
    headers = ['Dataset', 'Accuracy', 'ROC-AUC', 'Recall', 'F1-Score']
    table_data = []
    
    for dataset_name, metrics in metrics_data:
        table_data.append([
            dataset_name,
            f"{metrics[0]:.4f}",
            f"{metrics[1]:.4f}",
            f"{metrics[2]:.4f}",
            f"{metrics[3]:.4f}"
        ])
    
    return tabulate(table_data, headers=headers, tablefmt='html')

# Realizar evaluación final y guardar resultados
print("\nRealizando evaluación final...")
train_metrics = evaluate_model(model, train_loader, device, "Training", results_dir)
val_metrics = evaluate_model(model, val_loader, device, "Validation", results_dir)
test_metrics = evaluate_model(model, test_loader, device, "Test", results_dir)

# Crear y guardar la tabla de resultados
metrics_data = [
    ("Training", train_metrics),
    ("Validation", val_metrics),
    ("Test", test_metrics)
]

# Guardar resultados en HTML
results_html = f"""
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .results-container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .metrics-table {{
            margin-bottom: 30px;
        }}
        .visualizations {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <div class="results-container">
        <h2>Model Evaluation Results</h2>
        <div class="metrics-table">
            {create_results_table(metrics_data)}
        </div>
        
        <h2>Visualizations</h2>
        <div class="visualizations">
            <div>
                <h3>Training and Validation Loss</h3>
                <img src="loss_curves.png" alt="Loss Curves">
            </div>
            <div>
                <h3>Training Set Confusion Matrix</h3>
                <img src="training_confusion_matrix.png" alt="Training Confusion Matrix">
            </div>
            <div>
                <h3>Validation Set Confusion Matrix</h3>
                <img src="validation_confusion_matrix.png" alt="Validation Confusion Matrix">
            </div>
            <div>
                <h3>Test Set Confusion Matrix</h3>
                <img src="test_confusion_matrix.png" alt="Test Confusion Matrix">
            </div>
        </div>
    </div>
</body>
</html>
"""

# Guardar el archivo HTML
with open(os.path.join(results_dir, 'results_summary.html'), 'w') as f:
    f.write(results_html)

# Visualización de las pérdidas de entrenamiento y validación
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
plt.close()

print(f"\nResultados guardados en: {results_dir}")
print("- results_summary.html: Tabla de métricas y visualizaciones")
print("- loss_curves.png: Gráfico de pérdidas")
print("- *_confusion_matrix.png: Matrices de confusión para cada conjunto de datos")
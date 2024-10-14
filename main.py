import subprocess
import sys
import os

def run_script(script_path):
    print(f"Running {script_path}...")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_path}:")
        print(result.stderr)
        return False
    else:
        print(f"{script_path} completed successfully.")
        return True

def main():
    # Definir la ruta base del proyecto
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Definir las rutas de los scripts
    data_preprocessing_script = os.path.join(base_path, "data", "data_preprocessing.py")
    experiment_balancing_script = os.path.join(base_path, "data", "experimenting_balancing_methods.py")
    train_models_script = os.path.join(base_path, "models", "train_models.py")

    # Ejecutar data_preprocessing.py
    if not run_script(data_preprocessing_script):
        print("Data preprocessing failed. Stopping execution.")
        return

    # Ejecutar experimenting_balancing_methods.py
    if not run_script(experiment_balancing_script):
        print("Balancing methods experiment failed. Stopping execution.")
        return

    # Ejecutar train_models.py
    if not run_script(train_models_script):
        print("Model training failed.")
        return

    print("All processes completed successfully.")

if __name__ == "__main__":
    main()
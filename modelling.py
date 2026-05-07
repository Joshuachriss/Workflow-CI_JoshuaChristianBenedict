import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

# ========================
# KONFIGURASI
# ========================
DATA_PATH = "iris_preprocessing/iris_preprocessing.csv"
EXPERIMENT_NAME = "iris-classification-basic"
MODEL_NAME = "RandomForestClassifier"


def load_data(path: str):
    """Load dan split dataset."""
    df = pd.read_csv(path)
    X = df.drop('species', axis=1)
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data dimuat: {df.shape}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train():
    """Melatih model RandomForest menggunakan MLflow autolog."""
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    # Aktifkan autolog sebelum run dimulai
    mlflow.sklearn.autolog()

    # Cek apakah dijalankan via `mlflow run .` (ada MLFLOW_RUN_ID di env)
    existing_run_id = os.environ.get("MLFLOW_RUN_ID")

    if existing_run_id:
        # Via mlflow run . -> sambungkan ke run yang sudah dibuat MLflow Project
        run_context = mlflow.start_run(run_id=existing_run_id)
    else:
        # Standalone: python modelling.py -> buat run baru
        mlflow.set_experiment(EXPERIMENT_NAME)
        run_context = mlflow.start_run(run_name="rf_autolog")

    with run_context as run:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"\n=== Hasil Evaluasi ===")
        print(f"Accuracy  : {acc:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred,
              target_names=['setosa', 'versicolor', 'virginica']))
        print(f"\nRun ID    : {run.info.run_id}")
        print(f"Experiment: {EXPERIMENT_NAME}")

        with open("run_id.txt", "w") as f:
                f.write(run.info.run_id)
        print("run_id.txt tersimpan.")


if __name__ == "__main__":
    print("Memulai training model (Basic - Autolog)...")
    train()
    print("\nSelesai. Jalankan: mlflow ui")
    print("Buka browser: http://127.0.0.1:5000")

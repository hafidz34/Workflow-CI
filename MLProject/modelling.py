import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import argparse

def load_data(train_path, test_path):
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print("Data train dan test berhasil dimuat.")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error: File data tidak ditemukan - {e}")
        return None, None
    except Exception as e:
        print(f"Error saat memuat data: {e}")
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    N_ESTIMATORS = args.n_estimators
    RANDOM_STATE = args.random_state

    # Path data
    TRAIN_DATA_PATH = os.path.join("loan_approval_preprocessing", "loan_approval_train_preprocessing.csv")
    TEST_DATA_PATH = os.path.join("loan_approval_preprocessing", "loan_approval_test_preprocessing.csv")

    train_df, test_df = load_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    if train_df is None or test_df is None:
        exit(1)

    target = 'loan_approved'
    try:
        X_train = train_df.drop(columns=[target])
        y_train = train_df[target]
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target]
        print("Fitur dan target berhasil dipisahkan.")
    except KeyError:
        print(f"Error: Kolom target '{target}' tidak ditemukan.")
        exit(1)

    # DISABLE autolog untuk kontrol penuh
    mlflow.sklearn.autolog(disable=True)
    print("MLflow autolog disabled - using manual logging.")

    with mlflow.start_run(run_name="CI_Run") as run:
        
        run_id = run.info.run_id
        print(f"Logging ke active run: {run_id}")
        
        # Training
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
        print(f"Model RandomForestClassifier diinisialisasi (n_estimators={N_ESTIMATORS}).")
        
        # Log parameters
        mlflow.log_params({
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE
        })
        
        print("Melatih model...")
        model.fit(X_train, y_train)
        print("Model selesai dilatih.")
        
        print("Melakukan prediksi pada test set...")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print("Metrik Evaluasi pada Test Set:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        # Log metrics
        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": f1
        })
        
        # Log model dengan path eksplisit
        print("Menyimpan model ke artifacts/model/...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",  # Ini akan create folder 'model' di artifacts
            signature=mlflow.models.infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.iloc[:5]
        )
        
        print(f"✅ Run selesai. Model tersimpan di artifacts/model")
        print(f"Run ID: {run_id}")
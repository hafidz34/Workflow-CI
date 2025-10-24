import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import argparse
import json

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
    parser.add_argument("--experiment_name", type=str, default="Loan Approval CI")
    parser.add_argument("--run_name", type=str, default="CI Run")
    args = parser.parse_args()

    N_ESTIMATORS = args.n_estimators
    RANDOM_STATE = args.random_state
    EXPERIMENT_NAME = args.experiment_name
    RUN_NAME = args.run_name

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
    except Exception as e:
        print(f"Error saat memisahkan fitur/target: {e}")
        exit(1)

    # Start an explicit mlflow run so we can get run_id easily
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=RUN_NAME) as run:
        run_id = run.info.run_id
        print(f"MLflow run started: {run_id}")

        # Enable autolog (still useful)
        mlflow.sklearn.autolog(
            log_model_signatures=True,
            log_input_examples=True,
            log_models=True,
            disable=False
        )
        print("MLflow autolog diaktifkan.")

        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
        print(f"Model RandomForestClassifier diinisialisasi (n_estimators={N_ESTIMATORS}, random_state={RANDOM_STATE}).")

        print("Melatih model...")
        model.fit(X_train, y_train)
        print("Model selesai dilatih.")

        print("Melakukan prediksi pada test set...")
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print("Metrik Evaluasi pada Test Set:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")

        # Log metrics
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)

        # Explicitly log model artifact so mlflow models build-docker can target "runs:/<run_id>/model"
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", registered_model_name=None)
        print("Model disimpan ke artifact path 'model'.")

        # Save run id to file for CI to pick up
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        # Optionally save a small run summary
        summary = {
            "run_id": run_id,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        with open("run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Run selesai. run_id disimpan di run_id.txt dan ringkasan di run_summary.json.")

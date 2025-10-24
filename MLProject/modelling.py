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
    # Definisikan argumen sesuai dengan parameter di MLproject
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--experiment_name", type=str, default="Loan Approval CI") # Tetap ada untuk fleksibilitas
    parser.add_argument("--run_name", type=str, default="CI Run") # Tetap ada untuk fleksibilitas
    args = parser.parse_args()

    N_ESTIMATORS = args.n_estimators
    RANDOM_STATE = args.random_state
    # Nama Eksperimen dan Run akan diatur oleh perintah `mlflow run` di ci.yml

    # Definisikan path ke data (relatif terhadap MLproject file)
    TRAIN_DATA_PATH = os.path.join("loan_approval_preprocessing", "loan_approval_train_preprocessing.csv")
    TEST_DATA_PATH = os.path.join("loan_approval_preprocessing", "loan_approval_test_preprocessing.csv")

    train_df, test_df = load_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    if train_df is None or test_df is None:
        exit(1) # Keluar dengan kode error jika data gagal

    target = 'loan_approved'
    try:
        # Pisahkan Fitur (X) dan Target (y)
        X_train = train_df.drop(columns=[target])
        y_train = train_df[target]
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target]
        print("Fitur dan target berhasil dipisahkan.")
    except KeyError:
        print(f"Error: Kolom target '{target}' tidak ditemukan.")
        exit(1) # Keluar dengan kode error
    except Exception as e:
        print(f"Error saat memisahkan fitur/target: {e}")
        exit(1) # Keluar dengan kode error

    # Aktifkan MLflow Autologging
    # Ini akan otomatis log ke run yang dibuat oleh `mlflow run`
    # Nama eksperimen dari `mlflow run` akan otomatis digunakan
    mlflow.sklearn.autolog(
        log_model_signatures=True,
        log_input_examples=True,
        log_models=True, # Pastikan ini True agar artifact model dibuat
        disable=False
    )
    print("MLflow autolog diaktifkan.")

    # Cek run aktif (opsional, untuk logging)
    active_run = mlflow.active_run()
    if active_run:
        run_id = active_run.info.run_id
        print(f"Logging ke active run: {run_id}")
    else:
        # Seharusnya tidak terjadi jika dijalankan via `mlflow run`
        print("Warning: Tidak ada active run MLflow. Autolog mungkin membuat run baru.")

    # --- Kode training sekarang di luar 'with' block ---
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    print(f"Model RandomForestClassifier diinisialisasi (n_estimators={N_ESTIMATORS}, random_state={RANDOM_STATE}).")

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

    # Log metrik test secara manual (opsional, tapi memastikan tercatat)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)

    print(f"Run selesai. Autolog telah mencatat hasilnya.")
    print("Hasil run tersimpan di folder 'mlruns'.")
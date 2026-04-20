# dags/monitoring_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def check_model_drift(**kwargs):
    """So sánh feature distribution hiện tại vs lúc train"""
    import numpy as np
    from scipy import stats

    X_train = np.load("/opt/airflow/data/processed/X_train.npy")
    # Giả lập production data (thực tế: query từ DB/log)
    X_prod = X_train + np.random.normal(0, 0.1, X_train.shape)

    drift_detected = False
    for i in range(X_train.shape[1]):
        stat, p_value = stats.ks_2samp(X_train[:, i], X_prod[:, i])
        if p_value < 0.05:
            print(f"⚠️ Feature {i}: drift detected (p={p_value:.4f})")
            drift_detected = True

    kwargs['ti'].xcom_push(key='drift', value=drift_detected)
    if not drift_detected:
        print("✅ No significant drift detected")

def check_prediction_quality(**kwargs):
    """Kiểm tra model accuracy trên batch mới"""
    import numpy as np, mlflow, mlflow.sklearn
    from sklearn.metrics import f1_score

    mlflow.set_tracking_uri("http://mlflow:5000")
    model = mlflow.sklearn.load_model("models:/FraudDetector/Production")

    X_test = np.load("/opt/airflow/data/processed/X_test.npy")
    y_test = np.load("/opt/airflow/data/processed/y_test.npy")

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"📊 Current F1 on holdout: {f1:.4f}")

    if f1 < 0.80:
        raise ValueError(f"❌ Model degraded! F1={f1:.4f} — cần retrain!")

with DAG(
    dag_id="monitoring_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["lab6", "monitoring"],
) as dag:

    t1 = PythonOperator(task_id="check_drift", python_callable=check_model_drift)
    t2 = PythonOperator(task_id="check_quality", python_callable=check_prediction_quality)

    t1 >> t2
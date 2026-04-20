# dags/training_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def train_model(**kwargs):
    import numpy as np
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Tạo experiment với artifact location là HTTP proxy
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("fraud-detection")
    if exp is None:
        mlflow.create_experiment(
            "fraud-detection",
            artifact_location="mlflow-artifacts:/fraud-detection"
        )
    mlflow.set_experiment("fraud-detection")

    X_train = np.load("/opt/airflow/data/processed/X_train.npy")
    X_test  = np.load("/opt/airflow/data/processed/X_test.npy")
    y_train = np.load("/opt/airflow/data/processed/y_train.npy")
    y_test  = np.load("/opt/airflow/data/processed/y_test.npy")

    with mlflow.start_run() as run:
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        mlflow.log_params(params)

        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        metrics = {
            "f1":        f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(clf, "model", registered_model_name="FraudDetector")

        print(f"✅ Run ID: {run.info.run_id}")
        print(f"   F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}")
        kwargs['ti'].xcom_push(key='run_id', value=run.info.run_id)

def promote_model(**kwargs):
    """Promote model lên Production nếu F1 đủ tốt"""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()

    run_id = kwargs['ti'].xcom_pull(task_ids='train_model', key='run_id')
    run = client.get_run(run_id)
    f1 = run.data.metrics.get("f1", 0)

    if f1 >= 0.85:
        # Lấy version mới nhất
        versions = client.get_latest_versions("FraudDetector", stages=["None"])
        if versions:
            client.transition_model_version_stage(
                name="FraudDetector",
                version=versions[0].version,
                stage="Production"
            )
            print(f"✅ Model v{versions[0].version} promoted — F1={f1:.4f}")
    else:
        raise ValueError(f"❌ F1={f1:.4f} < 0.85, không promote!")

with DAG(
    dag_id="training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
    tags=["lab6", "training"],
) as dag:

    t1 = PythonOperator(task_id="train_model", python_callable=train_model)
    t2 = PythonOperator(task_id="promote_model", python_callable=promote_model)

    t1 >> t2
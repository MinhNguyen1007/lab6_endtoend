# dags/data_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import os

def download_data(**kwargs):
    """Giả lập download hoặc copy data vào thư mục working"""
    src = "/opt/airflow/data/creditcard.csv"
    dst = "/opt/airflow/data/raw/creditcard.csv"
    os.makedirs("/opt/airflow/data/raw", exist_ok=True)
    
    df = pd.read_csv(src)
    df.to_csv(dst, index=False)
    print(f"✅ Loaded {len(df)} rows")
    kwargs['ti'].xcom_push(key='row_count', value=len(df))

def preprocess_data(**kwargs):
    """Xử lý: scale, split train/test"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    df = pd.read_csv("/opt/airflow/data/raw/creditcard.csv")
    
    # Fraud detection: imbalanced → undersample majority
    fraud = df[df['Class'] == 1]
    normal = df[df['Class'] == 0].sample(n=len(fraud) * 10, random_state=42)
    df_balanced = pd.concat([fraud, normal]).sample(frac=1, random_state=42)
    
    X = df_balanced.drop('Class', axis=1)
    y = df_balanced['Class']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    os.makedirs("/opt/airflow/data/processed", exist_ok=True)
    np.save("/opt/airflow/data/processed/X_train.npy", X_train)
    np.save("/opt/airflow/data/processed/X_test.npy", X_test)
    np.save("/opt/airflow/data/processed/y_train.npy", y_train)
    np.save("/opt/airflow/data/processed/y_test.npy", y_test)
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

def validate_data(**kwargs):
    """Kiểm tra data quality đơn giản"""
    import numpy as np
    X_train = np.load("/opt/airflow/data/processed/X_train.npy")
    assert not pd.isnull(X_train).any(), "❌ NaN detected!"
    assert X_train.shape[1] == 30, "❌ Wrong feature count!"
    print(f"✅ Data valid — shape: {X_train.shape}")

with DAG(
    dag_id="data_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["lab6", "data"],
) as dag:

    t1 = PythonOperator(task_id="download_data", python_callable=download_data)
    t2 = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data)
    t3 = PythonOperator(task_id="validate_data", python_callable=validate_data)

    t1 >> t2 >> t3
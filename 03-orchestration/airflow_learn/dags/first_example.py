from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def say_hello():
    print("Hello, Airflow!")

with DAG(
    dag_id="hello_world",
    start_date=datetime(2025, 6, 11),
    schedule="@once",  # ✅ New in Airflow 2.8+
    catchup=False
) as dag:

    hello = PythonOperator(
        task_id="say_hello",
        python_callable=say_hello
    )

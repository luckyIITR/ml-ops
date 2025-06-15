from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime

def say_hello():
    print("Hello, Airflow!")

with DAG(
    dag_id="hello_world",
    start_date=datetime(2025, 6, 11),
    schedule="@once",  # ✅ New in Airflow 2.8+
    catchup=False
) as dag:
    # t1, t2 and t3 are examples of tasks created by instantiating operators
    t1 = BashOperator(
        task_id="print_date",
        bash_command="date",
    )
    
    t2 = BashOperator(
        task_id="sleep",
        depends_on_past=False,
        bash_command="sleep 5",
        retries=3,
    )
    
    hello = PythonOperator(
        task_id="say_hello",
        python_callable=say_hello
    )

    t1 >> [t2, hello]
## ✅ Project Directory Structure

```bash
ml_project_airflow/
├── dags/
│   └── ml_pipeline_dag.py             # Your DAG file with Airflow logic
│
├── include/                           # Additional config, SQLs, small data
│   └── example_config.json
│
├── scripts/                           # Python modules for task logic
│   ├── data_loader.py
│   ├── preprocess.py
│   └── train_model.py
│
├── models/                            # Saved model artifacts
│   └── model.b
│
├── config/
│   └── ml_config.yaml                 # ML-specific configuration
│
├── logs/                              # Airflow DAG/task logs
│
├── requirements.txt
├── README.md
└── .env                               # Secrets like MLflow URI, AWS keys

```

# Airflow DAG (Example Overview)

In dags/ml_pipeline_dag.py:
```python
from airflow import DAG
from airflow.decorators import task
from datetime import datetime
from scripts.data_loader import load_data
from scripts.preprocess import preprocess_data
from scripts.train_model import train

default_args = {"owner": "airflow", "start_date": datetime(2024, 1, 1)}

with DAG("ml_pipeline", schedule_interval="@daily", default_args=default_args, catchup=False) as dag:

    @task()
    def load():
        return load_data()

    @task()
    def preprocess(raw_data):
        return preprocess_data(raw_data)

    @task()
    def train_model(cleaned_data):
        return train(cleaned_data)

    raw = load()
    cleaned = preprocess(raw)
    train_model(cleaned)

```

### Tips
- Use XCom to pass serialized data (e.g., JSON, paths) between tasks.

- Avoid heavy DataFrames in XCom; prefer writing to file and passing paths.

- Use mlflow in training script to track metrics and model versions.

- For production, use task groups or @task_group to structure the DAG better.

- Always version your models and preprocessing logic.

### Note: heavy DataFrames are not suitable for Airflow's XCom, which is designed for passing small metadata (under 48 KB by default). When working with large DataFrames, here’s a scalable and production-ready solution:

# Best Practice: Pass File Paths Instead of DataFrames
Rather than passing the entire DataFrame, you should:

1. Save the DataFrame to disk (Parquet/CSV/Pickle) in a shared location (e.g., data/ folder, S3, GCS).

2. Return the path to that file from the task.

3. Downstream tasks read the DataFrame by loading it from the file.


```scripts/data_loader.py```:
```python
import pandas as pd

def load_data(output_path):
    df = pd.read_csv("https://your-source.com/data.csv")
    df.to_parquet(output_path, index=False)
    return output_path  # Return path for next task
```

```scripts/preprocess.py```:
```python
import pandas as pd

def preprocess_data(input_path, output_path):
    df = pd.read_parquet(input_path)
    df = df.dropna()
    df.to_parquet(output_path, index=False)
    return output_path
```

```scripts/train_model.py```:
```python
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.metrics import mean_squared_error

def train_model(data_path, model_path):
    df = pd.read_parquet(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train({"objective": "reg:squarederror"}, dtrain, num_boost_round=50)

    with open(model_path, "wb") as f:
        pickle.dump(booster, f)

    preds = booster.predict(dtrain)
    rmse = mean_squared_error(y, preds, squared=False)
    return rmse
```


```dags/ml_pipeline_dag.py```:
```python
from airflow import DAG
from airflow.decorators import task
from datetime import datetime
from scripts.data_loader import load_data
from scripts.preprocess import preprocess_data
from scripts.train_model import train_model

default_args = {"owner": "airflow", "start_date": datetime(2024, 1, 1)}

with DAG("ml_pipeline", schedule_interval=None, default_args=default_args, catchup=False) as dag:

    @task()
    def load():
        return load_data("data/raw_data.parquet")

    @task()
    def preprocess(input_path: str):
        return preprocess_data(input_path, "data/clean_data.parquet")

    @task()
    def train(cleaned_path: str):
        return train_model(cleaned_path, "models/model.b")

    raw_path = load()
    cleaned_path = preprocess(raw_path)
    train(cleaned_path)
```

#### Recap:
| Problem          | Solution                                                  |
| ---------------- | --------------------------------------------------------- |
| Heavy DataFrame  | Save to Parquet; pass file path via XCom                  |
| Parallel runs    | Use dynamic file naming if needed (`f"{ds}_raw.parquet"`) |
| Model versioning | Store model in versioned path: `models/model_{ds}.b`      |
| Performance      | Use `.feather` or `.parquet` over CSV for speed           |

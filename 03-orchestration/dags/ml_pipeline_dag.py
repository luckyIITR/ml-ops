from airflow import DAG
from airflow.decorators import task

from datetime import datetime
import pickle

from scripts.xg_boost_pipeline import read_dataframe, preprocess_data, train_model

default_args = {"owner": "airflow", "start_date": datetime(2025, 1, 1)}

with DAG("ml_pipeline", schedule="@once",  # ✅ New in Airflow 2.8+, 
         default_args=default_args, catchup=False) as dag:

    @task()
    def load():
        train_data_path = read_dataframe(year=2021, month=1, output_path="data/green_tripdata_2021-01.parquet")
        val_data_path = read_dataframe(year=2021, month=2, output_path="data/green_tripdata_2021-02.parquet")
        return {"train_data_path": train_data_path, "val_data_path": val_data_path}

    @task()
    def preprocess(paths):
        train_data_path = paths["train_data_path"]
        val_data_path = paths["val_data_path"]
        # Read the dataframes
        preprocess_data(df_path_train=train_data_path, df_path_val=val_data_path, output_path_x_train="data/X_train.pkl", output_path_x_val="data/X_val.pkl", output_path_y_train="data/y_train.pkl", output_path_y_val="data/y_val.pkl", output_path_dv="data/dv.pkl")
        
        return {
            "x_train_path": "data/X_train.pkl",
            "x_val_path": "data/X_val.pkl",
            "dv_path": "data/dv.pkl",
            "y_train_path": "data/y_train.pkl",
            "y_val_path": "data/y_val.pkl"
        }
        
    @task()
    def train_model_pipeline(cleaned_data_paths):
        x_train_path = cleaned_data_paths["x_train_path"]
        x_val_path = cleaned_data_paths["x_val_path"]
        dv_path = cleaned_data_paths["dv_path"]
        y_train_path = cleaned_data_paths["y_train_path"]
        y_val_path = cleaned_data_paths["y_val_path"]
        # Load the preprocessed data
        with open(x_train_path, "rb") as f:
            X_train = pickle.load(f)
        with open(x_val_path, "rb") as f:
            X_val = pickle.load(f)
        with open(dv_path, "rb") as f:
            dv = pickle.load(f)
        with open(y_train_path, "rb") as f:
            y_train = pickle.load(f)
        with open(y_val_path, "rb") as f:
            y_val = pickle.load(f)
            
        # Train the model
        train_model(X_train, y_train, X_val, y_val, dv)
        
        

    paths = load()
    cleaned_data_paths = preprocess(paths)
    train_model_pipeline(cleaned_data_paths)
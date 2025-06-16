import pickle
from pathlib import Path

import pandas as pd


from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

def read_dataframe(year, month, output_path):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    
    df.to_parquet(output_path, index=False)
    return output_path  # Return path for next task


def preprocess_data(df_path_train, df_path_val, output_path_x_train, output_path_x_val, output_path_y_train, output_path_y_val, output_path_dv):
    """
    Preprocess the data and save the transformed features and target variables.
    """

    df_train = pd.read_parquet(df_path_train)
    df_val = pd.read_parquet(df_path_val)
    
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts_train = df_train[categorical + numerical].to_dict(orient='records')
    dicts_val = df_val[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(dicts_train)
    X_val = dv.transform(dicts_val)
    
    y_train = df_train['duration'].values
    y_val = df_val['duration'].values
    
    # Save both objects
    with open(output_path_x_train, "wb") as f_out:
        pickle.dump(X_train, f_out)

    with open(output_path_x_val, "wb") as f_out:
        pickle.dump(X_val, f_out)

    with open(output_path_dv, "wb") as f_out:
        pickle.dump(dv, f_out)

    with open(output_path_y_train, "wb") as f_out:
        pickle.dump(y_train, f_out)

    with open(output_path_y_val, "wb") as f_out:
        pickle.dump(y_val, f_out)

    return output_path_x_train, output_path_x_val, output_path_y_train, output_path_y_val, output_path_dv


def train_model(X_train, y_train, X_val, y_val, dv):
    import xgboost as xgb
    import mlflow
    import os
    from dotenv import load_dotenv

    load_dotenv()

    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") # fill in with your AWS profile. More info: https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/setup.html#setup-credentials
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv('AWS_SECRET_ACCESS_KEY') # fill in with your AWS profile. More info: https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/setup.html#setup-credentials
    TRACKING_SERVER_HOST = os.getenv('TRACKING_SERVER_HOST') # fill in with the public DNS of the EC2 instance
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

    print(f"tracking URI: '{mlflow.get_tracking_uri()}'")
    mlflow.set_experiment("nyc-duration")
    
    with mlflow.start_run() as run:
        mlflow.autolog()
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }


        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
            
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        return run.info.run_id


def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    # run_id = train_model(X_train, y_train, X_val, y_val, dv)
    # print(f"MLflow run_id: {run_id}")
    # return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    with open("run_id.txt", "w") as f:
        f.write(run_id)
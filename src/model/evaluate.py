import mlflow
from pyspark.sql import DataFrame
from src.data import preprocess
from src.utils import feature_label
from sklearn.metrics import root_mean_squared_error, r2_score
from src.config import config
from src.data import load, preprocess
from src.utils import spark_utils
import os 
import pandas as pd



def load_model_safe(model_name, version_id=None, alias=None):
    
    # set tracking uri
    os.environ["DATABRICKS_HOST"] = config.databricks_host
    os.environ["DATABRICKS_TOKEN"] = config.databricks_token
    mlflow.set_tracking_uri("databricks")
    
    # 2. Initialize client
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(config.experiment_name)
    
    # 3. get model uri through run_id
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    model_uri = f"runs:/{run_id}/teen_phone_addiction_predictor"
    print("model_uri:", model_uri)

    # 4. Load model
    try:
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"Failed to load model '{model_name}' at {model_uri}: {e}")
        return None


def evaluate_model(model_name, evaluate_spark_df: DataFrame, version_id=None, alias=None) -> bool:
    model = load_model_safe(model_name, version_id, alias)
    if not model:
        print("could not load the model from databricks.")
    
    # when using pyspark.DataFrame
    # transformed_df = preprocess.preprocess_spark_dataframe(evaluate_spark_df)
    # df_val = transformed_df.select(feature_label.FEATURE_COLS + [feature_label.LABEL_COL]).toPandas()
    
    # when using pandas DataFrame
    data_dir = os.path.dirname(os.path.realpath("__file__")) + "/src/data"
    df = pd.read_csv(data_dir + "/teen_phone_addiction_dataset.csv")
    df_processed = preprocess.preprocess_pd_dataset(df)
    df_val = df_processed.sample(n=10)
    
    y_val = df_val[feature_label.LABEL_COL]
    X_val = df_val[feature_label.FEATURE_COLS]
    val_predictions = model.predict(X_val)

    # Metrics
    rmse_val = root_mean_squared_error(y_val, val_predictions)
    r2_val = r2_score(y_val, val_predictions)

    # metrics to be saved
    metrics = {
        "rmse_val":   rmse_val,
        "r2_val":     r2_val,
    }
    
    return metrics

if __name__ == "__main__":
    raw_df = load.load_data(spark_utils.spark)
    _, _, test_df = load.split_raw_data(raw_df)
    
    metrics = evaluate_model(f"{config.catalog_name}.{config.schema_name}.{config.model_name}", test_df)
    print("Evaluation Result on evaluate dataset:", metrics)
    
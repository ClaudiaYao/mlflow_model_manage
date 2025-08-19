
import requests
import pandas as pd
import json
from src.config import config
from src.utils import feature_label, spark_utils
from src.data import preprocess, load 
from sklearn.metrics import root_mean_squared_error, r2_score

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = f'https://dbc-dc1a5663-c463.cloud.databricks.com/serving-endpoints/{config.endpoint_name}/invocations'
    headers = {'Authorization': f'Bearer {config.databricks_token}', 'Content-Type': 'application/json'}
    
    
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    print(f"Calling endpoint: {url}")
    
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

raw_df = load.load_data(spark_utils.spark)
_, _, test_df = load.split_raw_data(raw_df)
transformed_df = preprocess.create_preprocessing_pipeline(test_df)

df = transformed_df.select(feature_label.FEATURE_COLS + [feature_label.LABEL_COL]).toPandas()

y_test = df[feature_label.LABEL_COL]
X_test = df.drop(feature_label.LABEL_COL, axis = 1)


res = score_model(X_test)
test_predictions = res['predictions']
# Metrics
rmse_test = root_mean_squared_error(y_test, test_predictions)
r2_test = r2_score(y_test, test_predictions)

# metrics to be saved
metrics = {
    "rmse_val":   rmse_test,
    "r2_val":     r2_test,
}
    
print(metrics)
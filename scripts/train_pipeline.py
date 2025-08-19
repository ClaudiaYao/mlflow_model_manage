
from src.data import load, preprocess
from src.model import train, logging_register
from src.utils import spark_utils
from src.config import config 


# -----------------------------
# 1. Load raw data
# -----------------------------
raw_df = load.load_data(spark_utils.spark)

# -----------------------------
# 2. Preprocessing
# -----------------------------
transformed_df = preprocess.preprocess_spark_dataframe(raw_df)

# -----------------------------
# 4. Train model and get evaluation and test metrics
# -----------------------------
cur_model, parameters, metrics, signature, input_example = train.train_with_model(transformed_df)

# -----------------------------
# 7. Register new model if better
# -----------------------------

logging_register.log_register_model_if_best(cur_model, f"{config.catalog_name}.{config.schema_name}.{config.model_name}", metrics, parameters, signature, input_example)

print("The model training pipeline has been done!")



  
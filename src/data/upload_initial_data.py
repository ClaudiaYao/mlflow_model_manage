
import pandas as pd 
from src.config import config 
from src.utils import spark_utils
import os
from pathlib import Path 


def upload_initial_data():

    # load data
    data_dir = Path.cwd()
    print("data_dir:", data_dir)
    df = pd.read_csv(data_dir / "src/data/teen_phone_addiction_dataset.csv")
    
    spark_df = spark_utils.spark.createDataFrame(df)
    # Save to Databricks
    spark_df.write.mode("overwrite").saveAsTable(f"{config.catalog_name}.{config.schema_name}.{config.table_name}")
    print("the initial data has been uploaded to databricks.")

upload_initial_data()
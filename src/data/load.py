
import os 
from pyspark.sql import SparkSession
from src.config import config 
from pyspark.sql import DataFrame

def load_data(spark: SparkSession):
    df = spark.table(f"{config.catalog_name}.{config.schema_name}.{config.table_name}")
    return df

def split_raw_data(spark_df:DataFrame, seed=42):
    # 4. Preprocess first, then split
    # Here we split the original data into train/val/test AFTER preprocessing
    train_df_spark, val_df_spark, test_df_spark = spark_df.randomSplit([0.7, 0.15, 0.15], seed=seed)
    
    return train_df_spark, val_df_spark, test_df_spark

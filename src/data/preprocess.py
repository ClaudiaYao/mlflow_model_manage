from pyspark.sql.functions import when, regexp_replace, col
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
import mlflow
from src.utils import spark_utils
from src.config import config
import pandas as pd 
from pyspark.sql import DataFrame


# Stage 1: Gender encode
def gender_encode_column(df):
    return df.withColumn(
        "Gender_encode",
        when(col("Gender") == "Female", 0)
        .when(col("Gender") == "Male", 1)
        .otherwise(2)
    )

# Stage 2: School Grade encode
def school_grade_encode_column(df:DataFrame):
    return df.withColumn(
        "School_Grade_encode",
        regexp_replace(col("School_Grade"), "[^0-9]", "").cast("int")
    )


def phone_usage_purpose_column(df: DataFrame):
    return df.withColumn(
        "Phone_Usage_Purpose_encode",
        when(col("Phone_Usage_Purpose") == "Browsing", 0)
        .when(col("Phone_Usage_Purpose") == "Education", 1)
        .when(col("Phone_Usage_Purpose") == "Social Media", 2)
        .when(col("Phone_Usage_Purpose") == "Gaming", 3)
        .when(col("Phone_Usage_Purpose") == "Other", 4)
    )
# ======================
# Preprocessing entrypoints
# ======================

def preprocess_spark_dataframe(df: DataFrame):
    
    df = gender_encode_column(df)
    df = school_grade_encode_column(df)
    transformed_df = phone_usage_purpose_column(df)
    
    # Return both fitted pipeline and transformed Spark DataFrame
    return transformed_df


def preprocess_pd_dataset(data: pd.DataFrame):
    
    def gender_encode(x):
        if x == "Female":
            return 0
        elif x == "Male":
            return 1
        else:
            return 2
        
    def phone_usage_purpose_encode(x):
        if x == "Browsing":
            return 0
        elif x == "Education":
            return 1
        elif x == "Social Media":
            return 2
        elif x == "Gaming":
            return 3
        elif x == "Other":
            return 4
        
    data['Gender_encode'] = data['Gender'].apply(gender_encode).astype("Int32")
    data['School_Grade_encode'] = data['School_Grade'].str[:-2].astype("Int32")
    data['Phone_Usage_Purpose_encode'] = data['Phone_Usage_Purpose'].apply(phone_usage_purpose_encode).astype("Int32")
    
    return data


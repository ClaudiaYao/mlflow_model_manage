from functools import lru_cache
from src.config import config 
from databricks.connect import DatabricksSession

@lru_cache(maxsize=5)
def get_spark_session():
    """Create a singleton SparkSession."""
    print("start connecting ...")
    spark = DatabricksSession.builder \
        .serverless(True) \
        .host(config.databricks_host) \
        .token(config.databricks_token) \
        .getOrCreate()

    
    print("generate spark session")
    return spark


spark = get_spark_session()


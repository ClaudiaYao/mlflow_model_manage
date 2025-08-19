import os
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()
class Config:
    catalog_name = os.getenv("CATALOG_NAME")
    schema_name = os.getenv("SCHEMA_NAME")
    databricks_host = os.getenv("DATABRICKS_HOST")
    databricks_token = os.getenv("DATABRICKS_TOKEN")
    table_name = os.getenv("TABLE_NAME")
    model_name = os.getenv("MODEL_NAME")
    experiment_name = os.getenv("EXPERIMENT_NAME")
    artifact_name = os.getenv("ARTIFACT_NAME")
    endpoint_name = os.getenv("ENDPOINT_NAME")
    env = os.getenv("ENV", "development")  # default to "development"

config = Config()
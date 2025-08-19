import os
from dotenv import load_dotenv
import yaml
from pathlib import Path

def running_on_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

if running_on_databricks():
    project_root = Path.cwd().parent
    os.chdir(project_root)
    print("project_root:", project_root)
else:
    project_root = Path.cwd()
    
with open(project_root / "project_config_teen_addiction.yml") as f:
    yml_config = yaml.safe_load(f)
    
class Config:
    if not running_on_databricks():
        # Load variables from .env into environment
        load_dotenv()
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")
    
    catalog_name = yml_config["CATALOG_NAME"]
    schema_name = yml_config["SCHEMA_NAME"]
    table_name = yml_config["TABLE_NAME"]
    model_name = yml_config["MODEL_NAME"]
    experiment_name = yml_config["EXPERIMENT_NAME"]
    artifact_name = yml_config["ARTIFACT_NAME"]
    endpoint_name = yml_config["ENDPOINT_NAME"]
    env = yml_config["ENV"]
    
    running_on_databricks = running_on_databricks()

config = Config()
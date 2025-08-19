import mlflow
import pandas as pd 
from mlflow.models.signature import ModelSignature
from src.config import config

def log_register_model_if_best(model, model_name, metrics: dict, parameters: dict, signature: ModelSignature, input_example: pd.DataFrame):
    # Get current best model metrics from registry
    mlflow.set_tracking_uri("databricks")
    client = mlflow.tracking.MlflowClient()
    
    try: 
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except Exception as e:
        print(f"{e}")
        print(f"model '{model_name}' does not exist yet. Log and register the current model.")
        log_register_model(model, model_name, metrics, parameters, signature, input_example, register=True)
        return True
    
    if versions:
        current_best_run_id = versions[0].run_id
        current_best_metrics = client.get_run(current_best_run_id).data.metrics
        if metrics["rmse_val"] <= current_best_metrics["rmse_val"]:
            print("Not better than current best model, skipping logging.")
            return False

    # Log and register new model
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        parameters['run_id'] = run_id
        mlflow.log_metrics(metrics)
        mlflow.log_params(parameters)
        
        mlflow.log_input(input_example, context="training")
        mlflow.sklearn.log_model(model, 
                                 name = config.artifact_name,
                                 registered_model_name=model_name,
                                 signature=signature)
        
        return True
    

def log_register_model(model, model_name, metrics: dict, parameters: dict, signature: ModelSignature, input_example: pd.DataFrame, register=True):
    """Register model in Unity Catalog."""
    
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri('databricks-uc')
    client = mlflow.tracking.MlflowClient()
    
    print("🔄 Registering the model in UC...")
    # Log and register new model
    
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        parameters['run_id'] = run_id
        mlflow.log_metrics(metrics)
        mlflow.log_params(parameters)
        
        mlflow.log_input(input_example, context="training")
        if not register:
            model_info = mlflow.sklearn.log_model(model, 
                                    name = config.artifact_name, 
                                    signature=signature)
            print("model is logged.")
            return None
        else:
            model_info = mlflow.sklearn.log_model(model, 
                            name = config.artifact_name,
                            registered_model_name=model_name,
                            signature=signature)

            latest_version = model_info.registered_model_version
            print(f"✅ Model registered as version {latest_version}.")
            print("model_url:", model_info.model_uri)
            
            client.set_registered_model_alias(
                name=model_name,
                alias="latest-model",
                version=latest_version,
            )
            return latest_version
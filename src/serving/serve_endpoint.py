"""Model serving module for Marvel characters."""

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from src.config import config

class ModelServing:
    """Manages model serving in Databricks for Marvel characters."""

    def __init__(self, model_name: str, endpoint_name: str) -> None:
        """Initialize the Model Serving Manager.

        :param model_name: Name of the model to be served
        :param endpoint_name: Name of the serving endpoint
        """
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name
        self.model_name = model_name

    def get_latest_model_version(self) -> str:
        """Retrieve the latest version of the model.

        :return: Latest version of the model as a string
        """
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        print(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(
        self, version: str = "latest-model", workload_size: str = "Small", scale_to_zero: bool = True
    ) -> None:
        """Deploy or update the model serving endpoint in Databricks for Marvel characters.

        :param version: Model version to serve (default: "latest")
        :param workload_size: Size of the serving workload (default: "Small")
        :param scale_to_zero: Whether to enable scale-to-zero (default: True)
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        entity_version = self.get_latest_model_version() if version == "latest-model" else version

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities,
                ),
            )
            print(f"endpoint {config.endpoint_name} is created.")
        else:
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)
            print(f"endpoint {config.endpoint_name} is updated.")

if __name__ == "__main__":

    # You could choose either deploying or update deployed endpoint with Python code or with Databricks UI
    ms = ModelServing(f"{config.catalog_name}.{config.schema_name}.{config.model_name}", config.endpoint_name)
    ms.deploy_or_update_serving_endpoint()
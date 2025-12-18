"""Model serving module for Recommendation System."""

import mlflow
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from src.utilities.config import config
from scripts.workspace_authentication import get_workspace_client


class ModelServing:
    """Manages model serving in Databricks for Recommendation Models."""

    def __init__(self, model_name: str, endpoint_name: str) -> None:
        """
        Initialize the Model Serving Manager.

        :param model_name: Fully-qualified UC model name
        :param endpoint_name: Name of the Databricks serving endpoint
        """
        self.workspace = get_workspace_client()
        self.endpoint_name = endpoint_name
        self.model_name = model_name

    def get_latest_model_version(self) -> str:
        """
        Retrieve the latest model version using the alias 'latest-model'.

        :return: Latest model version as a string
        """
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(
            name=self.model_name,
            alias="latest-model"
        ).version

        print(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(
        self, 
        version: str = "latest-model",
        workload_size: str = "Small",
        scale_to_zero: bool = True
    ) -> None:
        """
        Deploy or update the model serving endpoint in Databricks.

        :param version: Model version or 'latest-model'
        :param workload_size: Size of compute workload
        :param scale_to_zero: Auto scale-to-zero configuration
        """
        endpoint_exists = any(
            item.name == self.endpoint_name 
            for item in self.workspace.serving_endpoints.list()
        )

        # Resolve correct version
        entity_version = (
            self.get_latest_model_version() 
            if version == "latest-model" 
            else version
        )

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,   # UC model path
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(served_entities=served_entities),
            )
            print(f"Serving endpoint '{self.endpoint_name}' created.")
        else:
            self.workspace.serving_endpoints.update_config(
                name=self.endpoint_name,
                served_entities=served_entities
            )
            print(f"Serving endpoint '{self.endpoint_name}' updated.")


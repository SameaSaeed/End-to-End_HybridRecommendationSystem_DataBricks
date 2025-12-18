import requests
import pandas as pd
import json
from src.config import config
from pyspark.sql import DataFrame

def call_endpoint(data: DataFrame | pd.DataFrame | dict):
    """
    Call Databricks serving endpoint for hybrid recommendation model.
    
    Parameters
    ----------
    data : Spark DataFrame, Pandas DataFrame, or dict
        Must contain columns 'als_prediction' and 'cf_prediction'.

    Returns
    -------
    predictions : list
        Hybrid predictions returned by the endpoint.
    """
    url = f"https://{config.databricks_host}/serving-endpoints/{config.endpoint_name}/invocations"

    headers = {
        "Authorization": f"Bearer {config.databricks_token}",
        "Content-Type": "application/json"
    }

    # Convert Spark DataFrame to Pandas
    if isinstance(data, DataFrame):
        df = data.toPandas()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        raise ValueError("data must be Spark DataFrame, Pandas DataFrame, or dict")

    # Ensure required columns exist
    required_cols = ["als_prediction", "cf_prediction"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input must contain columns: {required_cols}")

    # Prepare JSON payload
    payload = {"columns": df.columns.tolist(), "data": df.values.tolist()}
    data_json = json.dumps(payload, allow_nan=True)

    response = requests.post(url, headers=headers, data=data_json)
    if response.status_code != 200:
        raise RuntimeError(f"Endpoint request failed: {response.status_code}, {response.text}")

    return response.json().get("predictions", [])

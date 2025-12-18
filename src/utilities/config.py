import os
from dotenv import load_dotenv
import yaml
from pathlib import Path

def running_on_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

# Determine project root
project_root = Path.cwd().parent if running_on_databricks() else Path.cwd()

# Load config.yaml
with open(project_root / "config.yaml") as f:
    yml_config = yaml.safe_load(f)

# Load paths.yaml
with open(project_root / "src/paths.yaml") as f:
    paths_yaml = yaml.safe_load(f)

# Load .env if running locally
if not running_on_databricks():
    load_dotenv()
    databricks_host = os.getenv("DATABRICKS_HOST")
    databricks_token = os.getenv("DATABRICKS_TOKEN")


class Config:
    def __init__(self):
        # 1️⃣ Load config.yaml variables as attributes
        for k, v in yml_config.items():
            setattr(self, k.lower(), v)

        # 2️⃣ Resolve paths.yaml with placeholders
        self.resolved_paths = {}
        for key, path in paths_yaml.items():
            val = path.format(**{k.upper(): getattr(self, k.lower()) for k in yml_config.keys()})
            # Replace placeholders that reference other paths
            for pk, pv in self.resolved_paths.items():
                val = val.replace(f"{{{pk.upper()}}}", pv)
            self.resolved_paths[key.lower()] = val
            setattr(self, key.lower(), val)

        # Optional: attach databricks credentials if available
        if not running_on_databricks():
            self.databricks_host = databricks_host
            self.databricks_token = databricks_token

        # Flag for environment
        self.running_on_databricks = running_on_databricks()


# Instantiate a single config object
config = Config()

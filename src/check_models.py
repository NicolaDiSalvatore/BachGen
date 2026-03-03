
import os

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# Load environment variables
load_dotenv()

# Set DagsHub URI
# os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/NicolaDiSalvatore/BachFormer.mlflow"
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

client = MlflowClient()

print(f"Checking models at {mlflow.get_tracking_uri()}...")


try:
    # List all registered models
    models = client.search_registered_models()
    if not models:
        print("No registered models found.")
    else:
        print(f"Found {len(models)} registered models:")
        for m in models:
            print(f"- {m.name}")
            # List versions for each model
            versions = client.search_model_versions(f"name='{m.name}'")
            for v in versions:
                print(f"  Version: {v.version}, Stage: {v.current_stage}, Run ID: {v.run_id}")

except Exception as e:
    print(f"Error listing models: {e}")

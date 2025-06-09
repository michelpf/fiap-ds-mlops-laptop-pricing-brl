import mlflow
import json

mlflow.set_tracking_uri("https://dagshub.com/michelpf/fiap-ds-mlops-laptop-pricing-brl.mlflow")
client = mlflow.tracking.MlflowClient()

model_name = "laptop-pricing-model"
latest_model = client.get_latest_versions(model_name, stages=["None"])[0]
run = client.get_run(latest_model.run_id)

metrics = run.data.metrics
params = run.data.params

md = f"""
## 📊 MLflow Report: `{model_name}`

**Run ID**: `{run.info.run_id}`  
**Model Version**: `{latest_model.version}`

### 🔢 Metrics
{chr(10).join([f"- `{k}`: {v}" for k, v in metrics.items()])}

### ⚙️ Parameters
{chr(10).join([f"- `{k}`: {v}" for k, v in params.items()])}
"""

with open("mlflow_report.md", "w") as f:
    f.write(md)
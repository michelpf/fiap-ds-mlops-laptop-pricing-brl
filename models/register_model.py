import os
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

dagshub.init(repo_owner='michelpf', repo_name='fiap-ds-mlops-laptop-pricing-brl', mlflow=True)

client = MlflowClient()
model_name = "laptop-pricing-model"

# Última versão registrada (produção)
registered_versions = sorted(
    client.search_model_versions(f"name='{model_name}'"),
    key=lambda v: int(v.version),
    reverse=True
)

if not registered_versions:
    raise ValueError(f"No registered versions found for model '{model_name}'")

prod_version = registered_versions[0]
prod_metrics = client.get_run(prod_version.run_id).data.metrics

# Último run experimental (mais recente)
all_runs = mlflow.search_runs(search_all_experiments=True, 
                              order_by=["start_time DESC"], 
                              filter_string="metrics.MAPE > 0",
                              max_results=5)
if all_runs.empty:
    raise ValueError("No experimental runs found.")

latest_exp_run_id = all_runs.iloc[0]["run_id"]
latest_exp_run = client.get_run(latest_exp_run_id)
exp_metrics = latest_exp_run.data.metrics

# Função auxiliar
def format_metrics(metrics: dict):
    return "\n".join([f"- `{k}`: {v:.4f}" for k, v in metrics.items()])

# Registro automático do novo modelo, se ainda não registrado
already_registered = any(v.run_id == latest_exp_run_id for v in registered_versions)

summary_lines = []

if not already_registered:
    model_uri = f"runs:/{latest_exp_run_id}/model"
    new_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=latest_exp_run_id
    )
    summary_lines.append(f"✅ **Novo modelo registrado:** versão `{new_version.version}` (run_id={latest_exp_run_id})")
else:
    summary_lines.append("ℹ️ O modelo mais recente **já está registrado**.")

# Construção do relatório para o summary do GitHub Actions
summary = f"""
## 📊 MLflow Report: `{model_name}`

### 🏁 Production Model (Última versão registrada)
- **Run ID**: `{prod_version.run_id}`
- **Model Version**: `{prod_version.version}`

#### 🔢 Métricas de Produção
{format_metrics(prod_metrics)}

---

### 🧪 Último Experimento
- **Run ID**: `{latest_exp_run_id}`

#### 🔢 Métricas do Experimento
{format_metrics(exp_metrics)}

---

### 📈 Comparação
"""

for metric in prod_metrics:
    if metric in exp_metrics:
        delta = exp_metrics[metric] - prod_metrics[metric]
        summary += f"- `{metric}`: Experimento = {exp_metrics[metric]:.4f}, Produção = {prod_metrics[metric]:.4f}, Δ = {delta:+.4f}\n"

summary += "\n---\n" + "\n".join(summary_lines)

# Escreve no summary do GitHub Actions
summary_file = os.getenv("GITHUB_STEP_SUMMARY")
if summary_file:
    with open(summary_file, "a") as f:
        f.write(summary)
else:
    print(summary)
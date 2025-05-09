import dagshub
import mlflow


mlflow.set_tracking_uri("https://dagshub.com/franztac/Water--End-to-End-ML_Proj.mlflow")
dagshub.init(repo_owner="franztac", repo_name="Water--End-to-End-ML_Proj", mlflow=True)

with mlflow.start_run():
    mlflow.log_param("parameter name", "value")
    mlflow.log_metric("metric name", 1)

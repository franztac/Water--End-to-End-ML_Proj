from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import logging
import dagshub
import mlflow
from mlflow.models import infer_signature
import yaml

# Initialize DagsHub for experiment tracking
dagshub.init(repo_owner="franztac", repo_name="Water--End-to-End-ML_Proj", mlflow=True)

# Set the experiment name in MLflow
mlflow.set_experiment("DVC PIPELINE")

# Set the tracking URI for MLflow to log the experiment in DagsHub
mlflow.set_tracking_uri("https://dagshub.com/franztac/Water--End-to-End-ML_Proj.mlflow")


logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s - %(filename)s] %(message)s"
)


def load_data(filepath: str) -> pd.DataFrame:
    try:
        logging.info(f"loading data from {filepath}")
        return pd.read_csv(filepath)
    except Exception as e:
        logging.error(f"error loading data from {filepath}: {e}")
        raise


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        logging.info("Preparing data: separing features from target!")
        X = data.drop(columns=["Potability"])
        y = data.Potability
        return X, y
    except Exception as e:
        logging.error(f"error preparing data: {e}")
        raise


def load_model(filepath: str):
    try:
        logging.info(f"loading model from {filepath}...")
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        logging.error(f"error loading model from {filepath}: {e}")
        raise


def evaluation_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
) -> dict:
    try:
        params = yaml.safe_load(open("params.yaml", "r"))
        test_size = params["data_collection"]["test_size"]
        n_estimators = params["model_building"]["n_estimators"]

        logging.info(f"preparing metrics dict for {model}")
        y_pred = model.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logging.info("preparing report for reports/metrics.json...")

        mlflow.log_param("Test_size", test_size)
        mlflow.log_param("n_estimators", n_estimators)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")
        cm_path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(cm_path)

        # Log confusion matrix artifact
        mlflow.log_artifact(cm_path)

        # Log the model
        # mlflow.sklearn.log_model(model, model_name.replace(" ", "_"))

        metrics_dict = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
        return metrics_dict

    except Exception as e:
        logging.error(f"error evaluating model: {e}")
        raise


def save_metrics(metrics_dict: dict, metrics_path: str) -> None:
    try:
        logging.info(f"saving metrics to {metrics_path}")
        with open(metrics_path, "w") as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        logging.error(f"error saving metrics to {metrics_path}: {e}")
        raise


def main():
    try:
        test_datapath = "./data/processed/test_processed.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"
        model_name = "Best Model"

        test_data = load_data(test_datapath)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)

        # Start MLflow run
        with mlflow.start_run() as run:
            metrics = evaluation_model(model, X_test, y_test, model_name)
            save_metrics(metrics, metrics_path)

            # Log artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)

            # Log the source code file
            mlflow.log_artifact(__file__)

            signature = infer_signature(X_test, model.predict(X_test))
            mlflow.sklearn.log_model(model, "Best Model", signature=signature)

            # Save run ID and model info to JSON File --> helps to "register" a model (step4)
            run_info = {"run_id": run.info.run_id, "model_name": "Best Model"}
            reports_path = "reports/run_info.json"
            with open(reports_path, "w") as file:
                json.dump(run_info, file, indent=4)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()

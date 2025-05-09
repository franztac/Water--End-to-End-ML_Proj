import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml
import logging


logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s - %(filename)s] %(message)s"
)


def load_params(params_path: str) -> int:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
            logging.info(f"params: n_estimators loaded from {params_path}")
            return params["model_building"]["n_estimators"]

    except Exception as e:
        logging.info(f"error loading parameters from {params_path}: {e}")
        raise


def load_data(filepath: str) -> pd.DataFrame:
    try:
        logging.info(f"loading data from {filepath}")
        return pd.read_csv(filepath)
    except Exception as e:
        logging.error(f"error loading data from {filepath}: {e}")
        raise


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        logging.info("preparing data: separating features and target")
        X = data.drop(columns=["Potability"])
        y = data.Potability
        return X, y
    except Exception as e:
        logging.error(f"error preparing data: {e}")
        raise


def train_model(
    X: pd.DataFrame, y: pd.Series, n_estimators: int
) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        logging.info(f"training {clf} model...")
        clf.fit(X, y)
        return clf
    except Exception as e:
        logging.error(f"error training model: {e}")
        raise


def save_model(model: RandomForestClassifier, model_name: str) -> None:
    try:
        with open(model_name, "wb") as file:
            logging.info(f"saving model to {model_name}")
            pickle.dump(model, file)
    except Exception as e:
        logging.error(f"error saving model to {model_name}: {e}")
        raise


def main():
    try:
        params_path = "params.yaml"
        data_path = "./data/processed/train_processed.csv"
        model_name = "models/model.pkl"

        n_estimators = load_params(params_path)
        train_data = load_data(data_path)
        X_train, y_train = prepare_data(train_data)

        model = train_model(X_train, y_train, n_estimators)
        save_model(model, model_name)
        print("Model trained and saved successfully!")
    except Exception as e:
        logging.error(f"an error occurred: {e}")
        raise


if __name__ == "__main__":
    main()

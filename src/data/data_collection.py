import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
from typing import Tuple
import logging


logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s - %(filename)s] %(message)s"
)


def load_data(filepath: str) -> pd.DataFrame:
    try:
        logging.info(f"{filepath} loaded")
        return pd.read_csv(filepath)
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise


# df = pd.read_csv(r"C:/Users/UTENTE/Git Repos/water_potability.csv")


def load_params(filepath: str) -> float:
    try:
        with open(filepath, "r") as file:
            params = yaml.safe_load(file)
            logging.info(f"params: test_size loaded from {filepath}")
            return params["data_collection"]["test_size"]
    except Exception as e:
        logging.error(f"Error loading parameters from {filepath}: {e}")
        raise


# test_size = params["data_collection"]["test_size"]


def split_data(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        logging.info(f"splitting data with test size of {test_size * 100} %")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        return train_df, test_df
    except Exception as e:
        logging.error(f"Error splitting dataset: {e}")
        raise


# train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)


def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        logging.info(f"saving data to {filepath}")
        df.to_csv(filepath, index=False)
    except Exception as e:
        logging.error(f"Error saving data to {filepath}: {e}")
        raise


# train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
# test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)


def main():
    try:
        params_filepath = "params.yaml"
        data_filepath = "C:/Users/UTENTE/Git Repos/water_potability.csv"
        raw_datapath = os.path.join("data", "raw")

        test_size = load_params(params_filepath)
        df = load_data(data_filepath)
        train_df, test_df = split_data(df, test_size=test_size)

        logging.info(f"creating folder {raw_datapath}")
        os.makedirs(raw_datapath, exist_ok=True)

        save_data(train_df, os.path.join(raw_datapath, "train.csv"))
        save_data(test_df, os.path.join(raw_datapath, "test.csv"))

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()

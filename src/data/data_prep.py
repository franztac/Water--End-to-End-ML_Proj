import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s - %(filename)s] %(message)s"
)


def load_data(filepath: str) -> pd.DataFrame:
    try:
        logging.info(f"loading: {filepath}")
        return pd.read_csv(filepath)
    except Exception as e:
        logging.error(f"error loading data from {filepath}: {e}")
        raise


# train_df = pd.read_csv("./data/raw/train.csv")
# test_df = pd.read_csv("./data/raw/test.csv")


def fill_missing_values_with_mean(df_original):
    try:
        logging.info("filling missing values with mean...")
        df = df_original.copy()
        for column in df.columns:
            if df[column].isna().any():
                mean_value = df[column].mean()
                df[column] = df[column].fillna(mean_value)

        return df

    except Exception as e:
        logging.error(f"error filling missing values: {e}")
        raise


def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        logging.info(f"saving data to {filepath}")
        return df.to_csv(filepath, index=False)
    except Exception as e:
        logging.error(f"error saving data to {filepath}: {e}")


# train_processed_df.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
# test_processed_df.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)


def main():
    try:
        raw_datapath = "./data/raw"
        processed_datapath = "./data/processed"

        train_df = load_data(os.path.join(raw_datapath, "train.csv"))
        test_df = load_data(os.path.join(raw_datapath, "test.csv"))

        train_processed_df = fill_missing_values_with_mean(train_df)
        test_processed_df = fill_missing_values_with_mean(test_df)

        logging.info(f"creating {processed_datapath} folder")
        os.makedirs(processed_datapath, exist_ok=True)

        save_data(
            train_processed_df,
            os.path.join(processed_datapath, "train_processed.csv"),
        )
        save_data(
            test_processed_df,
            os.path.join(processed_datapath, "test_processed.csv"),
        )

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()

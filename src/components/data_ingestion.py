import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.custom_exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion initiated.")
        try:
            dataset = pd.read_csv(r"notebook\data\Material_Lifespan_Dataset.csv")
            logging.info("Read the dataset into the pandas data frame")

            # Create a Train Data folder
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            # Copy the data frame as csv file into the RAW folder.
            dataset.to_csv(
                self.ingestion_config.raw_data_path, index=False, header=True
            )

            train_set, test_set = train_test_split(
                dataset, test_size=0.2, random_state=42
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Data Ingestion Completed.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as exp:
            raise CustomException(exp, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_trans_obj = DataTransformation()
    train_arr, test_arr, _ = data_trans_obj.initiate_data_transform(
        train_data_path, test_data_path
    )

    model_trainer_obj = ModelTrainer()
    res = model_trainer_obj.initiate_model_trainer(train_arr, test_arr)

    print(f"Model Result: {res}")

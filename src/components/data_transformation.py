import os
import sys
import pandas as pd
import numpy as np
from src.custom_exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self, df: pd.DataFrame, target_column_name: str):
        """This function is responsible for data trnasformation on numerical and categorical data.

        Args:
            df (pd.DataFrame): Train Dataset to get dataset cols
            target_column_name (str): target column name

        Raises:
            CustomException: Raise Custom Exception if anything raised.

        Returns:
            ColumnTransformer: Preprocessor object
        """
        try:
            logging.info("Getting Data Transformation Object")
            num_cols = df.select_dtypes(exclude="object").columns
            cat_cols = df.select_dtypes(include="object").columns
            num_cols = [col for col in num_cols if col != target_column_name]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder", OneHotEncoder()),
                ]
            )

            logging.info(
                "Preparing the column transformation on numerical and categorical data's"
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_cols),
                    ("cat_pipeline", cat_pipeline, cat_cols),
                ]
            )

            logging.info("Column Tranformation Completed.")

            return preprocessor

        except Exception as exp:
            raise CustomException(exp, sys)

    def initiate_data_transform(self, train_path, test_path):

        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            target_column_name = "PredictedHours"
            preprocessing_obj = self.get_data_transformation_object(
                train_data, target_column_name
            )

            X_train_df = train_data.drop(columns=[target_column_name], axis=1)
            y_train_df = train_data[target_column_name]

            X_test_df = test_data.drop(columns=[target_column_name], axis=1)
            y_test_df = test_data[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            X_train_arr = preprocessing_obj.fit_transform(X_train_df)
            X_test_arr = preprocessing_obj.fit_transform(X_test_df)

            train_arr = np.c_[X_train_arr, np.array(y_train_df)]
            test_arr = np.c_[X_test_arr, np.array(y_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            logging.info("Data Transormation completed.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as exp:
            raise CustomException(exp, sys)

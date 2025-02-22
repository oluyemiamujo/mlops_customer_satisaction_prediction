import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from statistics import median


class DataStrategy(ABC):
    """
    Abstract class defining startegy from handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessingStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            # Drop unnecessary columns
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    # "order_estimated_delivered_date",
                    # "order_purchase_timetamp",
                ],
                axis=1
            )

            # Fill missing values with the median of the respective columns
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)

            # Fill missing review comments with "No review"
            data["review_comment_message"].fillna("No review", inplace=True)

            # Select only numeric columns
            data = data.select_dtypes(include=[np.number])

            # Drop additional unnecessary columns
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data

        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e



class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing the data into train and test
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test
        """
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e

class DataCleaning:
    """
    Class for cleaning data which processes the data and splits into onto traoin and test portion
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        """
        try:
            return self.strategy.handle_data(self.data)

        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e


if __name__ == "__main__":
    data = pd.read_csv("data/olist_customers_dataset.csv")
    data_cleaning = DataCleaning(data, DataPreProcessingStrategy())
    data_cleaning.handle_data()
        
import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessingStrategy
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
    ]:
    
    """
    Data cleaning function. Cleaning the data and divide into train and test 
    
    Args:
      df: the ingetsed raw data.
    
    Returns:
      X_train: training fatures
      X_test: testing fatures
      y_train: training labels
      y_test: testing labels
    """
    try:
        process_strategy  =   DataPreProcessingStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning complete!")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e
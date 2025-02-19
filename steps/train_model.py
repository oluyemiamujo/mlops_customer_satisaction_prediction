import logging

import pandas as pd
from zenml import step

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
# from .config import ModelNameConfig

from .config import model_name




@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    # config: ModelNameConfig,
    ) -> RegressorMixin:
    """
    Train the model on the ingested data.

    Args:
      X_train: pd.DataFrame,
      X_test: pd.DataFrame,
      y_train: pd.Series,
      y_test: pd.Series,

    Returns:
      Trained model
    """
    # model_name = "LinearRegression" 
    try:
      model = None
      if model_name == "LinearRegression":
          model = LinearRegressionModel()
          trained_model = model.train(X_train, y_train)
          logging.info("Model {} trained successfully.".format(model_name))
          return trained_model
      else:
          raise ValueError("Model {} not supported.".format(model_name))
    except Exception as e:
      logging.error("Model {} not supported.".format(model_name))

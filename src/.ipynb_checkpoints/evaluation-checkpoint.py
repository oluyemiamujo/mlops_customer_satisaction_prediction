import logging

from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining the evaluation strategy fro the trained model.
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the performance scores for the model 
        Args:
            y_true: true labels
            y_pred: predicted labels
        Returns:
            None
        """

class MSE(Evaluation):
  """
  Evaluation strategy that uses the Mean Square Error (MSE).
  """

  def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
    try:
      logging.info("Calculate MSE")
      mse = mean_squared_error(y_true, y_pred)
      logging.info(f"MSE: {mse}")
      return mse
    except Exception as e:
      logging.error(f"Error in calculating MSE: {e}")
      raise e

class R2(Evaluation):
  """
  Evaluation strategy that uses the R2 score.
  """
  def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
    try:
      logging.info("Calculate R2 score")
      r2 = r2_score(y_true, y_pred)
      logging.info("R2 Score: {}".format(r2))
      return r2
    except Exception as e:
      logging.error(f"Error in calculating R2 score: {e}")
      raise e

class RMSE(Evaluation):
  """
  Evaluation strategy that uses the Root Mean Square Error (RMSE).
  """
  def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
    try:
      logging.info("Calculate RMSE")
      rmse = mean_squared_error(y_true, y_pred)
      logging.info("RMSE: {}".format(rmse))
      return rmse
    except Exception as e:
      logging.error(f"Error in calculating RMSE: {e}")
      raise e


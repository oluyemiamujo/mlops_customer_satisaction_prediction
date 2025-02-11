from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate import evaluate_model

@pipeline(enable_cache=True)
def train_pipeline(data_path: str) -> None:
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    mse, r2, rmse = evaluate_model(model, X_test, y_test)
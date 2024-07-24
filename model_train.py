import logging
import pandas as pd
from zenml import step

from src.model_dev import LinearRegressionModel

import mlflow
from zenml.client import Client

from config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker = experiment_tracker.name) # type: ignore
def train_model(X_train: pd.DataFrame,
                y_train: pd.Series, 
                config: ModelNameConfig) -> LinearRegressionModel:
    try:
        model = None

        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            model.train(X_train, y_train)

            return model
        else: 
            raise ValueError("Model {} is not supported".format(config.model_name))
        
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise e 
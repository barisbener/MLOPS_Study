import logging 
from zenml import step
import pandas as pd

from src.model_dev import LinearRegressionModel
from src.evaluation import MSE, R2, RMSE

from typing_extensions import Annotated
from typing import Tuple

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker



@step(experiment_tracker = experiment_tracker.name) 

def evaluate_model(model: LinearRegressionModel, 
                   X_test: pd.DataFrame, 
                   y_test: pd.Series) -> Tuple[Annotated[float, "mse"],
                                                  Annotated[float, "r2_score"],
                                                  Annotated[float, "rmse"]]:
    
    try: 
        mse_class = MSE()
        r2_class = R2()
        rmse_class = RMSE()
        
        prediction = model.predict(X_test)

        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("MSE", mse)
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("R2", r2)
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("RMSE", rmse)

        return mse, r2, rmse

    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e 


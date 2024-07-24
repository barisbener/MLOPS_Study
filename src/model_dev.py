import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


class Model(ABC):

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame):
        pass


class LinearRegressionModel(Model):

    def __init__(self): 
        self.model = None

    def train(self, X_train, y_train, **kwargs):
        try:
            self.model = LinearRegression(**kwargs)
            self.model.fit(X_train, y_train)
            logging.info("Model Training Completed")

        except Exception as e:
            logging.error("Error in Training Model: {}".format(e))
            raise e
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet.")
            
            predictions = self.model.predict(X)
            
            return predictions
        except Exception as e:
            logging.error("Error in Making Predictions: {}".format(e))
            raise e
        


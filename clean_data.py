import logging
import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy

from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(df: pd.DataFrame) -> Tuple[Annotated[pd.DataFrame, "X_train"],
                                          Annotated[pd.DataFrame, "X_test"],
                                          Annotated[pd.Series, "y_train"],
                                          Annotated[pd.Series, "y_test"]
]:
    try:
        process_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_df = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning1 = DataCleaning(pd.DataFrame(processed_df), divide_strategy)
        X_train,X_test,y_train,y_test = data_cleaning1.handle_data()

        logging.info("Data Cleaning Completed")

        return X_train,X_test,y_train,y_test  # type: ignore
    
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e


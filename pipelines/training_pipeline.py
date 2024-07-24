from zenml import pipeline
from ingest_data import ingest_data
from clean_data import clean_data
from model_train import train_model
from evaluation import evaluate_model
from config import ModelNameConfig


@pipeline(enable_cache=False)

def train_pipeline(data_path: str):
    
    df = ingest_data(data_path)
    X_tr, X_te, y_tr, y_te = clean_data(df)
    model = train_model(X_tr, y_tr)
    evaluate_model(model, X_te, y_te)



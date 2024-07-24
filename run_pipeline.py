from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri()) # type: ignore
    train_pipeline(data_path = r"C:\Users\baris\OneDrive\Masaüstü\Personal\MLOps\MLOps_Project_Own\data\olist_customers_dataset.csv")



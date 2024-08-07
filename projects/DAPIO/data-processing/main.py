import pandas as pd
from data_processing.preprocessing.data_cleaner import DataCleaner
from data_processing.preprocessing.data_transformer import DataTransformer
from machine_learning.model_trainer import ModelTrainer
from data_storage.data_loader import DataLoader
from data_storage.data_saver import DataSaver

def main():
    # Load data from database
    db_url = "postgresql://user:password@host:port/dbname"
    data_loader = DataLoader(db_url)
    data = data_loader.load_data("my_table")

    # Preprocess data
    data_cleaner = DataCleaner(data)
    data = data_cleaner.preprocess_data()

    data_transformer = DataTransformer(data)
    data = data_transformer.transform_data()

    # Train model
    target = "target_column"
    model_trainer = ModelTrainer(data, target)
    model = model_trainer.train_model()

    # Evaluate model
    model_trainer.evaluate_model(model)

    # Save data and model
    data_saver = DataSaver(data, model)
    data_saver.save_data("preprocessed_data.csv")
    data_saver.save_model("trained_model.joblib")

if __name__ == "__main__":
    main()

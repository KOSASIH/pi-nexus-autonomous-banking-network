from data_loader import load_data
from trainer import train_model

from models import NexusModel

if __name__ == "__main__":
    data = load_data("data.csv")
    model = NexusModel(input_shape=(10,), num_classes=2)
    train_model(model, data)

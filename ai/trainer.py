from data_loader import load_data

from models import NexusModel


def train_model(model: NexusModel, data: pd.DataFrame) -> None:
    """Train the Nexus AI model."""
    X, y = data.drop("target", axis=1), data["target"]
    model.compile()
    model.model.fit(X, y, epochs=10, batch_size=32)

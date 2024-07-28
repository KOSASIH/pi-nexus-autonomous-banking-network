import numpy as np

def detect_threats(model, data):
    # Use the trained model to predict threats in real-time
    predictions = model.predict(data)
    threats = np.where(predictions > 0.5)[0]
    return threats

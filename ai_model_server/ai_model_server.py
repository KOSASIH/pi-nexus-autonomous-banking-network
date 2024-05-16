# ai_model_server.py
import asyncio
import torch
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class DataPoint(BaseModel):
    features: List[float]
    label: float

async def train_model(data: List[DataPoint]):
    # Create a PyTorch neural network model
    model = nn.Sequential(
        nn.Linear(10, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

    # Define a custom loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Scale the data using Scikit-learn's StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform([dp.features for dp in data])

    # Train the model using asynchronous I/O
    async def train_step():
        for epoch in range(100):
            for dp in scaled_data:
                # Perform a forward pass
                output = model(torch.tensor(dp))
                loss = criterion(output, torch.tensor(dp.label))

                # Backpropagate and update the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Yield control to other coroutines
                await asyncio.sleep(0.01)

    # Train the model asynchronously
    await train_step()

    # Return the trained model
    return model

@app.post("/predict")
async def predict(data: List[DataPoint]):
    # Train the model using the provided data
    model = await train_model(data)

    # Make predictions on the provided data
    predictions = []
    for dp in data:
        output = model(torch.tensor(dp.features))
        predictions.append(output.item())

    # Return the predictions
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

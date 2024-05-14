import pytest
from web_app import app

def test_predict():
    # Set up test client
    client = app.test_client()

    # Send POST request with sample data
    response = client.post('/predict', json={'feature1': 1, 'feature2': 2, 'feature3': 3})

    # Check that the response is correct
    assert response.status_code == 200
    assert response.json['prediction'] == 0
